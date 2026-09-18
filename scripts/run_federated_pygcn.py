#!/usr/bin/env python3
"""Federated PyGCN 2 training and centralized-versus-federated comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from run_pygcn2_comparison import (
    PyGCN2,
    load_expression,
    make_arrays,
    projected_edges,
)


def metrics(model, arrays, indices, edge_index, edge_attr, device):
    abundance, protein_features, covariates, labels = [item.to(device) for item in arrays]
    selected = torch.tensor(indices, device=device)
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(
            model(abundance[selected], protein_features, covariates[selected], edge_index, edge_attr)
        ).cpu().numpy()
    truth = labels[selected].cpu().numpy()
    prediction = probability >= 0.5
    return {
        "n": len(indices),
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "f1": f1_score(truth, prediction, zero_division=0),
        "roc_auc": roc_auc_score(truth, probability),
    }


def train_local(global_state, arrays, indices, config, epochs, seed, edge_index, edge_attr, device):
    torch.manual_seed(seed)
    abundance, protein_features, covariates, labels = [item.to(device) for item in arrays]
    model = PyGCN2(protein_features.shape[1], config["hidden_dim"], config["heads"], config["dropout"]).to(device)
    model.load_state_dict(global_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    selected = torch.tensor(indices, device=device)
    positive_weight = (labels[selected] == 0).sum() / (labels[selected] == 1).sum()
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(abundance[selected], protein_features, covariates[selected], edge_index, edge_attr)
        loss = loss_function(logits, labels[selected])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fedavg(states, counts):
    total = sum(counts)
    result = {}
    for key in states[0]:
        result[key] = sum(count * state[key] for state, count in zip(states, counts)) / total
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("proteomics/synthetic_proteomics_chr22"))
    parser.add_argument("--graph-features", type=Path, default=Path("federated_data/graph_protein_features.csv"))
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("proteomics/federated_pyg_results"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    expression, metadata = load_expression(args.data_dir)
    graph_features = pd.read_csv(args.graph_features).set_index("protein_id")
    proteins = sorted(set(expression.columns) & set(graph_features.index))
    edge_index, edge_attr = projected_edges(args.edges, proteins)
    arrays = make_arrays(expression, metadata, proteins, graph_features)
    labels = metadata.loc[expression.index, "phenotype"].to_numpy()
    strata = metadata.loc[expression.index, "site"].to_numpy() + "_" + labels.astype(str)
    all_indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=args.seed, stratify=strata)
    sites = metadata.loc[expression.index, "site"].to_numpy()
    site_train = {site: train_idx[sites[train_idx] == site] for site in sorted(set(sites))}
    config = {"hidden_dim": 32, "heads": 4, "dropout": 0.20, "lr": 0.001, "weight_decay": 1e-5}
    template = PyGCN2(arrays[1].shape[1], config["hidden_dim"], config["heads"], config["dropout"])
    global_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)
    rounds = []
    for round_number in range(1, args.rounds + 1):
        local_states, counts = [], []
        for offset, site in enumerate(sorted(site_train)):
            state = train_local(global_state, arrays, site_train[site], config, args.local_epochs,
                                args.seed + round_number * 10 + offset, edge_index, edge_attr, device)
            local_states.append(state)
            counts.append(len(site_train[site]))
        global_state = fedavg(local_states, counts)
        model = PyGCN2(arrays[1].shape[1], config["hidden_dim"], config["heads"], config["dropout"]).to(device)
        model.load_state_dict(global_state)
        result = metrics(model, arrays, test_idx, edge_index, edge_attr, device)
        result["round"] = round_number
        rounds.append(result)
    final_model = PyGCN2(arrays[1].shape[1], config["hidden_dim"], config["heads"], config["dropout"]).to(device)
    final_model.load_state_dict(global_state)
    test_metrics = metrics(final_model, arrays, test_idx, edge_index, edge_attr, device)
    site_metrics = []
    for site in sorted(site_train):
        site_test = test_idx[sites[test_idx] == site]
        site_metrics.append({"site": site, **metrics(final_model, arrays, site_test, edge_index, edge_attr, device)})
    pd.DataFrame(rounds).to_csv(args.output_dir / "round_metrics.csv", index=False)
    pd.DataFrame(site_metrics).to_csv(args.output_dir / "site_test_metrics.csv", index=False)
    summary = {
        "model": "Federated PyGCN 2",
        "device": str(device),
        "n_samples": len(labels),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_proteins": len(proteins),
        "projected_undirected_edges": int(edge_index.shape[1] // 2),
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "test_metrics": test_metrics,
        "site_test_metrics": site_metrics,
        "config": config,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
