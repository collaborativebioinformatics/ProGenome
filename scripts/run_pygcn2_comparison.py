#!/usr/bin/env python3
"""Train an attention- and edge-aware PyG model on the complete protein graph."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch_geometric.nn import GATv2Conv


def load_expression(data_dir: Path):
    metadata = pd.read_csv(data_dir / "sample_metadata.csv").set_index("sample_id")
    metadata["site"] = metadata.index.to_series().str.split("_", n=1).str[0].str.lower()
    matrices = []
    for path in sorted(data_dir.glob("site*_proteomics_log2.csv")):
        matrix = pd.read_csv(path).set_index("protein_id")
        matrices.append(matrix.drop(columns=["gene_symbol"]).T)
    return pd.concat(matrices).loc[metadata.index], metadata


def projected_edges(edge_path: Path, proteins: list[str]):
    protein_set = set(proteins)
    opener = gzip.open if edge_path.suffix == ".gz" else open
    table = pd.read_csv(opener(edge_path, "rt"), usecols=["source_protein", "target_protein", "weight", "lift"])
    pairs: dict[tuple[str, str], list[float]] = {}
    for row in table.itertuples(index=False):
        source = [p for p in str(row.source_protein).split(";") if p in protein_set]
        target = [p for p in str(row.target_protein).split(";") if p in protein_set]
        for left in source:
            for right in target:
                if left != right:
                    item = pairs.setdefault(tuple(sorted((left, right))), [0.0, 0.0])
                    item[0] += float(row.weight)
                    item[1] += float(row.lift)
    index = {protein: i for i, protein in enumerate(proteins)}
    edges, attributes = [], []
    for (left, right), (weight, lift) in pairs.items():
        for source, target in ((left, right), (right, left)):
            edges.append((index[source], index[target]))
            attributes.append((weight, lift))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attributes, dtype=torch.float32)
    edge_attr = (edge_attr - edge_attr.mean(dim=0)) / edge_attr.std(dim=0).clamp_min(1e-6)
    return edge_index, edge_attr


def make_arrays(expression, metadata, proteins, graph_features):
    abundance = expression[proteins].to_numpy(dtype=np.float32)
    abundance = np.nan_to_num(abundance, nan=np.nanmedian(abundance, axis=0))
    abundance = (abundance - abundance.mean(axis=0)) / np.where(abundance.std(axis=0) == 0, 1, abundance.std(axis=0))
    fixed = graph_features.loc[proteins, ["graph_degree", "graph_weight", "graph_lift"]]
    fixed = (fixed - fixed.mean()) / fixed.std().replace(0, 1)
    protein_features = torch.tensor(fixed.fillna(0).to_numpy(dtype=np.float32))
    covariates = torch.tensor(
        np.column_stack([
            metadata.loc[expression.index, "age"].to_numpy() / 85.0,
            metadata.loc[expression.index, "sex"].to_numpy(),
            (metadata.loc[expression.index, "site"] == "site2").to_numpy(),
            (metadata.loc[expression.index, "site"] == "site3").to_numpy(),
        ]), dtype=torch.float32,
    )
    labels = torch.tensor(metadata.loc[expression.index, "phenotype"].to_numpy(), dtype=torch.float32)
    return torch.tensor(abundance), protein_features, covariates, labels


class PyGCN2(nn.Module):
    def __init__(self, protein_feature_dim, hidden_dim, heads, dropout):
        super().__init__()
        self.dropout = dropout
        self.attention = GATv2Conv(protein_feature_dim, hidden_dim, heads=heads,
                                   concat=True, edge_dim=2, dropout=dropout)
        self.refinement = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1,
                                    concat=False, edge_dim=2, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, abundance, protein_features, covariates, edge_index, edge_attr):
        hidden = torch.relu(self.attention(protein_features, edge_index, edge_attr))
        hidden = torch.dropout(hidden, self.dropout, self.training)
        hidden = torch.relu(self.refinement(hidden, edge_index, edge_attr))
        pooled = abundance @ hidden / abundance.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        return self.classifier(torch.cat([pooled, covariates], dim=1)).squeeze(1)


def score(probabilities, labels):
    predictions = probabilities >= 0.5
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
    }


def fit(arrays, indices, validation, config, epochs, patience, seed, device, edge_index, edge_attr):
    torch.manual_seed(seed)
    abundance, protein_features, covariates, labels = [item.to(device) for item in arrays]
    model = PyGCN2(protein_features.shape[1], config["hidden_dim"], config["heads"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    positive_weight = (labels[indices] == 0).sum() / (labels[indices] == 1).sum()
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    best_score, best_state, stale = -np.inf, None, 0
    train_idx = torch.tensor(indices, device=device)
    valid_idx = torch.tensor(validation, device=device)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(abundance[train_idx], protein_features, covariates[train_idx], edge_index, edge_attr)
        loss = loss_function(logits, labels[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_prob = torch.sigmoid(model(abundance[valid_idx], protein_features, covariates[valid_idx], edge_index, edge_attr))
        current = roc_auc_score(labels[valid_idx].cpu().numpy(), valid_prob.cpu().numpy())
        if current > best_score:
            best_score = current
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(abundance[valid_idx], protein_features, covariates[valid_idx], edge_index, edge_attr))
    return model, score(probabilities.cpu().numpy(), labels[valid_idx].cpu().numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("proteomics/synthetic_proteomics_chr22"))
    parser.add_argument("--graph-features", type=Path, default=Path("federated_data/graph_protein_features.csv"))
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("proteomics/pygcn2_results"))
    parser.add_argument("--tuning-epochs", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    expression, metadata = load_expression(args.data_dir)
    graph_features = pd.read_csv(args.graph_features).set_index("protein_id")
    proteins = sorted(set(expression.columns) & set(graph_features.index))
    edge_index, edge_attr = projected_edges(args.edges, proteins)
    arrays = make_arrays(expression, metadata, proteins, graph_features)
    arrays = (*arrays[:1], arrays[1], arrays[2], arrays[3])
    edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)
    labels = metadata.loc[expression.index, "phenotype"].to_numpy()
    strata = metadata.loc[expression.index, "site"].to_numpy() + "_" + labels.astype(str)
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=args.seed, stratify=strata)
    tune_train, tune_valid = train_test_split(train_idx, test_size=0.2, random_state=args.seed, stratify=labels[train_idx])
    configs = [
        {"hidden_dim": 16, "heads": 2, "dropout": 0.20, "lr": 0.003, "weight_decay": 1e-4},
        {"hidden_dim": 24, "heads": 2, "dropout": 0.35, "lr": 0.001, "weight_decay": 1e-4},
        {"hidden_dim": 32, "heads": 4, "dropout": 0.20, "lr": 0.001, "weight_decay": 1e-5},
    ]
    tuning = []
    for number, config in enumerate(configs, 1):
        _, metrics = fit(arrays, tune_train, tune_valid, config, args.tuning_epochs, args.patience,
                         args.seed + number, device, edge_index, edge_attr)
        tuning.append({**config, "configuration": number, **metrics})
    tuning_frame = pd.DataFrame(tuning).sort_values("roc_auc", ascending=False)
    tuning_frame.to_csv(args.output_dir / "hyperparameter_search.csv", index=False)
    best = configs[int(tuning_frame.iloc[0]["configuration"]) - 1]
    _, test_metrics = fit(arrays, train_idx, test_idx, best, args.epochs, args.patience,
                          args.seed, device, edge_index, edge_attr)
    folds = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (fit_part, valid_part) in enumerate(cv.split(train_idx, labels[train_idx]), 1):
        _, result = fit(arrays, train_idx[fit_part], train_idx[valid_part], best, args.epochs,
                        args.patience, args.seed + fold, device, edge_index, edge_attr)
        folds.append({"fold": fold, **result})
    cv_frame = pd.DataFrame(folds)
    cv_frame.to_csv(args.output_dir / "cross_validation_metrics.csv", index=False)
    summary = {
        "model": "PyGCN 2", "device": str(device), "n_samples": len(labels),
        "n_proteins": len(proteins), "projected_undirected_edges": int(edge_index.shape[1] // 2),
        "projected_directed_edges": int(edge_index.shape[1]), "best_configuration": best,
        "test_metrics": test_metrics, "cv_mean": cv_frame.drop(columns="fold").mean().to_dict(),
        "cv_std": cv_frame.drop(columns="fold").std(ddof=1).to_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([test_metrics]).to_csv(args.output_dir / "test_metrics.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
