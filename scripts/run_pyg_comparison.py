#!/usr/bin/env python3
"""Compare a PyG GCN patient-graph classifier with logistic regression."""

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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


def load_expression(data_dir: Path):
    metadata = pd.read_csv(data_dir / "sample_metadata.csv").set_index("sample_id")
    metadata["site"] = metadata.index.to_series().str.split("_", n=1).str[0].str.lower()
    matrices = []
    for path in sorted(data_dir.glob("site*_proteomics_log2.csv")):
        matrix = pd.read_csv(path).set_index("protein_id")
        matrices.append(matrix.drop(columns=["gene_symbol"]).T)
    expression = pd.concat(matrices).loc[metadata.index]
    return expression, metadata


def projected_edges(edge_path: Path, proteins: list[str], max_edges: int) -> torch.Tensor:
    protein_set = set(proteins)
    opener = gzip.open if edge_path.suffix == ".gz" else open
    edge_table = pd.read_csv(opener(edge_path, "rt"), usecols=["source_protein", "target_protein", "weight"])
    pairs: dict[tuple[str, str], float] = {}
    for row in edge_table.itertuples(index=False):
        source = [p for p in str(row.source_protein).split(";") if p in protein_set]
        target = [p for p in str(row.target_protein).split(";") if p in protein_set]
        for left in source:
            for right in target:
                if left != right:
                    key = tuple(sorted((left, right)))
                    pairs[key] = pairs.get(key, 0.0) + float(row.weight)
    selected = sorted(pairs.items(), key=lambda item: item[1], reverse=True)[:max_edges]
    index = {protein: i for i, protein in enumerate(proteins)}
    edges = [(index[left], index[right]) for (left, right), _ in selected]
    edges += [(right, left) for left, right in edges]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_graphs(expression, metadata, proteins, graph_features, edge_index):
    scale = graph_features.loc[proteins, ["graph_degree", "graph_weight", "graph_lift"]].copy()
    scale = (scale - scale.mean()) / scale.std().replace(0, 1)
    graph_features = torch.tensor(scale.fillna(0).to_numpy(dtype=np.float32))
    graphs = []
    for sample_id, row in expression[proteins].iterrows():
        abundance = row.to_numpy(dtype=np.float32)
        abundance = np.nan_to_num(abundance, nan=np.nanmedian(abundance))
        x = torch.cat([torch.tensor(abundance[:, None]), graph_features], dim=1)
        covariates = torch.tensor(
            [float(metadata.loc[sample_id, "age"]) / 85.0,
             float(metadata.loc[sample_id, "sex"]),
             float(metadata.loc[sample_id, "site"] == "site2"),
             float(metadata.loc[sample_id, "site"] == "site3")],
            dtype=torch.float32,
        ).view(1, -1)
        graphs.append(Data(x=x, edge_index=edge_index, covariates=covariates,
                           y=torch.tensor([int(metadata.loc[sample_id, "phenotype"])], dtype=torch.long)))
    return graphs


class GCNClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.classifier = nn.Sequential(nn.Linear(16 + 4, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, batch):
        hidden = torch.relu(self.conv1(batch.x, batch.edge_index))
        hidden = torch.relu(self.conv2(hidden, batch.edge_index))
        pooled = global_mean_pool(hidden, batch.batch)
        return self.classifier(torch.cat([pooled, batch.covariates], dim=1)).squeeze(1)


def evaluate(model, loader):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for batch in loader:
            probabilities.extend(torch.sigmoid(model(batch)).cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    predictions = probabilities >= 0.5
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
    }


def fit(train_graphs, validation_graphs, epochs, seed):
    torch.manual_seed(seed)
    model = GCNClassifier(train_graphs[0].x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    labels = torch.cat([graph.y for graph in train_graphs])
    positive_weight = (labels == 0).sum() / (labels == 1).sum()
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight.float())
    for _ in range(epochs):
        model.train()
        for batch in DataLoader(train_graphs, batch_size=64, shuffle=True):
            optimizer.zero_grad()
            loss = loss_function(model(batch), batch.y.float())
            loss.backward()
            optimizer.step()
    return model, evaluate(model, DataLoader(validation_graphs, batch_size=128))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("proteomics/synthetic_proteomics_chr22"))
    parser.add_argument("--graph-features", type=Path, default=Path("federated_data/graph_protein_features.csv"))
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("proteomics/pyg_results"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-edges", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expression, metadata = load_expression(args.data_dir)
    graph_features = pd.read_csv(args.graph_features).set_index("protein_id")
    proteins = sorted(set(expression.columns) & set(graph_features.index))
    edge_index = projected_edges(args.edges, proteins, args.max_edges)
    graphs = make_graphs(expression, metadata, proteins, graph_features, edge_index)
    labels = metadata.loc[expression.index, "phenotype"].to_numpy()
    strata = metadata.loc[expression.index, "site"].to_numpy() + "_" + labels.astype(str)
    train_idx, test_idx = train_test_split(np.arange(len(graphs)), test_size=0.2, random_state=args.seed, stratify=strata)
    train_labels = labels[train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    model, test_metrics = fit([graphs[i] for i in train_idx], test_graphs, args.epochs, args.seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_metrics = []
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(train_idx, train_labels), 1):
        _, result = fit([graphs[train_idx[i]] for i in fit_idx],
                        [graphs[train_idx[i]] for i in valid_idx], args.epochs, args.seed + fold)
        result["fold"] = fold
        fold_metrics.append(result)
    cv_frame = pd.DataFrame(fold_metrics)
    cv_frame.to_csv(args.output_dir / "cross_validation_metrics.csv", index=False)
    summary = {
        "model": "PyG GCN",
        "n_samples": len(graphs),
        "n_proteins": len(proteins),
        "projected_edges": int(edge_index.shape[1] // 2),
        "epochs": args.epochs,
        "test_metrics": test_metrics,
        "cv_mean": cv_frame.drop(columns="fold").mean().to_dict(),
        "cv_std": cv_frame.drop(columns="fold").std(ddof=1).to_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([test_metrics]).to_csv(args.output_dir / "test_metrics.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
