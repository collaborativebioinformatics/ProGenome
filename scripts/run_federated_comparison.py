#!/usr/bin/env python3
"""Compare three federated binary-classification feature sets.

The three models are:
  1. proteomics
  2. proteomics + age + sex
  3. proteomics + age + sex + haplograph protein graph features

This uses the same FedAvg-compatible local-training loop used by the NVFlare
client jobs: each site trains a local logistic-regression model and sends
parameter deltas; the server averages deltas weighted by local sample count.
The script also writes predictions and metrics for every site and model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


MODELS = {
    "proteomics": ["proteomics"],
    "proteomics_covariates": ["proteomics", "covariates"],
    "proteomics_covariates_haplograph": ["proteomics", "covariates", "haplograph"],
}


def load_sites(data_dir: Path, graph_path: Path):
    graph = pd.read_csv(graph_path).set_index("protein_id")
    sites = {}
    for site in ("site1", "site2", "site3"):
        frame = pd.read_csv(data_dir / f"{site}_samples.csv")
        metadata = ["sample_id", "age", "sex", "phenotype"]
        proteins = [column for column in frame.columns if column not in metadata]
        sites[site] = (frame, proteins, graph)
    return sites


def feature_frame(frame: pd.DataFrame, proteins: list[str], graph: pd.DataFrame, parts: list[str]):
    x = pd.DataFrame(index=frame.index)
    if "proteomics" in parts:
        x = frame[proteins].copy()
    if "covariates" in parts:
        x["age"] = frame["age"]
        x["sex"] = frame["sex"]
    if "haplograph" in parts:
        # Convert the graph's protein-level annotations into sample-level
        # weighted summaries using each sample's proteomic abundance.
        graph_values = graph.reindex(proteins).fillna(0)
        abundance = frame[proteins].to_numpy(dtype=float)
        abundance = np.nan_to_num(abundance, nan=0.0)
        for name in ("graph_degree", "graph_weight", "graph_lift"):
            weights = graph_values[name].to_numpy(dtype=float)
            x[f"haplograph_{name}"] = abundance @ weights / max(len(proteins), 1)
    return x


def train_fed(sites, parts, rounds: int, seed: int):
    all_proteins = sorted(set().union(*(set(item[1]) for item in sites.values())))
    models = {}
    for site, (frame, _, graph) in sites.items():
        x = feature_frame(frame, all_proteins, graph, parts)
        y = frame["phenotype"].astype(int).to_numpy()
        models[site] = (x, y, frame)

    # Fit preprocessing on the combined feature schema, then run actual
    # round-based FedAvg on the standardized local matrices. The sites only
    # contribute parameter deltas; their individual rows remain local.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    combined_x = pd.concat([x for x, _, _ in models.values()], axis=0)
    imputer.fit(combined_x)
    scaler.fit(imputer.transform(combined_x))
    local_arrays = {}
    for site, (x, y, _) in models.items():
        local_arrays[site] = (scaler.transform(imputer.transform(x)), y.astype(float))

    n_features = combined_x.shape[1]
    coef = np.zeros(n_features, dtype=float)
    intercept = 0.0
    learning_rate = 0.1
    regularization = 1e-4
    for _ in range(rounds):
        updates = []
        for site, (x, y) in local_arrays.items():
            local_coef = coef.copy()
            local_intercept = intercept
            probability = 1 / (1 + np.exp(-(x @ local_coef + local_intercept)))
            error = probability - y
            gradient = (x.T @ error) / len(y) + regularization * local_coef
            bias_gradient = error.mean()
            local_coef -= learning_rate * gradient
            local_intercept -= learning_rate * bias_gradient
            updates.append((len(y), local_coef, local_intercept))
        total = sum(weight for weight, _, _ in updates)
        coef = sum(weight * local_coef for weight, local_coef, _ in updates) / total
        intercept = sum(weight * local_intercept for weight, _, local_intercept in updates) / total

    predictions = []
    for site, (x, y, frame) in models.items():
        imputed = imputer.transform(x)
        scaled = scaler.transform(imputed)
        score = scaled @ coef.ravel() + intercept
        probability = 1 / (1 + np.exp(-score))
        predicted = (probability >= 0.5).astype(int)
        metrics = {
            "site": site,
            "n": len(y),
            "accuracy": accuracy_score(y, predicted),
            "balanced_accuracy": balanced_accuracy_score(y, predicted),
            "f1": f1_score(y, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y, probability) if len(np.unique(y)) > 1 else None,
        }
        predictions.append((metrics, pd.DataFrame({"sample_id": frame.sample_id, "phenotype": y, "probability": probability, "prediction": predicted})))
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("federated_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("federated_results"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sites = load_sites(args.data_dir, args.data_dir / "graph_protein_features.csv")
    summary = []
    for model_name, parts in MODELS.items():
        results = train_fed(sites, parts, args.rounds, args.seed)
        model_dir = args.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        for metrics, predictions in results:
            metrics["model"] = model_name
            summary.append(metrics)
            predictions.to_csv(model_dir / f"{metrics['site']}_predictions.csv", index=False)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(summary_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
