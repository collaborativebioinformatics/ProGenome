#!/usr/bin/env python3
"""Score the federated global model centrally on the same held-out people as the central model.

    python federated/evaluate_global.py            # finds FL_global_model.pt in outputs/federated/<chrom>/workspace

Loads the server's saved global model, rebuilds the full v2 graph exactly as train_gnn_v2.py does
(--modality both --init raw) and reports test balanced accuracy / AUC next to the central run's numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
GENOMICS = HERE.parent
sys.path.insert(0, str(GENOMICS)); sys.path.insert(0, str(HERE))
import haplokg  # noqa: E402
import haplokg_proteins as hp  # noqa: E402
from model import ProGenomeGNN  # noqa: E402

CO = ("cluster", "co_occurs", "cluster")
MEAS = ("individual", "measured", "protein")
RMEAS = ("protein", "rev_measured", "individual")


def load_global_state(workspace: Path, filename: str) -> dict:
    hits = sorted(workspace.rglob(filename))
    if not hits:
        raise SystemExit(f"no {filename} under {workspace}")
    ckpt = torch.load(hits[-1], map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    return {k: (torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v) for k, v in state.items()}, hits[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-file", default="FL_global_model.pt", help="or best_FL_global_model.pt")
    args = parser.parse_args()
    kg_dir = GENOMICS / "outputs" / "kg" / args.chrom
    fed_dir = GENOMICS / "outputs" / "federated" / args.chrom
    cfg_path = fed_dir / "model_args.json"

    kg = haplokg.load_kg(kg_dir); pt = hp.load_protein_tables(kg_dir)
    data = torch.load(kg_dir / "hetero_v2.pt", weights_only=False)
    split = haplokg.load_or_make_split(kg, GENOMICS / "outputs" / "splits" / args.chrom / f"split_seed{args.seed}.csv", seed=args.seed)
    cfg = json.loads(cfg_path.read_text())
    relations = [tuple(r) for r in cfg["relations"]]
    x = {"individual": torch.from_numpy(np.concatenate([kg["carries"].toarray().astype(np.float32), pt["abundance"].toarray().astype(np.float32),
                                                        pt["observed"].toarray().astype(np.float32)], axis=1)),
         "cluster": data["cluster"].x, "block": data["block"].x, "gene": data["gene"].x, "protein": data["protein"].x}
    ei = {rel: data[rel].edge_index for rel in relations}
    lift = data[CO].edge_attr[:, 1]
    ew = {CO: torch.log(lift) / torch.log(lift).max(), MEAS: data[MEAS].edge_attr[:, 0], RMEAS: data[RMEAS].edge_attr[:, 0]}
    y = data["individual"].y_phenotype.numpy()
    test = (split == "test") & (y >= 0)

    state, path = load_global_state(fed_dir / "workspace", args.model_file)
    model = ProGenomeGNN(str(cfg_path)); model.load_state_dict(state); model.eval()
    with torch.no_grad():
        logits, _ = model(x, ei, ew)
    prob = torch.softmax(logits, 1)[:, 1].numpy()
    fed = {"balanced_accuracy": float(balanced_accuracy_score(y[test], (prob[test] >= 0.5).astype(int))),
           "auc": float(roc_auc_score(y[test], prob[test])), "n_test": int(test.sum()), "model_file": str(path)}

    central_path = GENOMICS / "outputs" / "gnn_v2" / args.chrom / "phenotype_both_raw" / "metrics.json"
    central_path = central_path if central_path.exists() else GENOMICS / "outputs_brev" / "progenome-a100" / "gnn_v2" / args.chrom / "phenotype_both_raw" / "metrics.json"
    central = json.loads(central_path.read_text())["test"] if central_path.exists() else {}
    # per-site view of the same test people (what each hospital would see)
    ind2 = pt["individuals"]; sites = pt["label_maps"]["site"]
    per_site = []
    for i, s in enumerate(sites):
        m = test & (ind2["site_code"].to_numpy() == i)
        per_site.append({"site": s, "n_test": int(m.sum()), "balanced_accuracy": float(balanced_accuracy_score(y[m], (prob[m] >= 0.5).astype(int))),
                         "auc": float(roc_auc_score(y[m], prob[m]))})
    report = {"federated_global_model": fed, "central_model_same_test_people": {k: central.get(k) for k in ("balanced_accuracy", "roc_auc", "n")},
              "federated_per_site": per_site}
    (fed_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
