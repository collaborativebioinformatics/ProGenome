#!/usr/bin/env python3
"""Baseline: L2 logistic regression on the sparse carrier matrix.

Individuals are the rows, kept haploblock clusters the 0/1 columns.  One model
per target (ancestry, population, sex); C is chosen on the validation split;
metrics are reported on the untouched test split.  Sex is the negative control
(autosomal chromosome -> expect chance level).  The split is saved so that the
GNN is scored on exactly the same people.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import haplokg

TARGETS = ("ancestry", "population", "sex")


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--kg-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    args = parser.parse_args()
    kg_dir = args.kg_dir or here / "outputs" / "kg" / args.chrom
    out_dir = args.out_dir or here / "outputs" / "baseline" / args.chrom
    split_path = args.split_path or here / "outputs" / "splits" / args.chrom / f"split_seed{args.seed}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    kg = haplokg.load_kg(kg_dir)
    ind, clusters, X = kg["individuals"], kg["clusters"], kg["carries"].astype(np.float32)
    split = haplokg.load_or_make_split(kg, split_path, seed=args.seed)
    masks = {name: split == name for name in ("train", "val", "test")}
    print({k: int(v.sum()) for k, v in masks.items()}, "unlabelled:", int((split == "unlabelled").sum()))

    report = {"chrom": args.chrom, "seed": args.seed, "n_features": int(X.shape[1]), "targets": {}}
    for target in TARGETS:
        y = ind[f"{target}_code"].to_numpy()
        ok = y >= 0                                    # a few individuals lack a label for this target
        tr, va, te = masks["train"] & ok, masks["val"] & ok, masks["test"] & ok
        classes = kg["label_maps"][target]

        majority = np.bincount(y[tr]).argmax()
        chance = metrics(y[te], np.full(te.sum(), majority))

        best = None
        for C in args.C_grid:
            model = LogisticRegression(C=C, max_iter=5000, random_state=args.seed)
            model.fit(X[tr], y[tr])
            val = metrics(y[va], model.predict(X[va]))
            if best is None or val["balanced_accuracy"] > best[1]["balanced_accuracy"]:
                best = (C, val, model)
        C, val, model = best
        test = metrics(y[te], model.predict(X[te]))
        report["targets"][target] = {"C": C, "val": val, "test": test, "majority_class_test": chance, "n_classes": len(classes)}
        print(f"{target:11s} C={C:<5} val bal-acc={val['balanced_accuracy']:.3f}  TEST acc={test['accuracy']:.3f} "
              f"bal-acc={test['balanced_accuracy']:.3f} macro-F1={test['macro_f1']:.3f}  (majority-class acc={chance['accuracy']:.3f})")

        pd.DataFrame({
            "individual_id": ind.loc[te, "individual_id"].to_numpy(),
            "true": [classes[i] for i in y[te]],
            "pred": [classes[i] for i in model.predict(X[te])],
        }).to_csv(out_dir / f"{target}_test_predictions.csv", index=False)

        # which clusters drive each class (largest positive coefficients)
        coef = model.coef_ if model.coef_.shape[0] > 1 else np.vstack([-model.coef_[0], model.coef_[0]])
        rows = []
        for k, name in enumerate(classes):
            for j in np.argsort(coef[k])[::-1][:10]:
                rows.append({"class": name, "cluster_id": clusters.loc[j, "cluster_id"], "coef": float(coef[k, j]),
                             "support": int(clusters.loc[j, "support"])})
        pd.DataFrame(rows).to_csv(out_dir / f"{target}_top_clusters_per_class.csv", index=False)

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
