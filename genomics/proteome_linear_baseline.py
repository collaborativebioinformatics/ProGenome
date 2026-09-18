#!/usr/bin/env python3
"""Per-protein ridge regression: can the genome (carrier matrix) predict each protein's harmonised level?

The GNN compresses the genome into a 64-d embedding, which is the wrong tool for sparse
cis effects (one cluster -> one protein).  This is the honest baseline for that question:
one linear model per protein from the 6,551 carrier features, scored as test R^2, split by
whether the ground truth says the protein is cis-affected by a causal cluster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import haplokg
import haplokg_proteins as hp


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--synth-dir", type=Path, default=None, help="folder with measured_long.csv, sample_metadata.csv, ground_truth.json")
    parser.add_argument("--alpha", type=float, nargs="+", default=[10.0, 100.0, 1000.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    kg_dir = here / "outputs" / "kg" / args.chrom
    synth = args.synth_dir or here / "outputs" / "proteomics_synth" / args.chrom
    out_dir = here / "outputs" / "gnn_v2" / args.chrom / "proteome_ridge_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    kg = haplokg.load_kg(kg_dir)
    bed = here.parent / "proteomics" / "uniprot_chr22.bed"
    symbols = here.parent / "proteomics" / "synthetic_proteomics_chr22" / "gene_symbol_cache.csv"
    t = hp.build_protein_tables(kg, bed, synth / "measured_long.csv", synth / "sample_metadata.csv", symbols)
    Y, M = t["abundance"].toarray(), t["observed"].toarray().astype(bool)
    X = kg["carries"].astype(np.float32)
    split = haplokg.load_or_make_split(kg, here / "outputs" / "splits" / args.chrom / f"split_seed{args.seed}.csv", seed=args.seed)
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    Yf = np.where(M, Y, 0.0)                          # unobserved -> 0 (= the harmonised mean); good enough for a baseline

    def r2(mask, yhat):
        out = np.full(Y.shape[1], np.nan)
        for j in range(Y.shape[1]):
            m = mask & M[:, j]
            if m.sum() >= 5:
                ss_res = ((Y[m, j] - yhat[m, j]) ** 2).sum(); ss_tot = ((Y[m, j] - Y[m, j].mean()) ** 2).sum()
                out[j] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return out

    best = None
    for alpha in args.alpha:
        model = Ridge(alpha=alpha).fit(X[tr], Yf[tr])
        yhat = model.predict(X)                        # all rows once; r2() masks by split
        score = np.nanmean(r2(va, yhat))
        print(f"alpha={alpha:<7} val mean R^2 {score:.4f}")
        if best is None or score > best[1]:
            best = (alpha, score, yhat)
    alpha, _, yhat = best
    test_r2 = r2(te, yhat)

    truth = json.loads((synth / "ground_truth.json").read_text())
    prot = t["proteins"]["protein_id"].tolist()
    cis = {c["cis_protein"]: c for c in truth["causal_clusters"]}
    is_cis = np.array([p in cis for p in prot])
    table = pd.DataFrame({"protein_id": prot, "test_r2": test_r2, "is_cis": is_cis,
                          "beta_cis": [cis[p]["beta_cis"] if p in cis else np.nan for p in prot]})
    table.to_csv(out_dir / "protein_r2.csv", index=False)
    report = {"alpha": alpha, "mean_r2_all": float(np.nanmean(test_r2)), "median_r2_all": float(np.nanmedian(test_r2)),
              "mean_r2_cis": float(np.nanmean(test_r2[is_cis])), "mean_r2_other": float(np.nanmean(test_r2[~is_cis])),
              "n_cis": int(is_cis.sum()), "cis_proteins_with_r2_over_0.1": int((test_r2[is_cis] > 0.1).sum()),
              "other_proteins_with_r2_over_0.1": int((test_r2[~is_cis] > 0.1).sum())}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(table[is_cis].sort_values("test_r2", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
