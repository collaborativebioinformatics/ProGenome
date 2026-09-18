#!/usr/bin/env python3
"""Does the haploblock-cluster graph co-occur with phenotypes?

Three questions, all answered from the knowledge graph in outputs/kg/<chrom>:

1. cluster x phenotype   For every cluster: is being a carrier independent of
                         ancestry / population / sex?  (chi-square on the 2 x k
                         carrier-by-class table, Cramer's V, BH-FDR).  Sex is the
                         negative control: chr22 is autosomal, so nothing should
                         pass.
2. edge x phenotype      Do co-occurring clusters (lift edges) have more similar
                         ancestry profiles than random cluster pairs?  If the
                         graph structure tracks population structure, yes.
3. block x phenotype     Where along the chromosome is the graph most
                         ancestry-informative?  (max Cramer's V per block)

Writes CSV tables, a summary.json and three PNGs to outputs/cooccurrence/<chrom>/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

import haplokg

TARGETS = ("ancestry", "population", "sex")


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def association(carries: sparse.csr_matrix, codes: np.ndarray, n_classes: int) -> dict:
    """Carrier-vs-class independence test for every cluster at once.

    carries: (n_labelled x clusters) 0/1;  codes: class code per row (0..k-1).
    Cramer's V for a 2 x k table is sqrt(chi2 / n) because min(2-1, k-1) = 1.
    """
    n = carries.shape[0]
    onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, n_classes))
    carriers = np.asarray((onehot.T @ carries).todense(), dtype=float)          # k x clusters
    n_class = np.asarray(onehot.sum(axis=0)).ravel()                            # k
    support = carriers.sum(axis=0)                                              # clusters
    non_carriers = n_class[:, None] - carriers
    expected1 = np.outer(n_class, support) / n
    expected0 = n_class[:, None] - expected1
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((carriers - expected1) ** 2 / expected1, axis=0) + \
               np.nansum((non_carriers - expected0) ** 2 / expected0, axis=0)
    p = stats.chi2.sf(chi2, df=n_classes - 1)
    frac = carriers / n_class[:, None]                                          # P(carry | class)
    overall = support / n
    with np.errstate(divide="ignore", invalid="ignore"):
        enrich = np.where(overall > 0, frac / overall, np.nan)                  # k x clusters
    return {
        "chi2": chi2, "p": p, "q": bh_fdr(p), "cramers_v": np.sqrt(chi2 / n),
        "frac": frac, "enrich": enrich, "dominant": np.nanargmax(np.nan_to_num(enrich, nan=-1), axis=0),
        "support": support,
    }


def edge_profile_similarity(co: pd.DataFrame, enrich: np.ndarray, dominant: np.ndarray, rng: np.random.Generator) -> dict:
    """Cosine similarity of the two endpoints' ancestry-deviation profiles, real edges vs shuffled targets."""
    deviation = np.nan_to_num(enrich.T - 1.0)                                   # clusters x k, 0 = neutral
    norm = np.linalg.norm(deviation, axis=1, keepdims=True)
    unit = deviation / np.where(norm > 0, norm, 1.0)
    src, dst = co["src"].to_numpy(), co["dst"].to_numpy()
    perm = rng.permutation(dst)                                                 # keeps the degree of every source
    cos_real = (unit[src] * unit[dst]).sum(1)
    cos_null = (unit[src] * unit[perm]).sum(1)
    same_real = dominant[src] == dominant[dst]
    same_null = dominant[src] == dominant[perm]
    rho, rho_p = stats.spearmanr(co["lift"].to_numpy(), cos_real)
    return {
        "n_edges": int(len(co)),
        "profile_cosine_real_mean": float(cos_real.mean()),
        "profile_cosine_null_mean": float(cos_null.mean()),
        "same_dominant_ancestry_real": float(same_real.mean()),
        "same_dominant_ancestry_null": float(same_null.mean()),
        "spearman_lift_vs_profile_cosine": float(rho),
        "spearman_p": float(rho_p),
        "_cos_real": cos_real, "_cos_null": cos_null,
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--kg-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    kg_dir = args.kg_dir or here / "outputs" / "kg" / args.chrom
    out_dir = args.out_dir or here / "outputs" / "cooccurrence" / args.chrom
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    kg = haplokg.load_kg(kg_dir)
    ind, clusters, blocks, co = kg["individuals"], kg["clusters"], kg["blocks"], kg["co_occurs"]
    labelled = ind["ancestry_code"].to_numpy() >= 0
    carries = kg["carries"][labelled]
    summary = {"chrom": args.chrom, "n_individuals_labelled": int(labelled.sum()), "n_clusters": int(len(clusters)), "fdr": args.fdr}

    # ---- 1. cluster x phenotype -------------------------------------------------
    table = clusters[["cluster_idx", "cluster_id", "block_id", "block_idx", "support"]].copy()
    results = {}
    for target in TARGETS:
        codes = ind.loc[labelled, f"{target}_code"].to_numpy()
        classes = kg["label_maps"][target]
        res = association(carries, codes, len(classes))
        results[target] = res
        table[f"{target}_cramers_v"] = res["cramers_v"]
        table[f"{target}_p"] = res["p"]
        table[f"{target}_q"] = res["q"]
        table[f"{target}_dominant"] = [classes[i] for i in res["dominant"]]
        if target == "ancestry":
            for i, name in enumerate(classes):
                table[f"carrier_frac_{name}"] = res["frac"][i]
        sig = res["q"] < args.fdr
        summary[f"{target}_clusters_significant"] = int(sig.sum())
        summary[f"{target}_clusters_significant_frac"] = float(sig.mean())
        summary[f"{target}_cramers_v_median"] = float(np.median(res["cramers_v"]))
        summary[f"{target}_cramers_v_p90"] = float(np.quantile(res["cramers_v"], 0.9))
    table.to_csv(out_dir / "cluster_phenotype_association.csv", index=False)
    top = table.sort_values("ancestry_cramers_v", ascending=False).head(25)
    top.to_csv(out_dir / "top_ancestry_clusters.csv", index=False)

    # ---- 2. edge x phenotype ----------------------------------------------------
    edge = edge_profile_similarity(co, results["ancestry"]["enrich"], results["ancestry"]["dominant"], rng)
    cos_real, cos_null = edge.pop("_cos_real"), edge.pop("_cos_null")
    summary["edges"] = edge
    # by lift quartile
    q = pd.qcut(co["lift"], 4, labels=["Q1 (lowest lift)", "Q2", "Q3", "Q4 (highest lift)"])
    by_lift = pd.DataFrame({"lift_quartile": q, "cos": cos_real}).groupby("lift_quartile", observed=True)["cos"].agg(["mean", "count"]).reset_index()
    by_lift.to_csv(out_dir / "edge_similarity_by_lift_quartile.csv", index=False)
    summary["edges"]["profile_cosine_by_lift_quartile"] = dict(zip(by_lift["lift_quartile"].astype(str), by_lift["mean"].round(4)))

    # ---- 3. block x phenotype ---------------------------------------------------
    per_block = table.groupby("block_idx").agg(
        n_clusters_kept=("cluster_idx", "size"),
        max_ancestry_v=("ancestry_cramers_v", "max"),
        n_ancestry_significant=("ancestry_q", lambda s: int((s < args.fdr).sum())),
        max_sex_v=("sex_cramers_v", "max"),
    ).reset_index()
    per_block = blocks[["block_idx", "block_id", "start", "end", "block_length"]].merge(per_block, on="block_idx", how="left")
    per_block.to_csv(out_dir / "block_informativeness.csv", index=False)
    summary["blocks_with_kept_clusters"] = int(per_block["n_clusters_kept"].notna().sum())
    summary["blocks_with_significant_ancestry_cluster"] = int((per_block["n_ancestry_significant"].fillna(0) > 0).sum())

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- plots ------------------------------------------------------------------
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colours = {"ancestry": "#1f6f8b", "population": "#6a4c93", "sex": "#c44536"}
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(0, max(0.05, table[[f"{t}_cramers_v" for t in TARGETS]].to_numpy().max()), 60)
        for target in TARGETS:
            ax.hist(table[f"{target}_cramers_v"], bins=bins, histtype="step", linewidth=1.8,
                    label=f"{target} ({summary[f'{target}_clusters_significant']:,} clusters FDR<{args.fdr})", color=colours[target])
        ax.set_xlabel("Cramér's V (carrier status vs phenotype)")
        ax.set_ylabel("clusters")
        ax.set_title(f"{args.chrom}: how strongly each haploblock cluster tracks a phenotype")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_dir / "cramers_v_by_phenotype.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 3.6))
        pb = per_block.dropna(subset=["max_ancestry_v"])
        mid = (pb["start"] + pb["end"]) / 2e6
        ax.scatter(mid, pb["max_ancestry_v"], s=10, color=colours["ancestry"], label="ancestry (max V in block)")
        ax.scatter(mid, pb["max_sex_v"], s=10, color=colours["sex"], alpha=0.7, label="sex (negative control)")
        ax.set_xlabel(f"{args.chrom} position (Mb)"); ax.set_ylabel("max Cramér's V per block")
        ax.set_title("Where along the chromosome the haploblock graph tracks ancestry")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)      # below the axes, never on the points
        fig.tight_layout(); fig.savefig(out_dir / "informativeness_along_chromosome.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(-1, 1, 50)
        ax.hist(cos_null, bins=bins, alpha=0.6, color="#9a9a9a", label=f"shuffled pairs (mean {edge['profile_cosine_null_mean']:.2f})")
        ax.hist(cos_real, bins=bins, alpha=0.7, color=colours["ancestry"], label=f"lift edges (mean {edge['profile_cosine_real_mean']:.2f})")
        ax.set_xlabel("cosine similarity of endpoint ancestry profiles"); ax.set_ylabel("edges")
        ax.set_title("Co-occurring clusters share ancestry profiles")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout(); fig.savefig(out_dir / "edge_ancestry_similarity.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    print(json.dumps(summary, indent=2, default=float))
    print("\nTop ancestry-informative clusters:")
    cols = ["cluster_id", "support", "ancestry_cramers_v", "ancestry_dominant"] + [c for c in table.columns if c.startswith("carrier_frac_")]
    print(top[cols].head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
