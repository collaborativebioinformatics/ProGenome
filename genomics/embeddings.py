#!/usr/bin/env python3
"""How much phenotype structure do the embeddings carry?

For every individual embedding found (SVD of the carrier matrix, and the last
hidden layer of each trained GNN) this script
  * projects to 2-D with PCA and colours by ancestry and by sex (the control),
  * computes silhouette scores by ancestry / population / sex,
  * fits a 5-nearest-neighbour classifier on train individuals and scores the
    test split -> "how linearly-separable-for-free is the phenotype in this space".
Cluster embeddings are projected too, coloured by the ancestry they are enriched in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

import haplokg

ANCESTRY_COLOURS = {"AFR": "#d55e00", "AMR": "#cc79a7", "EAS": "#009e73", "EUR": "#0072b2", "SAS": "#e69f00"}
SEX_COLOURS = {"female": "#7b3294", "male": "#008837"}


def score(emb: np.ndarray, ind: pd.DataFrame, split: np.ndarray, label_maps: dict) -> dict:
    out = {}
    for target in ("ancestry", "population", "sex"):
        y = ind[f"{target}_code"].to_numpy()
        ok = y >= 0
        out[f"silhouette_{target}"] = float(silhouette_score(emb[ok], y[ok])) if len(np.unique(y[ok])) > 1 else float("nan")
        tr, te = (split == "train") & ok, (split == "test") & ok
        knn = KNeighborsClassifier(n_neighbors=5).fit(emb[tr], y[tr])
        out[f"knn5_test_balanced_accuracy_{target}"] = float(balanced_accuracy_score(y[te], knn.predict(emb[te])))
    return out


def plot_individuals(emb: np.ndarray, ind: pd.DataFrame, title: str, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = PCA(n_components=2, random_state=0).fit_transform(emb)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (column, colours, label) in zip(axes, [("ancestry", ANCESTRY_COLOURS, "ancestry"), ("sex", SEX_COLOURS, "sex (control)")]):
        values = ind[column].fillna("unlabelled")
        for name, colour in list(colours.items()) + [("unlabelled", "#bbbbbb")]:
            m = (values == name).to_numpy()
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=7, color=colour, label=f"{name} ({m.sum()})", alpha=0.75, linewidths=0)
        ax.set_title(f"{title} - coloured by {label}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(frameon=False, markerscale=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_clusters(emb: np.ndarray, assoc: pd.DataFrame, title: str, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = PCA(n_components=2, random_state=0).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    strong = assoc["ancestry_cramers_v"].to_numpy() >= 0.3
    ax.scatter(xy[~strong, 0], xy[~strong, 1], s=4, color="#cccccc", label="weakly ancestry-associated (V<0.3)", linewidths=0)
    for name, colour in ANCESTRY_COLOURS.items():
        m = strong & (assoc["ancestry_dominant"].to_numpy() == name)
        ax.scatter(xy[m, 0], xy[m, 1], s=6, color=colour, label=f"enriched in {name} ({m.sum()})", alpha=0.8, linewidths=0)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(frameon=False, markerscale=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    kg_dir = here / "outputs" / "kg" / args.chrom
    out_dir = here / "outputs" / "embeddings" / args.chrom
    out_dir.mkdir(parents=True, exist_ok=True)

    kg = haplokg.load_kg(kg_dir)
    ind = kg["individuals"]
    split = haplokg.load_or_make_split(kg, here / "outputs" / "splits" / args.chrom / f"split_seed{args.seed}.csv", seed=args.seed)
    assoc_path = here / "outputs" / "cooccurrence" / args.chrom / "cluster_phenotype_association.csv"
    assoc = pd.read_csv(assoc_path) if assoc_path.exists() else None

    sources = {}
    for path in sorted(out_dir.glob("svd*_individual.npy")):
        sources[path.stem.replace("_individual", "")] = (path, path.with_name(path.name.replace("individual", "cluster")))
    for path in sorted((here / "outputs" / "gnn" / args.chrom).glob("*/embedding_individual.npy")):
        sources[f"gnn_{path.parent.name}"] = (path, path.with_name("embedding_cluster.npy"))

    rows = []
    for name, (ind_path, cl_path) in sources.items():
        emb = np.load(ind_path)
        result = {"embedding": name, "dim": emb.shape[1], **score(emb, ind, split, kg["label_maps"])}
        rows.append(result)
        plot_individuals(emb, ind, name, out_dir / f"individuals_{name}.png")
        if cl_path.exists() and assoc is not None:
            plot_clusters(np.load(cl_path), assoc, f"clusters - {name}", out_dir / f"clusters_{name}.png")
        print(f"{name:32s} dim={emb.shape[1]:<4d} " + "  ".join(f"{k.split('_')[-1]}: sil={result[f'silhouette_{k.split('_')[-1]}']:.3f} "
              f"knn={result[f'knn5_test_balanced_accuracy_{k.split('_')[-1]}']:.3f}" for k in ("silhouette_ancestry", "silhouette_population", "silhouette_sex")))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "embedding_quality.csv", index=False)
    (out_dir / "embedding_quality.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
