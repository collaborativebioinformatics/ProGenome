"""
Generates two presentation-ready plots from the synthetic proteomics data:

1. A heatmap of the proteins most strongly driven by age/sex/phenotype
   (per protein_covariate_effects.csv), across all samples from all 3
   sites, with a colored strip showing each sample's phenotype - meant
   to show that the data has real, visually visible structure.

2. A boxplot of the single most phenotype-associated protein's intensity,
   split by phenotype and by site - a simple, easy-to-read proof that the
   injected signal is recoverable.

Run from the same folder as synthetic_proteomics_chr22/, or pass --data-dir.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

N_TOP_PROTEINS_HEATMAP = 40


def load_all(data_dir: Path):
    metadata = pd.read_csv(data_dir / "sample_metadata.csv", index_col="sample_id")
    effects = pd.read_csv(data_dir / "protein_covariate_effects.csv", index_col="protein_id")

    site_frames = []
    for site_csv in sorted(data_dir.glob("site*_proteomics_log2.csv")):
        df = pd.read_csv(site_csv, index_col="protein_id")
        gene_symbols = df["gene_symbol"]
        site_frames.append(df.drop(columns="gene_symbol"))

    matrix = pd.concat(site_frames, axis=1)  # proteins x all samples
    return matrix, metadata, effects, gene_symbols


def plot_heatmap(matrix: pd.DataFrame, metadata: pd.DataFrame, effects: pd.DataFrame,
                  gene_symbols: pd.Series, out_path: Path, max_samples_per_group: int = 30,
                  rng: np.random.Generator = None):
    # Select proteins specifically by phenotype effect (not the combined
    # age+sex+phenotype total), so the heatmap actually tells the
    # phenotype story rather than being diluted by age/sex-only proteins.
    top_proteins = effects["beta_phenotype"].abs().sort_values(ascending=False).head(N_TOP_PROTEINS_HEATMAP).index

    # With thousands of samples, a heatmap with one column per sample is
    # unreadable regardless of how strong the underlying signal is - this
    # is a display legibility limit, not a data problem. Subsample a
    # manageable number per phenotype group instead of plotting everyone.
    if rng is None:
        rng = np.random.default_rng(0)
    sampled_ids = []
    for pheno_value, group in metadata.groupby("phenotype"):
        candidates = [s for s in group.index if s in matrix.columns]
        n = min(max_samples_per_group, len(candidates))
        sampled_ids.extend(rng.choice(candidates, size=n, replace=False))

    sub = matrix.loc[top_proteins, sampled_ids]

    # Impute missing values with that protein's own mean, for display only.
    sub_filled = sub.apply(lambda row: row.fillna(row.mean()), axis=1)

    # Z-score each protein (row) so the heatmap shows relative variation,
    # not raw abundance differences between proteins.
    z = sub_filled.sub(sub_filled.mean(axis=1), axis=0).div(sub_filled.std(axis=1), axis=0)

    # Order samples by phenotype so the two groups appear as clean blocks.
    sample_order = metadata.loc[z.columns].sort_values("phenotype").index
    z = z[sample_order]

    phenotype_colors = metadata.loc[sample_order, "phenotype"].map({0: "#8ecae6", 1: "#e76f51"})

    row_labels = gene_symbols.loc[z.index]

    g = sns.clustermap(
        z,
        row_cluster=True,
        col_cluster=False,
        cmap="vlag",
        center=0,
        col_colors=phenotype_colors,
        yticklabels=row_labels,
        xticklabels=False,
        figsize=(12, 10),
        cbar_kws={"label": "Z-scored log2 intensity"},
    )
    g.ax_heatmap.set_xlabel(f"Samples ({len(sampled_ids)} of {matrix.shape[1]} shown, {max_samples_per_group}/group, ordered by phenotype)")
    g.ax_heatmap.set_ylabel("Protein (gene symbol)")
    g.fig.suptitle(
        f"Top {N_TOP_PROTEINS_HEATMAP} proteins by phenotype effect size",
        y=1.02, fontsize=13,
    )

    # Legend for the phenotype color strip
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor="#8ecae6", label="Phenotype = 0"),
                       Patch(facecolor="#e76f51", label="Phenotype = 1")]
    g.ax_heatmap.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.15, 1.15), frameon=False)

    g.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def plot_top_protein_boxplot(matrix: pd.DataFrame, metadata: pd.DataFrame, effects: pd.DataFrame,
                              gene_symbols: pd.Series, out_path: Path):
    top_protein = effects["beta_phenotype"].abs().idxmax()
    gene_name = gene_symbols.get(top_protein, top_protein)

    values = matrix.loc[top_protein].dropna()
    df = metadata.loc[values.index].copy()
    df["intensity"] = values.values
    if "site" not in df.columns:
        df["site"] = [s.split("_")[0] for s in df.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="site", y="intensity", hue="phenotype", palette=["#8ecae6", "#e76f51"], ax=ax)
    sns.stripplot(data=df, x="site", y="intensity", hue="phenotype", dodge=True,
                   color="black", alpha=0.4, size=3, ax=ax, legend=False)
    ax.set_title(f"{gene_name} ({top_protein}) intensity by phenotype and site\n"
                 f"(injected beta_phenotype = {effects.loc[top_protein, 'beta_phenotype']:.2f})")
    ax.set_xlabel("Site")
    ax.set_ylabel("log2 intensity")
    ax.legend(title="Phenotype", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate presentation plots from synthetic proteomics data")
    parser.add_argument("--data-dir", type=str, default="synthetic_proteomics_chr22",
                         help="Folder containing the site CSVs, sample_metadata.csv and protein_covariate_effects.csv")
    parser.add_argument("--out-dir", type=str, default="plots",
                         help="Where to save the generated PNG files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix, metadata, effects, gene_symbols = load_all(data_dir)

    heatmap_path = out_dir / "proteomics_heatmap_top_signal.png"
    plot_heatmap(matrix, metadata, effects, gene_symbols, heatmap_path, rng=np.random.default_rng(0))
    print(f"Saved heatmap -> {heatmap_path}")

    boxplot_path = out_dir / "top_phenotype_protein_boxplot.png"
    plot_top_protein_boxplot(matrix, metadata, effects, gene_symbols, boxplot_path)
    print(f"Saved boxplot -> {boxplot_path}")


if __name__ == "__main__":
    main()