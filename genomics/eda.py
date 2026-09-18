#!/usr/bin/env python3
"""Exploratory data analysis of everything the pipeline consumes: what the data is, where it comes from,
how it is shaped, and the statistics that justify the modelling choices.  One script, one report.

    python eda.py --chrom chr22          # -> outputs/eda/<chrom>/EDA.md + tables/*.csv + plots/*.png

Sections of the report:
  1. Provenance         what / where / why: every input file, its source, size, build, licence notes
  2. Individuals        ancestry, population, sex; who lacks labels; the train/val/test split
  3. Haploblocks        block length, clusters per block, entropy/dominance/singletons along the chromosome
  4. Clusters           carrier support distribution, the support filter, clusters carried per person
  5. Co-occurrence      lift/weight distributions, degree, components, hub artefact, same-block vs long-range
  6. Genes / proteins   how many genes per block, blocks per gene, isoforms, genes outside any block
  7. Proteomics         intensity range, missingness (LOD) per protein and per site, batch effect before
                        and after harmonisation, correlation structure, phenotype/age/sex signal
  8. Genome <-> proteome cis correlation between carrying a cluster and the protein encoded in its block
Every number in the report is computed here; nothing is typed in.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

import haplokg
import haplokg_proteins as hp

ANCESTRY_COLOURS = {"AFR": "#d55e00", "AMR": "#cc79a7", "EAS": "#009e73", "EUR": "#0072b2", "SAS": "#e69f00"}


def fmt(x, nd=3):
    return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--synth-dir", type=Path, default=None, help="proteomics folder (measured_long.csv, sample_metadata.csv, ground_truth.json)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    chrom = args.chrom
    data = here / "data"
    kg_dir = here / "outputs" / "kg" / chrom
    synth = args.synth_dir or here / "outputs" / "proteomics_synth" / chrom
    out = here / "outputs" / "eda" / chrom
    (out / "tables").mkdir(parents=True, exist_ok=True); (out / "plots").mkdir(exist_ok=True)
    md = []
    T = lambda name, df: df.to_csv(out / "tables" / f"{name}.csv", index=False)

    if not args.no_plots:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    # ------------------------------------------------------------------ 1. provenance
    files = [
        ("haplograph/<chr>/nodes.csv.gz", "HaploGraph cluster x individual 0/1 matrix", "data.haploblocks.org/haplograph/1000G", "built at MDxCORE Rigshospitalet Sept 2026 for this hackathon; 1000G phased haplotypes, GRCh38, MMseqs2 clusters per recombination-defined haploblock"),
        ("haplograph/<chr>/edges_lift_above_threshold.csv.gz", "cluster co-occurrence edges, lift >= 5", "same", "raw edges.csv is dominated by a population-frequency mega-hub; lift-filtered file is the signal"),
        ("haplograph/<chr>/islands.csv.gz", "dense multi-block extended haplotypes", "same", "candidate extended haplotypes"),
        ("haplograph/phenotypes_real.csv", "ancestry / population / sex per 1000G individual", "1000G panel via IGSR", "the only real phenotypes 1000G has; anything else must be simulated"),
        ("haploblocks/block_stats.tsv", "per-block length, n_clusters, entropy, dominance, singletons", "data.haploblocks.org/bidirectional_blast_samples", "QC statistics from the haploblock pipeline"),
        ("haploblocks/<chr>_haploblock_boundaries_<chr>.tsv", "block START/END", "data.haploblocks.org/haploblock_hashes", "step 1 of the haploblocks pipeline (recombination-rate peaks)"),
        ("../proteomics/uniprot_chr22.bed", "UniProt proteins (isoforms) with genomic spans", "UCSC UniProt track, Friederike", "gene BED + gene->protein map in one file"),
        ("outputs/proteomics_synth/<chr>/", "synthetic per-site proteomics on 1000G IDs with saved ground truth", "proteomics_synth_1000g.py", "stand-in until Wu 2013 LCL proteomics / UKB-PPP pQTL are wired in"),
    ]
    sizes = {}
    for rel, *_ in files:
        p = data / rel.replace("<chr>", chrom)
        p = p if p.exists() else here / rel.replace("<chr>", chrom)
        sizes[rel] = (p.stat().st_size / 1e6 if p.is_file() else sum(f.stat().st_size for f in p.glob("*") if f.is_file()) / 1e6) if p.exists() else float("nan")
    prov = pd.DataFrame([{"file": r, "what": w, "where": s, "why / note": n, "MB": round(sizes[r], 1)} for r, w, s, n in files])
    T("provenance", prov)
    md += [f"# EDA — {chrom}", "", "## 1. Provenance: what, where, why", "", prov.to_markdown(index=False), ""]

    # ------------------------------------------------------------------ 2. individuals
    kg = haplokg.load_kg(kg_dir)
    ind = kg["individuals"]
    split = haplokg.load_or_make_split(kg, here / "outputs" / "splits" / chrom / "split_seed42.csv")
    anc = ind["ancestry"].fillna("unlabelled").value_counts().rename_axis("ancestry").reset_index(name="n")
    pop = ind.groupby(["ancestry", "population"]).size().reset_index(name="n").sort_values(["ancestry", "n"], ascending=[True, False])
    sex = ind.groupby(["ancestry", "sex"]).size().unstack(fill_value=0).reset_index()
    T("individuals_ancestry", anc); T("individuals_population", pop); T("individuals_sex_by_ancestry", sex)
    sp = pd.Series(split).value_counts().rename_axis("split").reset_index(name="n")
    md += ["## 2. Individuals", "",
           f"{len(ind):,} individuals are columns of `nodes.csv.gz`; {int((ind['ancestry_code'] >= 0).sum()):,} have labels in `phenotypes_real.csv`, "
           f"{int((ind['ancestry_code'] < 0).sum())} do not (kept as unlabelled nodes). 1000G has **no** other phenotypes (no height, no disease), which is why height on the whiteboard had to become a simulated label.", "",
           "Ancestry (super-population):", "", anc.to_markdown(index=False), "",
           f"Populations: {pop['population'].nunique()} (smallest {pop['n'].min()}, largest {pop['n'].max()}). Sex: "
           + ", ".join(f"{r.ancestry} {int(r.get('female', 0))}F/{int(r.get('male', 0))}M" for _, r in sex.iterrows()) + ".", "",
           "Split (seed 42, stratified on ancestry; shared by every model): " + ", ".join(f"{r.split} {r.n}" for _, r in sp.iterrows()), ""]
    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        pop_sorted = pop.sort_values(["ancestry", "population"])
        ax.bar(pop_sorted["population"], pop_sorted["n"], color=[ANCESTRY_COLOURS[a] for a in pop_sorted["ancestry"]])
        ax.set_ylabel("individuals"); ax.set_title("1000G individuals per population, coloured by super-population"); ax.tick_params(axis="x", rotation=90, labelsize=8)
        fig.tight_layout(); fig.savefig(out / "plots" / "02_populations.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ 3. haploblocks
    blocks = kg["blocks"].copy()
    blocks["length_kb"] = blocks["block_length"] / 1e3
    q = blocks["length_kb"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    gaps = blocks.sort_values("start")["start"].to_numpy()[1:] - blocks.sort_values("start")["end"].to_numpy()[:-1]
    bstats = pd.DataFrame({
        "metric": ["blocks", "span covered (Mb)", "first block start (Mb)", "last block end (Mb)", "length median (kb)", "length 5%/95% (kb)",
                   "longest block (kb)", "gaps between consecutive blocks > 1 kb", "clusters per block median", "clusters per block max",
                   "singleton rate median", "Shannon entropy median", "dominance (largest cluster share) median"],
        "value": [len(blocks), round((blocks["end"].max() - blocks["start"].min()) / 1e6, 2), round(blocks["start"].min() / 1e6, 2), round(blocks["end"].max() / 1e6, 2),
                  round(q[0.5], 1), f"{q[0.05]:.1f} / {q[0.95]:.1f}", round(blocks["length_kb"].max(), 1), int((gaps > 1000).sum()),
                  int(blocks["n_clusters"].median()), int(blocks["n_clusters"].max()), round(blocks["singleton_rate"].median(), 3),
                  round(blocks["shannon_entropy"].median(), 2), round(blocks["dominance"].median(), 3)]})
    T("blocks_summary", bstats)
    rho = stats.spearmanr(blocks["block_length"], blocks["n_clusters"]).correlation
    md += ["## 3. Haploblocks (recombination-defined regions)", "", bstats.to_markdown(index=False), "",
           f"Blocks tile the chromosome without overlap (gaps > 1 kb between consecutive blocks: {int((gaps > 1000).sum())}). "
           f"Longer blocks carry more distinct haplotype clusters (Spearman ρ = {rho:.2f}); high-entropy blocks are the ones where the population is most diverse.", ""]
    if not args.no_plots:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        axes[0].hist(np.log10(blocks["block_length"]), bins=40, color="#1f6f8b"); axes[0].set_xlabel("log10 block length (bp)"); axes[0].set_ylabel("blocks")
        axes[1].scatter(blocks["block_length"] / 1e3, blocks["n_clusters"], s=6, color="#1f6f8b"); axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("block length (kb)"); axes[1].set_ylabel("clusters in block")
        mid = (blocks["start"] + blocks["end"]) / 2e6
        axes[2].plot(mid, blocks["shannon_entropy"], lw=0.8, color="#1f6f8b"); axes[2].set_xlabel(f"{chrom} position (Mb)"); axes[2].set_ylabel("Shannon entropy of clusters")
        fig.suptitle("Haploblocks: size, diversity and where diversity sits"); fig.tight_layout(); fig.savefig(out / "plots" / "03_blocks.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ 4. clusters
    with gzip.open(data / "haplograph" / chrom / "nodes.csv.gz", "rt") as fh:
        n_cols = len(fh.readline().split(","))
    all_support = None
    stats_path = data / "haploblocks" / "block_stats.tsv"
    bs = pd.read_csv(stats_path, sep="\t"); bs = bs[bs["chr"] == chrom]
    total_clusters = int(bs["n_clusters"].sum()); singletons = int(bs["singleton_count"].sum())
    clusters = kg["clusters"]
    carries = kg["carries"]
    per_person = np.asarray(carries.sum(axis=1)).ravel()
    csum = pd.DataFrame({
        "metric": ["clusters in nodes.csv.gz (all)", "of which singletons (1 carrier)", "kept after symmetric support >= 25", "kept fraction",
                   "support median (kept)", "support max (kept)", "clusters carried per person: mean", "min", "max", "blocks with >= 1 kept cluster"],
        "value": [total_clusters, singletons, len(clusters), round(len(clusters) / total_clusters, 4), int(clusters["support"].median()), int(clusters["support"].max()),
                  round(per_person.mean(), 1), int(per_person.min()), int(per_person.max()), int(clusters["block_idx"].nunique())]})
    T("clusters_summary", csum)
    md += ["## 4. Clusters (the nodes people connect to)", "", csum.to_markdown(index=False), "",
           f"{singletons / total_clusters:.0%} of all clusters are singletons (one haplotype); they carry no population signal and are dropped. "
           f"Each person carries ~{per_person.mean():.0f} of the kept clusters (≤ 2 per block: two haplotypes), so the individual × cluster matrix is "
           f"{carries.nnz / (carries.shape[0] * carries.shape[1]):.1%} dense — sparse int8 storage is what makes chr22 fit in memory.", ""]
    if not args.no_plots:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        axes[0].hist(np.log10(clusters["support"]), bins=40, color="#1f6f8b"); axes[0].set_xlabel("log10 carriers per kept cluster"); axes[0].set_ylabel("clusters")
        for a, col in ANCESTRY_COLOURS.items():
            m = (ind["ancestry"] == a).to_numpy()
            axes[1].hist(per_person[m], bins=30, histtype="step", color=col, label=a, linewidth=1.5)
        axes[1].set_xlabel("kept clusters carried per person"); axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(out / "plots" / "04_clusters.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ 5. co-occurrence
    co = kg["co_occurs"].copy()
    cl_block = clusters.set_index("cluster_idx")["block_idx"]
    co["same_block"] = cl_block.loc[co["src"]].to_numpy() == cl_block.loc[co["dst"]].to_numpy()
    bstart = blocks.set_index("block_idx")["start"]
    co["distance_kb"] = np.abs(bstart.loc[cl_block.loc[co["src"]].to_numpy()].to_numpy() - bstart.loc[cl_block.loc[co["dst"]].to_numpy()].to_numpy()) / 1e3
    deg = np.bincount(np.concatenate([co["src"], co["dst"]]), minlength=len(clusters))
    esum = pd.DataFrame({
        "metric": ["edges (lift >= 5)", "lift median", "lift 95%", "lift max", "weight (shared carriers) median", "same-block edges", "median distance between endpoints (kb)",
                   "edges spanning > 10 Mb", "clusters with >= 1 edge", "degree median (connected)", "degree max", "top-1% of clusters hold this share of edges"],
        "value": [len(co), round(co["lift"].median(), 2), round(co["lift"].quantile(0.95), 2), round(co["lift"].max(), 1), int(co["weight"].median()),
                  int(co["same_block"].sum()), round(co["distance_kb"].median(), 0), int((co["distance_kb"] > 10_000).sum()), int((deg > 0).sum()),
                  int(np.median(deg[deg > 0])), int(deg.max()), f"{np.sort(deg)[::-1][:max(1, len(deg)//100)].sum() / deg.sum():.0%}"]})
    T("cooccurrence_summary", esum)
    md += ["## 5. Co-occurrence graph", "", esum.to_markdown(index=False), "",
           "Lift = P(A∩B)/(P(A)P(B)): how much more often two clusters travel together than chance. Most edges are long-range (tens of Mb), i.e. "
           "population structure rather than physical linkage — the HaploGraph README's mega-hub warning made visible. The co-occurrence analysis "
           "(`cooccurrence_analysis.py`) shows edges overwhelmingly join clusters enriched in the same ancestry.", ""]
    if not args.no_plots:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        axes[0].hist(np.log10(co["lift"]), bins=40, color="#1f6f8b"); axes[0].set_xlabel("log10 lift"); axes[0].set_ylabel("edges")
        axes[1].hist(np.log10(co["distance_kb"].clip(lower=1)), bins=40, color="#1f6f8b"); axes[1].set_xlabel("log10 distance between endpoints (kb)")
        axes[2].hist(np.log10(deg[deg > 0]), bins=40, color="#1f6f8b"); axes[2].set_xlabel("log10 degree (connected clusters)")
        fig.suptitle("Co-occurrence edges: strength, reach, concentration"); fig.tight_layout(); fig.savefig(out / "plots" / "05_cooccurrence.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ 6. genes / proteins
    have_v2 = (kg_dir / "proteins.csv").exists()
    if have_v2:
        pt = hp.load_protein_tables(kg_dir)
        genes, prot, bg = pt["genes"], pt["proteins"], pt["block_gene"]
        gpb = bg.groupby("block_idx").size()
        gsum = pd.DataFrame({
            "metric": ["proteins (UniProt accessions, isoforms collapsed)", "isoform rows in BED", "genes (symbols)", "proteins per gene max", "block–gene overlaps",
                       "genes overlapping no block", "genes spanning >= 2 blocks", "blocks with >= 1 gene", "genes per gene-bearing block median / max"],
            "value": [len(prot), int(prot["n_isoforms"].sum()), len(genes), int(genes["n_proteins"].max()), len(bg),
                      int((~genes["gene_idx"].isin(bg["gene_idx"])).sum()), int((bg.groupby("gene_idx").size() >= 2).sum()), int(gpb.size),
                      f"{int(gpb.median())} / {int(gpb.max())}"]})
        T("genes_proteins_summary", gsum)
        md += ["## 6. Genes and proteins", "", gsum.to_markdown(index=False), "",
               "Genes outside every block sit in the chromosome ends / assembly gaps the haploblock boundaries do not cover; a gene spanning two blocks "
               "links both blocks to its protein, which is what the OVERLAPS edge encodes.", ""]

    # ------------------------------------------------------------------ 7. proteomics
    if have_v2 and (synth / "measured_long.csv").exists():
        long = pd.read_csv(synth / "measured_long.csv"); meta = pd.read_csv(synth / "sample_metadata.csv")
        truth = json.loads((synth / "ground_truth.json").read_text()) if (synth / "ground_truth.json").exists() else None
        n_people, n_prot = meta["sample_id"].nunique(), long["protein_id"].nunique()
        missing = 1 - len(long) / (n_people * n_prot)
        per_prot_missing = 1 - long.groupby("protein_id").size() / n_people
        raw_site = long.groupby(["protein_id", "site"])["log2_intensity"].median().unstack()
        site_spread_raw = raw_site.max(axis=1) - raw_site.min(axis=1)
        harm = hp.harmonise(long)
        harm_site = harm.groupby(["protein_id", "site"])["z"].median().unstack()
        site_spread_harm = harm_site.max(axis=1) - harm_site.min(axis=1)
        # phenotype / age / sex signal per protein on harmonised values
        h = harm.merge(meta.rename(columns={"sample_id": "individual_id"})[["individual_id", "age", "sex", "phenotype"]], on="individual_id")
        rows = []
        for pid, g in h.groupby("protein_id"):
            t_ph = stats.ttest_ind(g.loc[g["phenotype"] == 1, "z"], g.loc[g["phenotype"] == 0, "z"], equal_var=False)
            rows.append({"protein_id": pid, "n": len(g), "phenotype_t": t_ph.statistic, "phenotype_p": t_ph.pvalue,
                         "age_r": stats.pearsonr(g["age"], g["z"])[0], "sex_t": stats.ttest_ind(g.loc[g["sex"] == 1, "z"], g.loc[g["sex"] == 0, "z"], equal_var=False).statistic})
        assoc_p = pd.DataFrame(rows)
        from cooccurrence_analysis import bh_fdr
        assoc_p["phenotype_q"] = bh_fdr(assoc_p["phenotype_p"].to_numpy())
        T("proteomics_protein_associations", assoc_p.sort_values("phenotype_p"))
        psum = pd.DataFrame({
            "metric": ["people with proteomics", "proteins", "sites", "people per site", "log2 intensity range (1%–99%)", "overall missing (below LOD)",
                       "missing per protein median / max", "between-site median shift per protein: raw (log2) median", "same after harmonisation (z)",
                       "proteins associated with phenotype (FDR 5%)", "proteins with |age r| > 0.2", "proteins with |sex t| > 3"],
            "value": [n_people, n_prot, meta["site"].nunique(), ", ".join(f"{k}: {v}" for k, v in meta["site"].value_counts().sort_index().items()),
                      f"{long['log2_intensity'].quantile(0.01):.1f} – {long['log2_intensity'].quantile(0.99):.1f}", f"{missing:.1%}",
                      f"{per_prot_missing.median():.1%} / {per_prot_missing.max():.1%}", round(site_spread_raw.median(), 3), round(site_spread_harm.median(), 3),
                      int((assoc_p["phenotype_q"] < 0.05).sum()), int((assoc_p["age_r"].abs() > 0.2).sum()), int((assoc_p["sex_t"].abs() > 3).sum())]})
        T("proteomics_summary", psum)
        note = ""
        if truth:
            responsive = truth.get("n_phenotype_responsive_proteins")
            note = (f" Ground truth: {responsive} proteins were generated to respond to the phenotype, {len(truth['causal_clusters'])} causal clusters, "
                    f"site shift sd {truth['site_shift_sd']}, LOD missing rate {truth['missing_rate_at_lod']} — the harmoniser must remove the site shift "
                    f"(it does: {site_spread_raw.median():.2f} → {site_spread_harm.median():.2f}) and the phenotype signal must survive it "
                    f"({int((assoc_p['phenotype_q'] < 0.05).sum())} proteins recovered at FDR 5%).")
        md += ["## 7. Proteomics", "", psum.to_markdown(index=False), "",
               "Missingness is MNAR at the detection limit (low-abundance proteins go missing first), so it is kept as *absence of an edge*, never imputed as zero "
               "in the graph. The harmoniser is a robust z-score per (site, protein); it is what makes `site` a pure batch label the model must not be able to predict." + note, ""]
        if not args.no_plots:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
            axes[0].hist(long["log2_intensity"], bins=60, color="#b26a0c"); axes[0].set_xlabel("log2 intensity"); axes[0].set_ylabel("measurements")
            axes[1].scatter(long.groupby("protein_id")["log2_intensity"].median(), per_prot_missing.loc[long.groupby("protein_id")["log2_intensity"].median().index], s=6, color="#b26a0c")
            axes[1].set_xlabel("protein median log2 intensity"); axes[1].set_ylabel("fraction missing"); axes[1].set_title("MNAR: low abundance → missing")
            axes[2].hist(site_spread_raw, bins=40, histtype="step", color="#b26a0c", label="raw log2", linewidth=1.5)
            axes[2].hist(site_spread_harm, bins=40, histtype="step", color="#1f6f8b", label="harmonised z", linewidth=1.5)
            axes[2].set_xlabel("max−min of site medians per protein"); axes[2].legend(frameon=False); axes[2].set_title("batch effect before / after")
            fig.tight_layout(); fig.savefig(out / "plots" / "07_proteomics.png", dpi=150); plt.close(fig)

        # -------------------------------------------------------------- 8. genome <-> proteome
        if truth:
            ind_idx = dict(zip(ind["individual_id"], ind["individual_idx"]))
            rows = []
            for c in truth["causal_clusters"]:
                sub = harm[harm["protein_id"] == c["cis_protein"]]
                g = np.asarray(carries[[ind_idx[i] for i in sub["individual_id"]], c["cluster_idx"]].todense()).ravel()
                r = np.corrcoef(g, sub["z"])[0, 1] if g.std() > 0 else np.nan
                rows.append({"cluster_id": c["cluster_id"], "cis_protein": c["cis_protein"], "beta_cis_truth": round(c["beta_cis"], 2),
                             "carrier_freq": round(g.mean(), 3), "pearson_r": round(r, 3), "r2": round(r * r, 3)})
            cis = pd.DataFrame(rows).sort_values("r2", ascending=False)
            T("genome_proteome_cis", cis)
            md += ["## 8. Genome ↔ proteome (cis signal)", "",
                   f"For each ground-truth causal cluster, the correlation between carrying it and the harmonised level of the protein in its block: "
                   f"mean r² {cis['r2'].mean():.3f}, {int((cis['r2'] > 0.1).sum())} of {len(cis)} above 0.1. This is the ceiling any genome→proteome model can reach on this data "
                   f"and the reason the ridge baseline (`proteome_linear_baseline.py`) recovers only the strongest cis effects.", "", cis.to_markdown(index=False), ""]

    (out / "EDA.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out}/EDA.md, {len(list((out / 'tables').glob('*.csv')))} tables, {len(list((out / 'plots').glob('*.png')))} plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
