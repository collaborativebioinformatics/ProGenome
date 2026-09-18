#!/usr/bin/env python3
"""Synthetic proteomics for the *real* 1000 Genomes individuals in the knowledge graph.

Why: the graph's people are 1000G IDs (HG00096, ...).  Proteomics keyed to the same IDs is
what makes `Individual -MEASURED-> Protein` edges exist.  Until real per-individual
proteomics is in hand, this generator (same spirit as proteomics/generate_synthetic_proteomics.py)
produces log2 intensities for the chr22 proteins with a KNOWN ground truth, so the
genome+proteome model can be scored on whether it recovers it:

  * 3 hospital sites, mixed ancestry (so `site` is a pure batch label -> negative control)
  * phenotype (case/control) depends on a few "causal" haploblock clusters (+ age)   <- genomic signal
  * each causal cluster shifts one protein encoded in the same block (cis-pQTL-like)  <- genome->proteome link
  * proteins also respond to phenotype / age / sex (like Nolan's generator)
  * per-site batch shift + detection-limit missingness (the whiteboard's harmoniser problems)

Outputs (outputs/proteomics_synth/<chrom>/): site{1,2,3}_proteomics_log2.csv (rows proteins, cols samples,
same layout as proteomics/synthetic_proteomics_chr22), sample_metadata.csv, measured_long.csv
(individual_id, protein_id, site, log2_intensity) and ground_truth.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import haplokg


def load_protein_bed(path: Path) -> pd.DataFrame:
    bed = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "uniprot"])
    bed["protein_id"] = bed["uniprot"].str.split("-").str[0]
    # one row per protein: the union span of its isoforms
    return bed.groupby(["chrom", "protein_id"], as_index=False).agg(start=("start", "min"), end=("end", "max"))


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--bed", type=Path, default=here.parent / "proteomics" / "uniprot_chr22.bed")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-sites", type=int, default=3)
    parser.add_argument("--n-causal", type=int, default=20, help="causal clusters driving the phenotype")
    parser.add_argument("--prevalence", type=float, default=0.35)
    parser.add_argument("--causal-beta-sd", type=float, default=1.5, help="effect size sd of causal clusters on the phenotype logit")
    parser.add_argument("--cis-beta-sd", type=float, default=1.0, help="effect size sd of a causal cluster on its cis protein (log2)")
    parser.add_argument("--pheno-protein-frac", type=float, default=0.15, help="fraction of proteins that respond to the phenotype")
    parser.add_argument("--pheno-effect-sd", type=float, default=0.5, help="effect size sd of the phenotype on responding proteins")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = args.out_dir or here / "outputs" / "proteomics_synth" / args.chrom
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    kg = haplokg.load_kg(here / "outputs" / "kg" / args.chrom)
    ind, clusters, blocks, carries = kg["individuals"], kg["clusters"], kg["blocks"], kg["carries"]
    labelled = ind["ancestry_code"].to_numpy() >= 0
    people = ind[labelled].reset_index(drop=True)
    X = carries[labelled].toarray().astype(np.float32)                 # people x clusters

    # ---- sites: random, stratified on ancestry so every site is mixed --------------------
    site = np.empty(len(people), dtype=object)
    for anc in people["ancestry"].unique():
        idx = np.flatnonzero(people["ancestry"] == anc)
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, args.n_sites)):
            site[chunk] = f"SITE{k + 1}"
    age = rng.integers(18, 86, size=len(people))
    sex = (people["sex"] == "male").astype(int).to_numpy()

    # ---- proteins and which block encodes them ------------------------------------------
    prot = load_protein_bed(args.bed)
    prot = prot[prot["chrom"] == args.chrom].reset_index(drop=True)
    b = blocks[["block_idx", "start", "end"]].to_numpy()
    prot["block_idx"] = [
        int(b[(b[:, 1] <= e) & (b[:, 2] >= s)][:, 0][0]) if ((b[:, 1] <= e) & (b[:, 2] >= s)).any() else -1
        for s, e in zip(prot["start"], prot["end"])
    ]
    proteins = prot["protein_id"].tolist()
    P = len(proteins)

    # ---- causal clusters: in blocks that encode a protein, moderately common --------------
    eligible = clusters[(clusters["block_idx"].isin(prot["block_idx"])) & (clusters["support_frac"].between(0.05, 0.6))]
    causal = eligible.sample(n=min(args.n_causal, len(eligible)), random_state=args.seed)
    beta_pheno = rng.normal(0, args.causal_beta_sd, size=len(causal))
    # each causal cluster shifts one protein from its own block (cis effect)
    causal_protein = []
    for blk in causal["block_idx"]:
        choices = prot.index[prot["block_idx"] == blk].tolist()
        causal_protein.append(int(rng.choice(choices)))
    beta_cis = rng.normal(0, args.cis_beta_sd, size=len(causal))

    # ---- phenotype from genotype (+ age), calibrated to the requested prevalence ----------
    G = X[:, causal["cluster_idx"].to_numpy()]                          # people x causal (0/1)
    logit = G @ beta_pheno + 0.02 * (age - age.mean())
    intercept = np.quantile(logit, 1 - args.prevalence)                  # top `prevalence` fraction become cases (softly)
    p = 1 / (1 + np.exp(-(logit - intercept)))
    phenotype = (rng.random(len(people)) < p).astype(int)

    # ---- protein intensities -----------------------------------------------------------
    baseline = rng.uniform(6.0, 16.0, size=P)
    b_age, b_sex = rng.normal(0, 0.5, P), rng.normal(0, 0.5, P)
    responds = rng.random(P) < args.pheno_protein_frac                     # only some proteins track the phenotype
    b_pheno = np.where(responds, rng.normal(0, args.pheno_effect_sd, P), 0.0)
    site_shift = {f"SITE{k + 1}": rng.normal(0, 0.3, size=P) for k in range(args.n_sites)}   # batch effect
    age_z = (age - age.mean()) / age.std()
    M = (baseline[None, :] + np.outer(age_z, b_age) + np.outer(sex, b_sex) + np.outer(phenotype, b_pheno)
         + rng.normal(0, 0.8, size=(len(people), P)) + rng.normal(0, 0.3, size=(len(people), P)))
    for j, (pi, bc) in enumerate(zip(causal_protein, beta_cis)):
        M[:, pi] += bc * G[:, j]
    for k in range(len(people)):
        M[k] += site_shift[site[k]]
    scarcity = (baseline.max() - baseline) / (baseline.max() - baseline.min() + 1e-9)
    missing = rng.random(M.shape) < 0.15 * scarcity[None, :]           # MNAR at the detection limit
    M = np.round(M, 3); M[missing] = np.nan

    # ---- write ------------------------------------------------------------------------
    meta = pd.DataFrame({"sample_id": people["individual_id"], "site": site, "age": age, "sex": sex,
                         "phenotype": phenotype, "ancestry": people["ancestry"]})
    meta.to_csv(out_dir / "sample_metadata.csv", index=False)
    for s in sorted(set(site)):
        cols = meta.index[meta["site"] == s]
        frame = pd.DataFrame(M[cols].T, index=proteins, columns=meta.loc[cols, "sample_id"])
        frame.index.name = "protein_id"
        frame.to_csv(out_dir / f"{s.lower()}_proteomics_log2.csv")
    long = pd.DataFrame(M, index=meta["sample_id"], columns=proteins).stack(future_stack=True).dropna().reset_index()
    long.columns = ["individual_id", "protein_id", "log2_intensity"]
    long = long.merge(meta[["sample_id", "site"]], left_on="individual_id", right_on="sample_id").drop(columns="sample_id")
    long.to_csv(out_dir / "measured_long.csv", index=False)
    truth = {
        "seed": args.seed, "n_people": int(len(people)), "n_proteins": P, "prevalence_observed": float(phenotype.mean()),
        "sites": {s: int((site == s).sum()) for s in sorted(set(site))},
        "causal_clusters": [{"cluster_id": c, "cluster_idx": int(i), "beta_phenotype": float(bp),
                             "cis_protein": proteins[pi], "beta_cis": float(bc)}
                            for c, i, bp, pi, bc in zip(causal["cluster_id"], causal["cluster_idx"], beta_pheno, causal_protein, beta_cis)],
        "protein_effects": {"beta_age": b_age.tolist(), "beta_sex": b_sex.tolist(), "beta_phenotype": b_pheno.tolist(), "proteins": proteins},
        "site_shift_sd": 0.3, "missing_rate_at_lod": 0.15, "n_phenotype_responsive_proteins": int(responds.sum()),
        "params": vars(args) | {"bed": str(args.bed), "out_dir": str(out_dir)},
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"{len(people)} people, {P} proteins, sites {truth['sites']}, cases {phenotype.mean():.2f}, "
          f"{len(causal)} causal clusters, {long.shape[0]:,} measured values -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
