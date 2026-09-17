"""
Generates synthetic proteomic intensity matrices, one per hospital site,
for all genes found in a chr22 BED file, with UniProt IDs converted to
gene symbols via the UniProt REST API.

Each sample also gets age/sex/phenotype metadata, and protein intensities
are simulated as a function of these covariates: every protein gets its
own randomly-drawn effect size per covariate (most small, a few strong),
so the resulting data has real, recoverable signal rather than pure noise.

Output orientation: rows = proteins, columns = samples.

Each sample has a sample_id that should match an individual on the
genomics side, so the two can be linked later.
"""

import argparse
import numpy as np
import pandas as pd
import requests
from pathlib import Path

# --- Parameters ------------------------------------------------------------

RNG_SEED = 42
N_SITES = 3
N_SAMPLES_PER_SITE = 40
OUTPUT_DIR = Path("./synthetic_proteomics_chr22")

BIOLOGICAL_SD = 0.8      # inter-individual variance (unexplained by covariates)
TECHNICAL_SD = 0.3       # technical noise (instrument, run)
MISSING_RATE_AT_LOD = 0.15  # missing probability for the least abundant protein
BASELINE_MIN = 6.0        # log2 intensity range covering plasma proteome dynamic range
BASELINE_MAX = 16.0

# Covariate effect sizes: how strongly age/sex/phenotype shift a protein's
# log2 intensity. Drawn per protein from a normal distribution centered on
# 0, so most proteins get a small effect and a few get a strong one -
# mirrors real biology (covariate effects are heterogeneous across the
# proteome, not all-or-nothing).
AGE_EFFECT_SD = 0.5
SEX_EFFECT_SD = 0.5
PHENOTYPE_EFFECT_SD = 0.6

AGE_MIN = 18
AGE_MAX = 85
PHENOTYPE_PREVALENCE = 0.4  # fraction of samples with phenotype = 1

UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_CHUNK_SIZE = 90  # accessions per request, kept well under URL length limits


def load_proteins_from_bed(bed_path: Path) -> list:
    """
    Reads a BED12-style file (chrom, start, end, name, score, strand, ...)
    and returns the list of unique base protein IDs. UniProt isoform
    suffixes (e.g. Q9BXF3-1, Q9BXF3-2) are collapsed to the base ID
    (Q9BXF3), since they represent the same gene/protein.
    """
    df = pd.read_csv(bed_path, sep="\t", header=None, usecols=[3], names=["name"])
    base_ids = df["name"].str.split("-").str[0]
    return sorted(base_ids.unique())


def fetch_gene_symbols(protein_ids: list, cache_path: Path) -> dict:
    """
    Maps UniProt accession IDs to gene symbols using UniProt's search/stream
    REST endpoint (querying accession:ID1 OR accession:ID2 ... in batches
    and asking for the gene_names field directly).

    Results are cached to disk so re-running the script doesn't re-query
    UniProt every time. Falls back to using the UniProt ID itself (no
    internet, API change, or a genuinely unnamed entry) rather than
    failing the whole script.
    """
    cached = {}
    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col="protein_id")["gene_symbol"].to_dict()

    missing = [p for p in protein_ids if p not in cached]
    for i in range(0, len(missing), UNIPROT_CHUNK_SIZE):
        chunk = missing[i:i + UNIPROT_CHUNK_SIZE]
        query = " OR ".join(f"accession:{pid}" for pid in chunk)
        try:
            response = requests.get(
                UNIPROT_STREAM_URL,
                params={"query": query, "fields": "accession,gene_names", "format": "tsv"},
                timeout=30,
            )
            response.raise_for_status()
            lines = response.text.strip().split("\n")[1:]  # skip header row
            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1]:
                    cached[parts[0]] = parts[1].split()[0]  # first gene symbol if several listed
        except Exception as exc:
            print(f"Warning: UniProt gene symbol lookup failed for a batch ({exc}); "
                  f"affected IDs will fall back to their UniProt accession.")

    for p in protein_ids:
        cached.setdefault(p, p)

    pd.Series(cached, name="gene_symbol").rename_axis("protein_id").to_csv(cache_path)
    return cached


def generate_metadata(sample_ids: list, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generates age / sex / phenotype for each sample. These are the ground
    truth covariates that protein intensities are simulated against.

    sex: 0 = female, 1 = male.
    phenotype: 0 = control, 1 = case (arbitrary binary outcome).
    """
    n = len(sample_ids)
    age = rng.integers(AGE_MIN, AGE_MAX + 1, size=n)
    sex = rng.integers(0, 2, size=n)
    phenotype = (rng.random(n) < PHENOTYPE_PREVALENCE).astype(int)
    df = pd.DataFrame({"sample_id": sample_ids, "age": age, "sex": sex, "phenotype": phenotype})
    return df.set_index("sample_id")


def generate_site_matrix(proteins: list, site_samples: list, metadata: pd.DataFrame,
                          baselines: dict, effects: dict, age_mean: float, age_sd: float,
                          rng: np.random.Generator) -> pd.DataFrame:
    min_b, max_b = min(baselines.values()), max(baselines.values())
    age_z = (metadata.loc[site_samples, "age"].to_numpy() - age_mean) / age_sd
    sex_vals = metadata.loc[site_samples, "sex"].to_numpy()
    pheno_vals = metadata.loc[site_samples, "phenotype"].to_numpy()
    n_samples = len(site_samples)

    data = {}
    for protein in proteins:
        baseline = baselines[protein]
        beta_age, beta_sex, beta_pheno = effects[protein]

        signal = beta_age * age_z + beta_sex * sex_vals + beta_pheno * pheno_vals
        biological = rng.normal(0, BIOLOGICAL_SD, size=n_samples)
        technical = rng.normal(0, TECHNICAL_SD, size=n_samples)
        log2_intensity = baseline + signal + biological + technical

        # MNAR: the less abundant a protein is, the more likely it is
        # to fall below the detection limit and show up as missing.
        relative_scarcity = (max_b - baseline) / (max_b - min_b + 1e-9)
        missing_prob = MISSING_RATE_AT_LOD * relative_scarcity
        missing_mask = rng.random(n_samples) < missing_prob

        values = np.round(log2_intensity, 3)
        values[missing_mask] = np.nan
        data[protein] = values

    # Built sample-major first (simpler to fill row by row), transposed below.
    df = pd.DataFrame(data, index=site_samples)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic proteomics matrices for chr22 genes from a BED file"
    )
    parser.add_argument(
        "--bed", type=str, default="refSeqGenes_chr22",
        help="Path to the BED file listing chr22 genes/isoforms (default: refSeqGenes_chr22 in the current directory)",
    )
    parser.add_argument(
        "--skip-gene-symbols", action="store_true",
        help="Skip the UniProt gene symbol lookup (useful if you have no internet access).",
    )
    args = parser.parse_args()

    bed_path = Path(args.bed)
    proteins = load_proteins_from_bed(bed_path)
    print(f"Found {len(proteins)} unique proteins (isoforms collapsed) in {bed_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    if args.skip_gene_symbols:
        gene_symbols = {p: p for p in proteins}
    else:
        print("Looking up gene symbols on UniProt (cached after first run)...")
        gene_symbols = fetch_gene_symbols(proteins, OUTPUT_DIR / "gene_symbol_cache.csv")

    # Build the full sample list across all sites up front, so metadata and
    # per-protein covariate effects are consistent across the whole cohort.
    site_sample_ids = {
        f"SITE{i}": [f"SITE{i}_PT{str(j).zfill(3)}" for j in range(1, N_SAMPLES_PER_SITE + 1)]
        for i in range(1, N_SITES + 1)
    }
    all_sample_ids = [sid for ids in site_sample_ids.values() for sid in ids]

    metadata = generate_metadata(all_sample_ids, rng)
    metadata_path = OUTPUT_DIR / "sample_metadata.csv"
    metadata.to_csv(metadata_path)
    age_mean, age_sd = metadata["age"].mean(), metadata["age"].std()

    # One baseline abundance per protein, drawn once and shared across all
    # sites (represents the protein's intrinsic plasma abundance level).
    baselines = {p: rng.uniform(BASELINE_MIN, BASELINE_MAX) for p in proteins}
    pd.Series(baselines, name="baseline_log2").rename_axis("protein_id").to_csv(
        OUTPUT_DIR / "protein_baselines.csv"
    )

    # Injected signal: per-protein effect sizes for age/sex/phenotype.
    effects = {
        p: (
            rng.normal(0, AGE_EFFECT_SD),
            rng.normal(0, SEX_EFFECT_SD),
            rng.normal(0, PHENOTYPE_EFFECT_SD),
        )
        for p in proteins
    }
    effects_df = pd.DataFrame(effects, index=["beta_age", "beta_sex", "beta_phenotype"]).T
    effects_df.index.name = "protein_id"
    effects_df.to_csv(OUTPUT_DIR / "protein_covariate_effects.csv")

    for i in range(1, N_SITES + 1):
        site_name = f"SITE{i}"
        site_samples = site_sample_ids[site_name]
        df_sample_major = generate_site_matrix(
            proteins, site_samples, metadata, baselines, effects, age_mean, age_sd, rng
        )
        df = df_sample_major.T  # rows = proteins, columns = samples, as requested
        df.index.name = "protein_id"
        df.insert(0, "gene_symbol", [gene_symbols[p] for p in df.index])
        out_path = OUTPUT_DIR / f"{site_name.lower()}_proteomics_log2.csv"
        df.to_csv(out_path)
        print(f"{site_name}: {df.shape[0]} proteins x {len(site_samples)} samples -> {out_path}")

    print(f"\nSample metadata (age/sex/phenotype) -> {metadata_path}")
    print(f"Injected covariate effects (ground truth) -> {OUTPUT_DIR / 'protein_covariate_effects.csv'}")
    print(f"Gene symbol mapping -> {OUTPUT_DIR / 'gene_symbol_cache.csv'}")
    print(f"Baselines -> {OUTPUT_DIR / 'protein_baselines.csv'}")
    print("\nReminder: sample_id values (e.g. SITE1_PT001) must match the")
    print("identifiers used on the genomics side so the two can be linked.")


if __name__ == "__main__":
    main()