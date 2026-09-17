"""
Generates synthetic proteomic intensity matrices, one per hospital site,
for all genes found in a chr22 BED file, with UniProt IDs converted to
gene symbols via the UniProt REST API.

No biological signal injected: plausible intensities (log-normal) with an
MNAR (missing not at random) pattern, realistic for quantitative proteomics.

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

BIOLOGICAL_SD = 0.8      # inter-individual variance
TECHNICAL_SD = 0.3       # technical noise (instrument, run)
MISSING_RATE_AT_LOD = 0.15  # missing probability for the least abundant protein
BASELINE_MIN = 6.0        # log2 intensity range covering plasma proteome dynamic range
BASELINE_MAX = 16.0

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
    and asking for the gene_names field directly - simpler and more
    reliable here than the async idmapping job workflow, which is meant
    for translating between database identifier systems rather than
    pulling an annotation field for IDs you already have).

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

    # Fallback for anything still unmapped (failed lookup, no internet,
    # or a genuinely unnamed/uncharacterized entry).
    for p in protein_ids:
        cached.setdefault(p, p)

    pd.Series(cached, name="gene_symbol").rename_axis("protein_id").to_csv(cache_path)
    return cached


def generate_site_matrix(proteins: list, n_samples: int, site_name: str,
                          baselines: dict, rng: np.random.Generator) -> pd.DataFrame:
    sample_ids = [f"{site_name}_PT{str(i).zfill(3)}" for i in range(1, n_samples + 1)]
    data = {}
    min_b, max_b = min(baselines.values()), max(baselines.values())

    for protein in proteins:
        baseline = baselines[protein]
        biological = rng.normal(0, BIOLOGICAL_SD, size=n_samples)
        technical = rng.normal(0, TECHNICAL_SD, size=n_samples)
        log2_intensity = baseline + biological + technical

        # MNAR: the less abundant a protein is, the more likely it is
        # to fall below the detection limit and show up as missing.
        relative_scarcity = (max_b - baseline) / (max_b - min_b + 1e-9)
        missing_prob = MISSING_RATE_AT_LOD * relative_scarcity
        missing_mask = rng.random(n_samples) < missing_prob

        values = np.round(log2_intensity, 3)
        values[missing_mask] = np.nan
        data[protein] = values

    # Built sample-major first (simpler to fill row by row), transposed below.
    df = pd.DataFrame(data, index=sample_ids)
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

    # One baseline abundance per protein, drawn once and shared across all
    # sites (represents the protein's intrinsic plasma abundance level,
    # independent of which hospital measured it).
    baselines = {p: rng.uniform(BASELINE_MIN, BASELINE_MAX) for p in proteins}
    pd.Series(baselines, name="baseline_log2").rename_axis("protein_id").to_csv(
        OUTPUT_DIR / "protein_baselines.csv"
    )

    for i in range(1, N_SITES + 1):
        site_name = f"SITE{i}"
        df_sample_major = generate_site_matrix(proteins, N_SAMPLES_PER_SITE, site_name, baselines, rng)
        df = df_sample_major.T  # rows = proteins, columns = samples, as requested
        df.index.name = "protein_id"
        df.insert(0, "gene_symbol", [gene_symbols[p] for p in df.index])
        out_path = OUTPUT_DIR / f"{site_name.lower()}_proteomics_log2.csv"
        df.to_csv(out_path)
        print(f"{site_name}: {df.shape[0]} proteins x {N_SAMPLES_PER_SITE} samples -> {out_path}")

    print(f"\nGene symbol mapping -> {OUTPUT_DIR / 'gene_symbol_cache.csv'}")
    print(f"Baselines -> {OUTPUT_DIR / 'protein_baselines.csv'}")
    print("\nReminder: sample_id values (e.g. SITE1_PT001) must match the")
    print("identifiers used on the genomics side so the two can be linked.")


if __name__ == "__main__":
    main()