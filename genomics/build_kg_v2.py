#!/usr/bin/env python3
"""Schema v2: add genes, proteins and per-individual proteomics to the genome knowledge graph.

    python build_kg_v2.py --chrom chr22                                  # uses outputs/proteomics_synth/<chrom>
    python build_kg_v2.py --chrom chr22 --measured my_measured_long.csv --metadata my_samples.csv

Requires build_kg.py to have run.  Writes genes.csv, proteins.csv, block_gene.csv, gene_protein.csv,
measured.csv, abundance_z.npz, abundance_observed.npz, individuals_v2.csv, label_maps_v2.json and
hetero_v2.pt into outputs/kg/<chrom>/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import haplokg
import haplokg_proteins as hp


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--bed", type=Path, default=here.parent / "proteomics" / "uniprot_chr22.bed")
    parser.add_argument("--symbols", type=Path, default=here.parent / "proteomics" / "synthetic_proteomics_chr22" / "gene_symbol_cache.csv")
    parser.add_argument("--measured", type=Path, default=None, help="long csv: individual_id, protein_id, log2_intensity, site")
    parser.add_argument("--metadata", type=Path, default=None, help="csv: sample_id, site, age, sex, phenotype")
    args = parser.parse_args()
    kg_dir = here / "outputs" / "kg" / args.chrom
    synth = here / "outputs" / "proteomics_synth" / args.chrom
    measured = args.measured or synth / "measured_long.csv"
    metadata = args.metadata or synth / "sample_metadata.csv"

    kg = haplokg.load_kg(kg_dir)
    t = hp.build_protein_tables(kg, args.bed, measured, metadata, args.symbols)
    summary = hp.save_protein_tables(t, kg_dir)

    import torch
    data = torch.load(kg_dir / "hetero.pt", weights_only=False)
    data = hp.extend_hetero_data(data, t)
    torch.save(data, kg_dir / "hetero_v2.pt")
    print(data)
    print(json.dumps(summary, indent=2))
    print(f"wrote {kg_dir}/hetero_v2.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
