#!/usr/bin/env python3
"""Prepare site phenotype tables and model-ready federated site datasets."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def edge_features(path: Path) -> pd.DataFrame:
    """Build per-protein graph features from annotated haplograph edges."""
    with gzip.open(path, "rt") as handle:
        edges = pd.read_csv(handle)
    required = {"source_protein", "target_protein", "weight", "lift"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Edge file is missing columns: {sorted(missing)}")

    stats: dict[str, dict[str, float]] = {}
    for _, row in edges.iterrows():
        proteins = set()
        for column in ("source_protein", "target_protein"):
            value = str(row[column])
            if value and value != "nan":
                proteins.update(value.split(";"))
        for protein in proteins:
            item = stats.setdefault(protein, {"graph_degree": 0, "graph_weight": 0.0, "graph_lift": 0.0})
            item["graph_degree"] += 1
            item["graph_weight"] += float(row["weight"])
            item["graph_lift"] += float(row["lift"])
    result = pd.DataFrame.from_dict(stats, orient="index")
    result.index.name = "protein_id"
    return result.reset_index()


def site_samples(proteomics_path: Path, metadata: pd.DataFrame, site: str) -> pd.DataFrame:
    matrix = pd.read_csv(proteomics_path)
    sample_ids = [column for column in matrix.columns if column.startswith(f"{site}_")]
    if not sample_ids:
        raise ValueError(f"No samples for {site} in {proteomics_path}")
    values = matrix.set_index("protein_id")[sample_ids].T
    values.index.name = "sample_id"
    values = values.reset_index()
    values = values.merge(metadata, on="sample_id", validate="one_to_one")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("proteomics/synthetic_proteomics_chr22"))
    parser.add_argument("--edges", type=Path, default=Path("haplograph/edges_lift_above_threshold_uniprot.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("federated_data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.data_dir / "sample_metadata.csv")
    for site in ("SITE1", "SITE2", "SITE3"):
        site_metadata = metadata[metadata["sample_id"].str.startswith(f"{site}_")].copy()
        site_metadata.to_csv(args.output_dir / f"{site.lower()}_pheno.tsv", sep="\t", index=False)

        proteomics = args.data_dir / f"{site.lower()}_proteomics_log2.csv"
        samples = site_samples(proteomics, metadata, site)
        samples.to_csv(args.output_dir / f"{site.lower()}_samples.csv", index=False)

    graph = edge_features(args.edges)
    graph.to_csv(args.output_dir / "graph_protein_features.csv", index=False)
    print(f"Prepared federated data in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
