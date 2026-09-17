#!/usr/bin/env python3
"""Export genes, HGNC symbols, UniProt IDs, and genomic coordinates.

Input: a GTF/GFF3 annotation file (optionally gzip-compressed).
Output: TSV with chromosome, start, end, strand, HGNC symbol, and UniProt IDs.

The script supports common Ensembl/GENCODE attributes such as:
  gene_name, hgnc_symbol, hgnc_id, gene_id, uniprot_id, uniprot_ids,
  protein_id, and Dbxref=db:accession.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path
from typing import Iterable, TextIO


KEY_VALUE_RE = re.compile(r"([A-Za-z][A-Za-z0-9_.-]*)\s*(?:=|\s)\s*\"?([^;\"]+)\"?")
UNIPROT_RE = re.compile(r"(?:UniProtKB:|UniProt:|uniprot[:_])([A-Za-z0-9_-]+)", re.IGNORECASE)


def open_text(path: Path) -> TextIO:
    if str(path) == "-":
        return sys.stdin
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def parse_attributes(raw: str) -> dict[str, list[str]]:
    """Parse both GTF key-value attributes and GFF3 key=value attributes."""
    attrs: dict[str, list[str]] = {}
    for field in raw.strip().strip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        match = KEY_VALUE_RE.fullmatch(field)
        if not match:
            continue
        key, value = match.groups()
        attrs.setdefault(key.lower(), []).append(value.strip().strip('"'))
    return attrs


def first(attrs: dict[str, list[str]], *keys: str) -> str:
    for key in keys:
        values = attrs.get(key.lower(), [])
        if values and values[0]:
            return values[0]
    return ""


def all_values(attrs: dict[str, list[str]], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        values.extend(attrs.get(key.lower(), []))
    return values


def split_ids(values: Iterable[str]) -> list[str]:
    found: list[str] = []
    for value in values:
        for match in UNIPROT_RE.finditer(value):
            found.append(match.group(1))
        # Also support plain UniProt accessions in dedicated fields.
        for token in re.split(r"[,| ]+", value):
            token = token.strip()
            if re.fullmatch(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5})", token):
                found.append(token)
    return sorted(set(found))


def rows(path: Path):
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                print(f"Warning: skipping malformed line {line_number}", file=sys.stderr)
                continue
            chrom, source, feature, start, end, score, strand, phase, raw_attrs = columns
            if feature.lower() != "gene":
                continue
            attrs = parse_attributes(raw_attrs)

            symbol = first(attrs, "hgnc_symbol", "gene_name", "gene_symbol", "symbol")
            hgnc_id = first(attrs, "hgnc_id")
            if not symbol and hgnc_id:
                symbol = hgnc_id

            uniprot_values = all_values(
                attrs,
                "uniprot_id",
                "uniprot_ids",
                "uniprot",
                "protein_id",
                "dbxref",
                "db_xref",
            )
            uniprot_ids = split_ids(uniprot_values)

            yield {
                "chromosome": chrom,
                "start": start,
                "end": end,
                "strand": strand,
                "gene_id": first(attrs, "gene_id", "geneid"),
                "hgnc_id": hgnc_id,
                "hgnc_symbol": symbol,
                "uniprot_ids": ";".join(uniprot_ids),
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path, help="Input GTF/GFF3, or - for stdin")
    parser.add_argument("-o", "--output", type=Path, default=Path("gene_coordinates.tsv"))
    args = parser.parse_args()

    fieldnames = [
        "chromosome",
        "start",
        "end",
        "strand",
        "gene_id",
        "hgnc_id",
        "hgnc_symbol",
        "uniprot_ids",
    ]
    output = sys.stdout if str(args.output) == "-" else args.output.open("w", newline="")
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows(args.annotation))
    finally:
        if output is not sys.stdout:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
