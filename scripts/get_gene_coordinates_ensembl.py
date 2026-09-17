#!/usr/bin/env python3
"""Retrieve human gene coordinates, HGNC symbols, and UniProt IDs from Ensembl REST.

The script uses the Ensembl REST endpoint and writes one row per gene.
It queries Ensembl's human ``lookup`` endpoint in chunks, so it does not
require a local annotation file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

API = "https://rest.ensembl.org"

# GRCh38 primary chromosome lengths. Using fixed lengths avoids the occasional
# HTTP 500 returned by Ensembl's /info/assembly endpoint.
GRCH38_LENGTHS = {
    **{str(chromosome): length for chromosome, length in enumerate([
        248956422, 242193529, 198295559, 190214555, 181538259,
        170805979, 159345973, 145138636, 138394717, 133797422,
        135086622, 133275309, 114364328, 107043718, 101991189,
        90338345, 83257441, 80373285, 58617616, 64444167,
        46709983, 50818468,
    ], start=1)},
    "X": 156040895,
    "Y": 57227415,
}


def request_json(url: str, retries: int = 4) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "gene-coordinate-exporter/1.0",
        },
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt == retries - 1:
                detail = str(error)
                if isinstance(error, urllib.error.HTTPError):
                    try:
                        detail = f"HTTP {error.code}: {error.read().decode(errors='replace')[:500]}"
                    except Exception:
                        pass
                raise RuntimeError(f"Request failed after {retries} attempts: {url}\n{detail}") from error
            time.sleep(2**attempt)
    raise RuntimeError("Request failed")


def gene_rows(contig: str, start: int, end: int):
    url = (
        f"{API}/overlap/region/human/{contig}:{start}-{end}"
        "?feature=gene;db_type=core"
    )
    for gene in request_json(url):
        if gene.get("biotype") != "protein_coding":
            continue

        external_names = gene.get("external_names") or []
        symbol = gene.get("external_name", "")
        hgnc_id = ""
        for item in external_names:
            if not isinstance(item, dict):
                continue
            name = item.get("external_name", "")
            db = item.get("dbname", "")
            if db.lower() == "hgnc":
                hgnc_id = name
            elif not symbol and name:
                symbol = name

        # overlap/region usually includes external_name but not UniProt.
        details = request_json(
            f"{API}/lookup/id/{gene['id']}?expand=1"
        )
        uniprot_ids: set[str] = set()
        for dbxref in details.get("DBEntries", []):
            if str(dbxref.get("dbname", "")).lower() in {"uniprot", "uniprotkb"}:
                primary_id = dbxref.get("primary_id")
                if primary_id:
                    uniprot_ids.add(primary_id)

        yield {
            "chromosome": gene.get("seq_region_name", contig),
            "start": gene.get("start", ""),
            "end": gene.get("end", ""),
            "strand": gene.get("strand", ""),
            "ensembl_gene_id": gene.get("id", ""),
            "hgnc_id": hgnc_id,
            "hgnc_symbol": symbol,
            "uniprot_ids": ";".join(sorted(uniprot_ids)),
            "biotype": gene.get("biotype", ""),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        default=[str(i) for i in range(1, 23)] + ["X", "Y"],
        help="Chromosomes/contigs to query (default: 1-22 X Y)",
    )
    parser.add_argument("--assembly", default="GRCh38", help="Displayed assembly label")
    parser.add_argument("--chunk-size", type=int, default=5_000_000)
    parser.add_argument("-o", "--output", default="gene_coordinates_ensembl.tsv")
    args = parser.parse_args()

    # Ensembl's REST endpoint uses the current human reference assembly.
    if args.assembly != "GRCh38":
        print("Warning: Ensembl REST currently returns the current human assembly; "
              "the assembly label is not used to select an older release.", file=sys.stderr)

    fields = [
        "assembly", "chromosome", "start", "end", "strand",
        "ensembl_gene_id", "hgnc_id", "hgnc_symbol", "uniprot_ids", "biotype",
    ]
    output = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    try:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for chromosome in args.chromosomes:
            if chromosome not in GRCH38_LENGTHS:
                raise ValueError(
                    f"Unsupported chromosome '{chromosome}'. Use 1-22, X, or Y."
                )
            length = GRCH38_LENGTHS[chromosome]
            for start in range(1, length + 1, args.chunk_size):
                end = min(start + args.chunk_size - 1, length)
                print(f"Querying chr{chromosome}:{start}-{end}", file=sys.stderr)
                for row in gene_rows(chromosome, start, end):
                    row["assembly"] = args.assembly
                    writer.writerow(row)
                time.sleep(0.1)
    finally:
        if output is not sys.stdout:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
