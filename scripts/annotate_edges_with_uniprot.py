#!/usr/bin/env python3
"""Annotate haplograph edges with UniProt IDs by genomic interval overlap.

Node names are expected to contain coordinates such as:
    chr22_17420471-17463955_cluster123

The BED file is expected to contain at least:
    chromosome, start, end, uniprot_id

BED coordinates are interpreted as 0-based, half-open. Haplograph node
coordinates are interpreted as 1-based, inclusive; overlap is therefore
computed using inclusive genomic intervals after converting BED starts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import shutil
import tempfile
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

NODE_COORDINATES = re.compile(
    r"^(?P<chromosome>[^_]+)_(?P<start>\d+)-(?P<end>\d+)_(?P<cluster>.+)$"
)


@dataclass(frozen=True)
class Interval:
    chromosome: str
    start: int
    end: int
    uniprot_id: str


class IntervalIndex:
    def __init__(self, intervals: list[Interval]):
        self.by_chromosome: dict[str, list[Interval]] = {}
        for interval in intervals:
            self.by_chromosome.setdefault(interval.chromosome, []).append(interval)
        self.starts: dict[str, list[int]] = {}
        for chromosome, values in self.by_chromosome.items():
            values.sort(key=lambda value: (value.start, value.end, value.uniprot_id))
            self.starts[chromosome] = [value.start for value in values]

    def overlap_ids(self, chromosome: str, start: int, end: int) -> list[str]:
        values = self.by_chromosome.get(chromosome, [])
        starts = self.starts.get(chromosome, [])
        # Include intervals whose start is before or at the query end. Scan
        # backward enough to catch intervals that began before the insertion point.
        index = bisect_left(starts, end + 1)
        matches = {
            value.uniprot_id
            for value in values[:index]
            if value.end >= start and value.start <= end
        }
        return sorted(matches)


def read_bed(path: Path) -> IntervalIndex:
    intervals: list[Interval] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"BED line {line_number} has fewer than 4 columns")
            chromosome, bed_start, bed_end, uniprot_id = fields[:4]
            intervals.append(
                Interval(
                    chromosome=chromosome,
                    start=int(bed_start) + 1,
                    end=int(bed_end),
                    uniprot_id=uniprot_id.split("-", 1)[0],
                )
            )
    return IntervalIndex(intervals)


def parse_node(node: str) -> dict[str, str]:
    match = NODE_COORDINATES.match(node)
    if not match:
        return {"chr": "", "start": "", "end": "", "cluster": ""}
    return {
        "chr": match.group("chromosome"),
        "start": match.group("start"),
        "end": match.group("end"),
        "cluster": match.group("cluster"),
    }


def annotate_node(node: str, index: IntervalIndex) -> tuple[dict[str, str], str]:
    parsed = parse_node(node)
    if not parsed["chr"]:
        return parsed, ""
    proteins = index.overlap_ids(
        parsed["chr"], int(parsed["start"]), int(parsed["end"])
    )
    return parsed, ";".join(proteins)


def annotate(
    input_path: Path,
    output_path: Path,
    index: IntervalIndex,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(input_path, "rt", newline="") as source, gzip.open(
        output_path, "wt", newline=""
    ) as destination:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"source", "target"}.issubset(reader.fieldnames):
            raise ValueError("Input CSV must contain source and target columns")

        fieldnames = list(reader.fieldnames)
        for field in (
            "source_chr", "source_start", "source_end", "source_cluster",
            "target_chr", "target_start", "target_end", "target_cluster",
            "source_protein", "target_protein",
        ):
            if field not in fieldnames:
                fieldnames.append(field)
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            source, source_proteins = annotate_node(row["source"], index)
            target, target_proteins = annotate_node(row["target"], index)
            for field in ("chr", "start", "end", "cluster"):
                row[f"source_{field}"] = source[field]
                row[f"target_{field}"] = target[field]
            row["source_protein"] = source_proteins
            row["target_protein"] = target_proteins
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("haplograph/edges_lift_above_threshold.csv.gz"),
    )
    parser.add_argument(
        "--bed",
        type=Path,
        default=Path("proteomics/uniprot_chr22.bed"),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("haplograph/edges_lift_above_threshold_uniprot.csv.gz"),
    )
    args = parser.parse_args()

    if args.output.resolve() == args.edges.resolve():
        with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            annotate(args.edges, temporary_path, read_bed(args.bed))
            shutil.move(temporary_path, args.output)
        finally:
            temporary_path.unlink(missing_ok=True)
    else:
        annotate(args.edges, args.output, read_bed(args.bed))

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
