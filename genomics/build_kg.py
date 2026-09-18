#!/usr/bin/env python3
"""Build the haploblock knowledge graph for one chromosome.

    python build_kg.py --chrom chr22 --min-support 25

Reads data/ (see fetch_data.sh) and writes outputs/kg/<chrom>/:
  individuals.csv  clusters.csv  blocks.csv  co_occurs.csv  next_block.csv
  carries.npz      sparse 0/1 matrix, individuals x kept clusters
  label_maps.json  class order for ancestry / population / sex codes
  hetero.pt        torch_geometric HeteroData (load with torch.load(..., weights_only=False))
  summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import haplokg


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--data-dir", type=Path, default=here / "data")
    parser.add_argument("--out-dir", type=Path, default=None, help="default outputs/kg/<chrom>")
    parser.add_argument("--min-support", type=int, default=25,
                        help="keep clusters carried by >= this many individuals (default 25, same as the edge file)")
    parser.add_argument("--no-symmetric", action="store_true",
                        help="also keep near-universal clusters (default drops clusters with < min-support non-carriers)")
    parser.add_argument("--chunksize", type=int, default=8192)
    parser.add_argument("--no-torch", action="store_true", help="skip writing hetero.pt")
    args = parser.parse_args()
    out_dir = args.out_dir or here / "outputs" / "kg" / args.chrom

    graph_dir = args.data_dir / "haplograph" / args.chrom
    t0 = time.time()
    print(f"[1/4] streaming {graph_dir / 'nodes.csv.gz'} ...", flush=True)
    cluster_ids, block_ids, individual_ids, matrix = haplokg.read_node_matrix(graph_dir / "nodes.csv.gz", args.chunksize)
    print(f"      {matrix.shape[0]:,} clusters x {matrix.shape[1]:,} individuals, nnz={matrix.nnz:,}  ({time.time()-t0:.0f}s)")

    print("[2/4] loading phenotypes, block stats, edges", flush=True)
    phenotypes = haplokg.load_phenotypes(args.data_dir / "haplograph" / "phenotypes_real.csv")
    block_stats = haplokg.load_block_stats(args.data_dir / "haploblocks" / "block_stats.tsv", args.chrom)
    edges = haplokg.load_edges(graph_dir / "edges_lift_above_threshold.csv.gz")

    print(f"[3/4] building tables (min_support={args.min_support}, symmetric={not args.no_symmetric})", flush=True)
    tables = haplokg.build_tables(cluster_ids, block_ids, individual_ids, matrix, phenotypes, block_stats, edges,
                                  min_support=args.min_support, symmetric=not args.no_symmetric)
    summary = haplokg.save_tables(tables, out_dir)

    if not args.no_torch:
        import torch
        print("[4/4] writing HeteroData", flush=True)
        data = haplokg.to_hetero_data(tables)
        torch.save(data, out_dir / "hetero.pt")
        print(data)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}  ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
