#!/usr/bin/env python3
"""Load the haploblock knowledge graph into Neo4j so it can be browsed at localhost:7474.

Nodes        (:Individual {id, ancestry, population, sex, split})
             (:Cluster    {id, idx, block_id, support, support_frac, ancestry_dominant, ancestry_v})
             (:Block      {id, idx, chrom, start, end, length, n_clusters, entropy, dominance})
Relations    (Individual)-[:CARRIES]->(Cluster)
             (Cluster)-[:IN_BLOCK]->(Block)
             (Block)-[:NEXT_BLOCK]->(Block)
             (Cluster)-[:CO_OCCURS {weight, lift}]->(Cluster)

    docker compose up -d neo4j
    python neo4j_load.py --chrom chr22                 # everything (2.4M CARRIES edges, a few minutes)
    python neo4j_load.py --chrom chr22 --region 45035149-45534032   # just one region, seconds

Example Cypher once loaded:
    MATCH (i:Individual {id:'HG00096'})-[:CARRIES]->(c:Cluster)-[:IN_BLOCK]->(b:Block) RETURN i,c,b LIMIT 50
    MATCH (c:Cluster)-[r:CO_OCCURS]->(d:Cluster) WHERE r.lift > 40 RETURN c,r,d
    MATCH (c:Cluster) WHERE c.ancestry_v > 0.7 RETURN c.id, c.ancestry_dominant, c.support ORDER BY c.ancestry_v DESC
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

import config
import haplokg


def run_batches(session, query: str, rows: list[dict], batch: int, label: str) -> None:
    t0 = time.time()
    for i in range(0, len(rows), batch):
        session.run(query, rows=rows[i:i + batch]).consume()
    print(f"  {label:32s} {len(rows):>9,} rows  {time.time() - t0:6.1f}s")


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--kg-dir", type=Path, default=None)
    parser.add_argument("--uri", default=config.settings.neo4j_uri)            # NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD from .env
    parser.add_argument("--user", default=config.settings.neo4j_user)
    parser.add_argument("--password", default=config.settings.neo4j_password)
    parser.add_argument("--region", default=None, help="start-end (bp): only clusters in these blocks and their edges")
    parser.add_argument("--skip-carries", action="store_true", help="skip the 2.4M Individual->Cluster edges")
    parser.add_argument("--batch", type=int, default=10000)
    parser.add_argument("--wipe", action="store_true", help="delete everything in the database first")
    args = parser.parse_args()
    kg_dir = args.kg_dir or here / "outputs" / "kg" / args.chrom

    kg = haplokg.load_kg(kg_dir)
    ind, cl, bl, co, nb = kg["individuals"], kg["clusters"], kg["blocks"], kg["co_occurs"], kg["next_block"]
    assoc_path = here / "outputs" / "cooccurrence" / args.chrom / "cluster_phenotype_association.csv"
    if assoc_path.exists():
        assoc = pd.read_csv(assoc_path)[["cluster_idx", "ancestry_dominant", "ancestry_cramers_v"]]
        cl = cl.merge(assoc, on="cluster_idx", how="left")
    split_path = here / "outputs" / "splits" / args.chrom / "split_seed42.csv"
    ind["split"] = pd.read_csv(split_path)["split"].to_numpy() if split_path.exists() else "n/a"

    if args.region:
        start, end = (int(x) for x in args.region.split("-"))
        bl = bl[(bl["end"] >= start) & (bl["start"] <= end)]
        cl = cl[cl["block_idx"].isin(bl["block_idx"])]
        keep = set(cl["cluster_idx"])
        co = co[co["src"].isin(keep) & co["dst"].isin(keep)]
        nb = nb[nb["src"].isin(bl["block_idx"]) & nb["dst"].isin(bl["block_idx"])]
    carries = kg["carries"].tocsc()[:, cl["cluster_idx"].to_numpy()].tocoo()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    with driver.session() as s:
        if args.wipe:
            while s.run("MATCH (n) WITH n LIMIT 50000 DETACH DELETE n RETURN count(n) AS c").single()["c"]:
                pass
        for q in (
            "CREATE CONSTRAINT individual_id IF NOT EXISTS FOR (i:Individual) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT cluster_idx IF NOT EXISTS FOR (c:Cluster) REQUIRE c.idx IS UNIQUE",
            "CREATE CONSTRAINT block_idx IF NOT EXISTS FOR (b:Block) REQUIRE b.idx IS UNIQUE",
            "CREATE INDEX cluster_id IF NOT EXISTS FOR (c:Cluster) ON (c.id)",
            "CREATE INDEX individual_ancestry IF NOT EXISTS FOR (i:Individual) ON (i.ancestry)",
        ):
            s.run(q).consume()

        def clean(frame: pd.DataFrame) -> list[dict]:
            return [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                    for r in frame.to_dict("records")]

        print(f"loading {args.chrom} into {args.uri}")
        run_batches(s, """UNWIND $rows AS r MERGE (b:Block {idx: r.idx})
            SET b.id = r.id, b.chrom = r.chrom, b.start = r.start, b.end = r.end, b.length = r.length,
                b.n_clusters = r.n_clusters, b.entropy = r.entropy, b.dominance = r.dominance""",
            clean(bl.rename(columns={"block_idx": "idx", "block_id": "id", "chr": "chrom", "block_length": "length",
                                     "shannon_entropy": "entropy"})
                  [["idx", "id", "chrom", "start", "end", "length", "n_clusters", "entropy", "dominance"]]
                  .astype({"idx": int, "start": int, "end": int, "length": int})), args.batch, "Block nodes")

        cl_rows = cl.rename(columns={"cluster_idx": "idx", "cluster_id": "id"})
        cl_rows["ancestry_dominant"] = cl_rows.get("ancestry_dominant", pd.Series([None] * len(cl_rows)))
        cl_rows["ancestry_v"] = cl_rows.get("ancestry_cramers_v", pd.Series([np.nan] * len(cl_rows)))
        run_batches(s, """UNWIND $rows AS r MERGE (c:Cluster {idx: r.idx})
            SET c.id = r.id, c.block_id = r.block_id, c.support = r.support, c.support_frac = r.support_frac,
                c.ancestry_dominant = r.ancestry_dominant, c.ancestry_v = r.ancestry_v
            WITH c, r MATCH (b:Block {idx: r.block_idx}) MERGE (c)-[:IN_BLOCK]->(b)""",
            clean(cl_rows[["idx", "id", "block_id", "block_idx", "support", "support_frac", "ancestry_dominant", "ancestry_v"]]
                  .astype({"idx": int, "block_idx": int, "support": int})), args.batch, "Cluster nodes + IN_BLOCK")

        run_batches(s, """UNWIND $rows AS r MATCH (a:Block {idx: r.src}), (b:Block {idx: r.dst}) MERGE (a)-[:NEXT_BLOCK]->(b)""",
            clean(nb.astype(int)), args.batch, "NEXT_BLOCK")

        run_batches(s, """UNWIND $rows AS r MATCH (a:Cluster {idx: r.src}), (b:Cluster {idx: r.dst})
            MERGE (a)-[e:CO_OCCURS]->(b) SET e.weight = r.weight, e.lift = r.lift""",
            clean(co.astype({"src": int, "dst": int, "weight": float, "lift": float})), args.batch, "CO_OCCURS")

        run_batches(s, """UNWIND $rows AS r MERGE (i:Individual {id: r.id})
            SET i.ancestry = r.ancestry, i.population = r.population, i.sex = r.sex, i.split = r.split""",
            clean(ind.rename(columns={"individual_id": "id"})[["id", "ancestry", "population", "sex", "split"]]),
            args.batch, "Individual nodes")

        if not args.skip_carries:
            ids = ind["individual_id"].to_numpy()
            cidx = cl["cluster_idx"].to_numpy()
            rows = [{"i": ids[r], "c": int(cidx[c])} for r, c in zip(carries.row, carries.col)]
            run_batches(s, """UNWIND $rows AS r MATCH (i:Individual {id: r.i}), (c:Cluster {idx: r.c}) MERGE (i)-[:CARRIES]->(c)""",
                        rows, args.batch, "CARRIES")

        counts = s.run("""MATCH (n) WITH labels(n)[0] AS l, count(*) AS c RETURN l, c ORDER BY l""").data()
        rels = s.run("""MATCH ()-[r]->() WITH type(r) AS t, count(*) AS c RETURN t, c ORDER BY t""").data()
    driver.close()
    print("nodes:", {r["l"]: r["c"] for r in counts})
    print("relationships:", {r["t"]: r["c"] for r in rels})
    print("browse: http://localhost:7474  (neo4j / progenome)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
