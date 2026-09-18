#!/usr/bin/env python3
"""Look at the graph *before* embedding it.

Builds the cluster co-occurrence graph (nodes = haploblock clusters, edges =
lift >= 5 co-occurrence) with NetworkX, prints its statistics, exports GraphML
for Gephi / Cytoscape, and draws:

  edge_positions.png        every edge as (position of source, position of target)
                            coloured by lift -> long-range structure along the chromosome
  region_<start>-<end>.png  the subgraph of one region, nodes coloured by the
                            ancestry they are enriched in, sized by carrier support
  degree_distribution.png

With the nx-cugraph backend installed (GPU image) NetworkX dispatches the heavy
algorithms to cuGraph automatically when NX_CUGRAPH_AUTOCONFIG=True.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import haplokg

ANCESTRY_COLOURS = {"AFR": "#d55e00", "AMR": "#cc79a7", "EAS": "#009e73", "EUR": "#0072b2", "SAS": "#e69f00"}


def backend_note() -> str:
    try:
        import nx_cugraph  # noqa: F401
        auto = os.environ.get("NX_CUGRAPH_AUTOCONFIG", "")
        return f"nx-cugraph installed (NX_CUGRAPH_AUTOCONFIG={auto or 'unset'})"
    except ImportError:
        return "nx-cugraph not installed -> pure NetworkX on CPU"


def cluster_graph(kg: dict, assoc: pd.DataFrame | None) -> nx.Graph:
    cl = kg["clusters"].copy()
    coords = cl["block_id"].map(lambda b: haplokg.parse_block_id(b))
    cl["start"] = [c[1] for c in coords]
    cl["end"] = [c[2] for c in coords]
    if assoc is not None:
        cl = cl.merge(assoc[["cluster_idx", "ancestry_cramers_v", "ancestry_dominant"]], on="cluster_idx", how="left")
    G = nx.Graph(name="haploblock cluster co-occurrence")
    for row in cl.itertuples(index=False):
        G.add_node(int(row.cluster_idx), cluster_id=row.cluster_id, block_id=row.block_id, block_idx=int(row.block_idx),
                   start=int(row.start), end=int(row.end), support=int(row.support),
                   ancestry_dominant=getattr(row, "ancestry_dominant", "") or "",
                   ancestry_v=float(getattr(row, "ancestry_cramers_v", float("nan"))))
    co = kg["co_occurs"]
    G.add_edges_from(zip(co["src"].astype(int), co["dst"].astype(int),
                         ({"weight": float(w), "lift": float(l)} for w, l in zip(co["weight"], co["lift"]))))
    return G


def block_graph(G: nx.Graph) -> nx.Graph:
    """Collapse clusters into their haploblocks; edge = total lift between two blocks."""
    B = nx.Graph(name="haploblock co-occurrence (collapsed)")
    for _, d in G.nodes(data=True):
        if d["block_idx"] not in B:
            B.add_node(d["block_idx"], block_id=d["block_id"], start=d["start"], end=d["end"], n_clusters=0)
        B.nodes[d["block_idx"]]["n_clusters"] += 1
    for u, v, d in G.edges(data=True):
        bu, bv = G.nodes[u]["block_idx"], G.nodes[v]["block_idx"]
        if bu == bv:
            continue
        if B.has_edge(bu, bv):
            B[bu][bv]["total_lift"] += d["lift"]; B[bu][bv]["n_edges"] += 1
        else:
            B.add_edge(bu, bv, total_lift=d["lift"], n_edges=1)
    return B


def statistics(G: nx.Graph, B: nx.Graph) -> dict:
    degrees = np.array([d for _, d in G.degree()])
    lift_degree = dict(G.degree(weight="lift"))
    components = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    hubs = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:10]
    return {
        "backend": backend_note(),
        "clusters": G.number_of_nodes(), "co_occurrence_edges": G.number_of_edges(),
        "density": float(nx.density(G)),
        "degree_mean": float(degrees.mean()), "degree_median": float(np.median(degrees)), "degree_max": int(degrees.max()),
        "isolated_clusters": int((degrees == 0).sum()),
        "connected_components": len(components), "largest_component": components[0] if components else 0,
        "average_clustering_coefficient": float(nx.average_clustering(G)),
        "blocks_in_graph": B.number_of_nodes(), "block_block_edges": B.number_of_edges(),
        "same_block_edges": int(sum(1 for u, v in G.edges if G.nodes[u]["block_idx"] == G.nodes[v]["block_idx"])),
        "top_hubs": [{"cluster_id": G.nodes[n]["cluster_id"], "degree": G.degree(n), "lift_sum": round(lift_degree[n], 1),
                      "support": G.nodes[n]["support"], "ancestry_dominant": G.nodes[n]["ancestry_dominant"]} for n in hubs],
    }


def default_region(data_dir: Path, chrom: str) -> tuple[int, int]:
    """The densest published 'island' (extended haplotype) is a good first thing to look at."""
    islands = data_dir / "haplograph" / chrom / "islands.csv.gz"
    if islands.exists():
        with gzip.open(islands, "rt") as fh:
            top = pd.read_csv(fh).iloc[0]
        return int(top["span_start"]), int(top["span_end"])
    return 45_000_000, 45_500_000


def plot_edge_positions(G: nx.Graph, chrom: str, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys, lifts = [], [], []
    for u, v, d in G.edges(data=True):
        a, b = sorted((G.nodes[u]["start"], G.nodes[v]["start"]))
        xs.append(a / 1e6); ys.append(b / 1e6); lifts.append(d["lift"])
    order = np.argsort(lifts)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    sc = ax.scatter(np.array(xs)[order], np.array(ys)[order], c=np.array(lifts)[order], s=2, cmap="viridis", alpha=0.6,
                    norm=matplotlib.colors.LogNorm())
    ax.set_xlabel(f"{chrom} position of cluster A (Mb)"); ax.set_ylabel(f"{chrom} position of cluster B (Mb)")
    ax.set_title(f"{G.number_of_edges():,} co-occurrence edges (lift >= 5) between {G.number_of_nodes():,} clusters")
    fig.colorbar(sc, ax=ax, label="lift (log scale)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_region(G: nx.Graph, chrom: str, start: int, end: int, out: Path, seed: int = 42) -> dict:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    nodes = [n for n, d in G.nodes(data=True) if d["end"] >= start and d["start"] <= end]
    H = G.subgraph(nodes).copy()
    H.remove_nodes_from([n for n in H if H.degree(n) == 0])
    if H.number_of_nodes() == 0:
        return {"region": f"{chrom}:{start}-{end}", "nodes": 0, "edges": 0}
    pos = nx.spring_layout(H, weight="lift", seed=seed, k=1.5 / np.sqrt(H.number_of_nodes()))
    colours = [ANCESTRY_COLOURS.get(H.nodes[n]["ancestry_dominant"], "#888888") for n in H]
    sizes = [20 + 180 * H.nodes[n]["support"] / 2548 for n in H]
    widths = [0.2 + 1.5 * np.log10(d["lift"] / 5 + 1) for _, _, d in H.edges(data=True)]
    fig, ax = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_edges(H, pos, ax=ax, width=widths, edge_color="#b0b0b0", alpha=0.5)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=colours, node_size=sizes, linewidths=0.3, edgecolors="white")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=a) for a, c in ANCESTRY_COLOURS.items()]
    ax.legend(handles=handles, title="enriched in", frameon=False, loc="upper left")
    blocks = sorted({H.nodes[n]["block_id"] for n in H})
    ax.set_title(f"{chrom}:{start:,}-{end:,}  |  {H.number_of_nodes()} clusters in {len(blocks)} haploblocks, "
                 f"{H.number_of_edges()} co-occurrence edges\nnode size = carrier support, edge width = lift")
    ax.axis("off"); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return {"region": f"{chrom}:{start}-{end}", "nodes": H.number_of_nodes(), "edges": H.number_of_edges(), "blocks": len(blocks)}


def plot_degree_distribution(G: nx.Graph, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    degrees = np.array([d for _, d in G.degree()])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(degrees, bins=np.logspace(0, np.log10(degrees.max() + 1), 40), color="#1f6f8b")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("degree (co-occurring clusters)"); ax.set_ylabel("clusters"); ax.set_title("Degree distribution")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--kg-dir", type=Path, default=None)
    parser.add_argument("--assoc", type=Path, default=None, help="cluster_phenotype_association.csv from cooccurrence_analysis.py")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--region", default=None, help="start-end in bp, e.g. 45035149-45534032 (default: densest island)")
    parser.add_argument("--data-dir", type=Path, default=here / "data")
    parser.add_argument("--no-graphml", action="store_true")
    args = parser.parse_args()
    kg_dir = args.kg_dir or here / "outputs" / "kg" / args.chrom
    out_dir = args.out_dir or here / "outputs" / "graph" / args.chrom
    assoc_path = args.assoc or here / "outputs" / "cooccurrence" / args.chrom / "cluster_phenotype_association.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    kg = haplokg.load_kg(kg_dir)
    assoc = pd.read_csv(assoc_path) if assoc_path.exists() else None
    G = cluster_graph(kg, assoc)
    B = block_graph(G)
    stats = statistics(G, B)

    if not args.no_graphml:
        nx.write_graphml(G, out_dir / f"cluster_cooccurrence_{args.chrom}.graphml")
        nx.write_graphml(B, out_dir / f"block_cooccurrence_{args.chrom}.graphml")

    plot_edge_positions(G, args.chrom, out_dir / "edge_positions.png")
    plot_degree_distribution(G, out_dir / "degree_distribution.png")
    if args.region:
        start, end = (int(x) for x in args.region.split("-"))
    else:
        start, end = default_region(args.data_dir, args.chrom)
    stats["region_plot"] = plot_region(G, args.chrom, start, end, out_dir / f"region_{start}-{end}.png")

    (out_dir / "graph_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
