#!/usr/bin/env python3
"""Build a protein-centered knowledge graph from haplograph annotations."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def read_edges(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    return pd.read_csv(opener(path, "rt"))


def expression_features(data_dir: Path) -> pd.DataFrame:
    metadata = pd.read_csv(data_dir / "sample_metadata.csv").set_index("sample_id")
    matrices = []
    for path in sorted(data_dir.glob("site*_proteomics_log2.csv")):
        matrix = pd.read_csv(path).set_index("protein_id")
        matrices.append(matrix.drop(columns=["gene_symbol"]).T)
    expression = pd.concat(matrices).loc[metadata.index]
    phenotype = metadata["phenotype"]
    result = pd.DataFrame(index=expression.columns)
    result["mean_log2_intensity"] = expression.mean()
    result["detection_rate"] = expression.notna().mean()
    result["phenotype_difference"] = [
        expression.loc[phenotype == 1, protein].mean()
        - expression.loc[phenotype == 0, protein].mean()
        for protein in expression.columns
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("proteomics/synthetic_proteomics_chr22"))
    parser.add_argument("--graph-features", type=Path, default=Path("federated_data/graph_protein_features.csv"))
    parser.add_argument("--edges", type=Path, default=None, help="Optional annotated haplograph edge CSV/CSV.GZ")
    parser.add_argument("--output-dir", type=Path, default=Path("proteomics/knowledge_graph"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph_features = pd.read_csv(args.graph_features).set_index("protein_id")
    symbols = pd.read_csv(args.data_dir / "gene_symbol_cache.csv").set_index("protein_id")["gene_symbol"]
    effects = pd.read_csv(args.data_dir / "protein_covariate_effects.csv").set_index("protein_id")
    graph_features = graph_features.join(symbols).join(effects).join(expression_features(args.data_dir))

    graph = nx.MultiDiGraph(name="chr22 haplograph-proteomics knowledge graph")
    for protein, row in graph_features.iterrows():
        graph.add_node(
            f"protein:{protein}",
            type="protein",
            protein_id=protein,
            gene_symbol=row.get("gene_symbol", protein),
            graph_degree=float(row["graph_degree"]),
            graph_weight=float(row["graph_weight"]),
            graph_lift=float(row["graph_lift"]),
            beta_phenotype=float(row.get("beta_phenotype", 0.0)),
            mean_log2_intensity=float(row["mean_log2_intensity"]),
            detection_rate=float(row["detection_rate"]),
            phenotype_difference=float(row["phenotype_difference"]),
        )
        gene = row.get("gene_symbol", protein)
        if pd.notna(gene) and str(gene) != protein:
            graph.add_node(f"gene:{gene}", type="gene", gene_symbol=gene)
            graph.add_edge(f"protein:{protein}", f"gene:{gene}", relation="encoded_by")
        for site in ("site1", "site2", "site3"):
            graph.add_node(f"site:{site}", type="site", site=site)
            graph.add_edge(f"site:{site}", f"protein:{protein}", relation="measures")

    if args.edges is not None and args.edges.exists():
        edges = read_edges(args.edges)
        required = {"source", "target"}
        missing = required - set(edges.columns)
        if missing:
            raise ValueError(f"Annotated edge file is missing columns: {sorted(missing)}")
        for _, row in edges.iterrows():
            source = str(row["source"])
            target = str(row["target"])
            graph.add_node(f"haploblock:{source}", type="haploblock", label=source)
            graph.add_node(f"haploblock:{target}", type="haploblock", label=target)
            attrs = {"relation": "co_occurs"}
            for column in ("weight", "lift"):
                if column in row:
                    attrs[column] = float(row[column])
            graph.add_edge(f"haploblock:{source}", f"haploblock:{target}", **attrs)
            for side in ("source", "target"):
                for protein in str(row.get(f"{side}_protein", "")).split(";"):
                    if protein and protein != "nan" and f"protein:{protein}" in graph:
                        graph.add_edge(f"haploblock:{row[side]}", f"protein:{protein}", relation="contains")

    node_rows = [{"node_id": node, **attrs} for node, attrs in graph.nodes(data=True)]
    edge_rows = [{"source": source, "target": target, **attrs} for source, target, attrs in graph.edges(data=True)]
    pd.DataFrame(node_rows).to_csv(args.output_dir / "nodes.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(args.output_dir / "edges.csv", index=False)
    summary = pd.DataFrame(
        [{"node_type": attrs.get("type"), "count": sum(a.get("type") == attrs.get("type") for _, a in graph.nodes(data=True))}
         for _, attrs in graph.nodes(data=True)]
    ).drop_duplicates()
    summary.to_csv(args.output_dir / "node_summary.csv", index=False)

    top_features = graph_features.nlargest(25, "graph_degree").sort_values("graph_degree")
    fig, ax = plt.subplots(figsize=(9, 7))
    labels = top_features["gene_symbol"].where(
        top_features["gene_symbol"].notna(), top_features.index.to_series()
    )
    ax.barh(labels, top_features["graph_degree"], color="#4c78a8")
    ax.set(xlabel="Haplograph degree", ylabel="Protein / gene", title="Most connected proteins in the haplograph")
    fig.tight_layout()
    fig.savefig(args.output_dir / "top_proteins_by_graph_degree.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        graph_features["graph_degree"],
        graph_features["phenotype_difference"],
        c=graph_features["beta_phenotype"].abs(),
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    ax.set(
        xscale="log",
        xlabel="Haplograph degree (log scale)",
        ylabel="Mean log2 intensity: phenotype 1 - phenotype 0",
        title="Haplograph connectivity and proteomic phenotype signal",
    )
    fig.colorbar(ax.collections[0], ax=ax, label="Absolute injected phenotype effect")
    fig.tight_layout()
    fig.savefig(args.output_dir / "graph_degree_vs_phenotype_signal.png", dpi=180)
    plt.close(fig)

    graph_features.sort_values("graph_degree", ascending=False).head(30).to_csv(
        args.output_dir / "top_proteins_by_graph_degree.csv"
    )
    print(f"Wrote {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
