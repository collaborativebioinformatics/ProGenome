#!/usr/bin/env python3
"""Plot interpretable haplographs and compare them with proteomic models."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def read_edges(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    return pd.read_csv(opener(path, "rt"))


def draw_haplograph(edges: pd.DataFrame, output: Path, max_nodes: int) -> None:
    degree = pd.concat([edges["source"], edges["target"]]).value_counts()
    selected = set(degree.head(max_nodes).index)
    subset = edges[edges["source"].isin(selected) & edges["target"].isin(selected)].copy()
    graph = nx.Graph()
    for node, value in degree.loc[list(selected)].items():
        graph.add_node(node, degree=float(value))
    for _, row in subset.iterrows():
        graph.add_edge(row.source, row.target, weight=float(row.weight), lift=float(row.lift))
    positions = nx.spring_layout(graph, seed=42, weight="weight", iterations=80)
    fig, ax = plt.subplots(figsize=(13, 11))
    degrees = np.array([graph.nodes[n]["degree"] for n in graph])
    lifts = np.array([data["lift"] for _, _, data in graph.edges(data=True)])
    weights = np.array([data["weight"] for _, _, data in graph.edges(data=True)])
    norm = Normalize(vmin=float(lifts.min()), vmax=float(lifts.max()))
    nx.draw_networkx_nodes(graph, positions, node_size=30 + 250 * degrees / degrees.max(),
                           node_color=degrees, cmap="Blues", alpha=0.9, ax=ax)
    nx.draw_networkx_edges(
        graph, positions,
        width=0.4 + 3 * np.log1p(weights) / np.log1p(weights.max()),
        edge_color=lifts, edge_cmap=plt.cm.viridis, edge_vmin=norm.vmin, edge_vmax=norm.vmax,
        alpha=0.55, ax=ax,
    )
    nx.draw_networkx_labels(
        graph, positions,
        labels={node: node for node in graph},
        font_size=5, bbox={"alpha": 0.6, "color": "white", "pad": 0.2}, ax=ax,
    )
    nx.draw_networkx_edge_labels(
        graph, positions,
        edge_labels={(left, right): f"w={data['weight']:.0f}; l={data['lift']:.1f}"
                     for left, right, data in graph.edges(data=True)},
        font_size=4, rotate=False, label_pos=0.5, bbox={"alpha": 0.7, "color": "white", "pad": 0.1}, ax=ax,
    )
    ax.set_title(f"Haplograph: top {len(selected)} nodes by degree\n(edge width = log weight; edge color = lift)")
    ax.axis("off")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="Lift")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    subset.to_csv(output.with_suffix(".csv"), index=False)


def draw_combined(edges: pd.DataFrame, graph_features: pd.DataFrame, output: Path, max_blocks: int) -> None:
    block_degree = pd.concat([edges["source"], edges["target"]]).value_counts()
    blocks = set(block_degree.head(max_blocks).index)
    subset = edges[edges["source"].isin(blocks) | edges["target"].isin(blocks)].copy()
    proteins: set[str] = set()
    graph = nx.Graph()
    for block in blocks:
        graph.add_node(f"B:{block}", type="haploblock", degree=float(block_degree[block]))
    for _, row in subset.iterrows():
        source, target = f"B:{row.source}", f"B:{row.target}"
        if source in graph and target in graph:
            graph.add_edge(source, target, relation="co_occurs", weight=float(row.weight), lift=float(row.lift))
        for block, column in ((row.source, "source_protein"), (row.target, "target_protein")):
            if block not in blocks:
                continue
            for protein in str(row.get(column, "")).split(";"):
                if protein in graph_features.index:
                    proteins.add(protein)
                    graph.add_node(f"P:{protein}", type="protein",
                                   phenotype=float(graph_features.loc[protein, "phenotype_difference"]))
                    graph.add_edge(f"B:{block}", f"P:{protein}", relation="contains", weight=1, lift=1)
    if not proteins:
        raise ValueError("No protein memberships found for the selected haploblocks")
    positions = nx.spring_layout(graph, seed=42, weight="weight", iterations=100)
    fig, ax = plt.subplots(figsize=(14, 11))
    block_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "haploblock"]
    protein_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "protein"]
    protein_signal = np.array([graph.nodes[n]["phenotype"] for n in protein_nodes])
    nx.draw_networkx_nodes(graph, positions, nodelist=block_nodes, node_color="#4c78a8",
                           node_shape="s", node_size=65, label="Haploblock", ax=ax)
    nx.draw_networkx_nodes(graph, positions, nodelist=protein_nodes, node_color=protein_signal,
                           cmap="coolwarm", node_size=85, label="Protein", ax=ax)
    membership = [(u, v) for u, v, d in graph.edges(data=True) if d["relation"] == "contains"]
    cooccurrence = [(u, v) for u, v, d in graph.edges(data=True) if d["relation"] == "co_occurs"]
    nx.draw_networkx_edges(graph, positions, edgelist=membership, edge_color="#999999",
                           width=0.5, alpha=0.35, ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=cooccurrence, edge_color="#d55e00",
                           width=1.2, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(
        graph, positions,
        labels={node: node[2:] for node in graph},
        font_size=5, bbox={"alpha": 0.65, "color": "white", "pad": 0.2}, ax=ax,
    )
    edge_labels = {}
    for left, right, data in graph.edges(data=True):
        if data["relation"] == "co_occurs":
            edge_labels[(left, right)] = f"w={data['weight']:.0f}; l={data['lift']:.1f}"
        else:
            edge_labels[(left, right)] = "contains"
    nx.draw_networkx_edge_labels(
        graph, positions, edge_labels=edge_labels, font_size=4, rotate=False,
        label_pos=0.5, bbox={"alpha": 0.7, "color": "white", "pad": 0.1}, ax=ax,
    )
    ax.set_title("Integrated haploblock–protein network\nprotein color = phenotype 1 minus phenotype 0 expression")
    ax.legend(frameon=False)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def compare_with_model(graph_features: pd.DataFrame, coefficients_path: Path, output_dir: Path) -> dict[str, float]:
    coefficients = pd.read_csv(coefficients_path)
    protein_coefficients = coefficients[coefficients["feature"].str.startswith("proteins__")].copy()
    protein_coefficients["protein_id"] = protein_coefficients["feature"].str.replace("proteins__", "", regex=False)
    protein_coefficients = protein_coefficients.set_index("protein_id")[["coefficient"]]
    comparison = graph_features.join(protein_coefficients, how="inner")
    comparison["absolute_coefficient"] = comparison["coefficient"].abs()
    comparison["graph_degree_rank"] = comparison["graph_degree"].rank(ascending=False, method="min")
    comparison["model_rank"] = comparison["absolute_coefficient"].rank(ascending=False, method="min")
    comparison.to_csv(output_dir / "graph_vs_logistic_regression.csv")
    top_graph = set(comparison.nlargest(20, "graph_degree").index)
    top_model = set(comparison.nlargest(20, "absolute_coefficient").index)
    overlap = len(top_graph & top_model)
    statistics = {
        "proteins_compared": len(comparison),
        "top_20_overlap": overlap,
        "top_20_overlap_fraction": overlap / 20,
        "spearman_degree_vs_absolute_coefficient": comparison["graph_degree"].corr(
            comparison["absolute_coefficient"], method="spearman"
        ),
        "spearman_lift_vs_absolute_coefficient": comparison["graph_lift"].corr(
            comparison["absolute_coefficient"], method="spearman"
        ),
    }
    pd.DataFrame([statistics]).to_csv(output_dir / "graph_vs_logistic_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(comparison["graph_degree"], comparison["absolute_coefficient"], alpha=0.7, color="#0072b2")
    ax.set(xscale="log", yscale="log", xlabel="Haplograph degree", ylabel="Absolute logistic coefficient",
           title="Graph connectivity versus model importance")
    fig.tight_layout()
    fig.savefig(output_dir / "graph_vs_logistic_regression.png", dpi=180)
    plt.close(fig)
    return statistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--graph-features", type=Path, default=Path("federated_data/graph_protein_features.csv"))
    parser.add_argument("--coefficients", type=Path, default=Path("proteomics/logistic_regression_results/model_coefficients.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("proteomics/graph_comparison"))
    parser.add_argument("--max-nodes", type=int, default=60)
    parser.add_argument("--max-blocks", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = read_edges(args.edges)
    graph_features = pd.read_csv(args.graph_features).set_index("protein_id")
    expression = pd.read_csv("proteomics/knowledge_graph/nodes.csv")
    expression = expression[expression["type"] == "protein"].set_index("protein_id")
    graph_features = graph_features.join(expression[["phenotype_difference"]], how="left")
    draw_haplograph(edges, args.output_dir / "haplograph_network.png", args.max_nodes)
    draw_combined(edges, graph_features, args.output_dir / "haplograph_proteomics_network.png", args.max_blocks)
    print(compare_with_model(graph_features, args.coefficients, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
