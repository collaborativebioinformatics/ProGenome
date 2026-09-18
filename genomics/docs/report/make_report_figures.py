#!/usr/bin/env python3
"""Extra figures for the KT report, all from files under genomics/outputs* (nothing hand-typed).

    python docs/report/make_report_figures.py      # writes docs/report/figures/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
G = HERE.parents[1]                       # genomics/
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(G))
import haplokg  # noqa: E402
import haplokg_proteins as hp  # noqa: E402

ANC = {"AFR": "#d55e00", "AMR": "#cc79a7", "EAS": "#009e73", "EUR": "#0072b2", "SAS": "#e69f00"}
TEAL, AMBER, INK, GREY = "#0e7c7b", "#b26a0c", "#17232a", "#8a9599"
OUT = G / "outputs"
BREV = G / "outputs_brev" / "progenome-a100"


def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, fs=8.5, lw=1.2, pad=0.02):
    """Rounded box; note the pad extends the drawn box by `pad` on every side, so leave step >= w + 2*pad + gap."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={min(pad, 0.02)}", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, color=INK, fs=7.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.018, text, ha="center", va="bottom", fontsize=fs, color=GREY)



# --------------------------------------------------------------------- 0. figures produced by the pipeline scripts themselves: copy the current versions in
PIPELINE_FIGURES = {
    "cramers_v.png": OUT / "cooccurrence/chr22/cramers_v_by_phenotype.png",
    "edge_similarity.png": OUT / "cooccurrence/chr22/edge_ancestry_similarity.png",
    "informativeness.png": OUT / "cooccurrence/chr22/informativeness_along_chromosome.png",
    "embedding_individuals.png": OUT / "embeddings/chr22/individuals_gnn_ancestry_svd.png",
    "embedding_clusters.png": OUT / "embeddings/chr22/clusters_gnn_ancestry_svd.png",
    "eda_populations.png": OUT / "eda/chr22/plots/02_populations.png",
    "eda_blocks.png": OUT / "eda/chr22/plots/03_blocks.png",
    "eda_clusters.png": OUT / "eda/chr22/plots/04_clusters.png",
    "eda_cooccurrence.png": OUT / "eda/chr22/plots/05_cooccurrence.png",
    "eda_proteomics.png": OUT / "eda/chr22/plots/07_proteomics.png",
    "graph_region.png": OUT / "graph/chr22/region_45035149-45534032.png",
    "graph_edge_positions.png": OUT / "graph/chr22/edge_positions.png",
}


def sync_pipeline_figures():
    """Copy the plots that cooccurrence_analysis.py, embeddings.py, eda.py and graph_explore.py wrote under outputs/."""
    import shutil
    for name, src in PIPELINE_FIGURES.items():
        if src.exists():
            shutil.copyfile(src, FIG / name)
        else:
            print("   (missing, kept previous copy)", src)


# --------------------------------------------------------------------- 1. workflow
def workflow():
    fig, ax = plt.subplots(figsize=(16, 5.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    W, STEP, PAD = 0.112, 0.141, 0.006          # drawn width = W + 2*PAD = 0.124 < STEP: no overlap, room for the arrow
    stages = [("data.haploblocks.org\nHaploGraph chr22\n(pre-built graph)", TEAL), ("fetch_data.sh\nmd5-verified\ndownload", None),
              ("build_kg.py\nsparse carrier matrix\nPyG HeteroData", None), ("cooccurrence_analysis.py\ncluster / edge\nvs phenotype", None),
              ("baseline.py\nlogistic regression\nshared split", None), ("train_gnn.py\nhetero-GNN\nencoder", None), ("embeddings.py\nsilhouette / kNN\nPCA plots", None)]
    for i, (txt, col) in enumerate(stages):
        x = 0.012 + i * STEP
        box(ax, x, 0.64, W, 0.22, txt, fc="#d9efee" if col else "#ffffff", ec=TEAL if col else INK, fs=7.6, pad=PAD)
        if i < len(stages) - 1: arrow(ax, x + W + PAD, 0.75, x + STEP - PAD, 0.75)
    ax.text(0.012, 0.93, "v1  genome graph -> phenotypes", fontsize=11, weight="bold", color=TEAL)
    stages2 = [("proteomics_synth_1000g.py\n3 sites, 1000G ids\nground truth", AMBER), ("build_kg_v2.py\ngenes, proteins,\nharmonised MEASURED", AMBER),
               ("eda.py\nfull EDA\nreport", None), ("train_gnn_v2.py\ngenome / proteome / both\ncontrols, saliency", None),
               ("proteome_linear_baseline.py\nridge cis test", None), ("graphrag_decoder.py\nNIM LLM\ncited insight", None), ("federated/job.py\nNVFlare FedAvg\n3 sites + site-alone", None)]
    for i, (txt, col) in enumerate(stages2):
        x = 0.012 + i * STEP
        box(ax, x, 0.18, W, 0.24, txt, fc="#f6e7cf" if col else "#ffffff", ec=AMBER if col else INK, fs=7.4, pad=PAD)
        if i < len(stages2) - 1: arrow(ax, x + W + PAD, 0.30, x + STEP - PAD, 0.30)
    ax.text(0.012, 0.5, "v2  + proteomics -> integration -> decoder -> federated", fontsize=11, weight="bold", color=AMBER)
    xk = 0.012 + 2 * STEP + W / 2                      # build_kg.py -> build_kg_v2.py
    arrow(ax, xk, 0.64 - PAD, xk, 0.42 + PAD, "hetero.pt", TEAL)
    ax.text(0.012, 0.05, "Every stage is one script with a Makefile target; outputs land under genomics/outputs/<stage>/chr22/. The Docker image and brev_deploy.sh run the same chain on a GPU.", fontsize=8.5, color=GREY)
    fig.savefig(FIG / "workflow_pipeline.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 2. schema
def schema():
    """Five node boxes with gaps wide enough for the edge labels; labels sit above the arrows, never on them."""
    fig, ax = plt.subplots(figsize=(15, 5.4)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    W, H, Y = 0.13, 0.2, 0.42
    xs = [0.03, 0.235, 0.44, 0.645, 0.85]
    names = ["Individual", "Cluster", "Block", "Gene", "Protein"]
    counts = ["2,548", "6,551", "669", "458", "460"]
    cols = ["#e8e6f7", "#d9efee", "#d9efee", "#f6e7cf", "#f6e7cf"]
    props = ["ancestry, population, sex,\nsite, phenotype (labels)", "support, block statistics,\nSVD-32 vector", "length, n_clusters,\nentropy, dominance", "coordinates", "coordinates, isoforms"]
    for x, n, c, col, pr in zip(xs, names, counts, cols, props):
        box(ax, x, Y, W, H, f"{n}\n({c})", fc=col, fs=11, pad=0.008)
        ax.text(x + W / 2, Y - 0.05, pr, ha="center", va="top", fontsize=8, color=GREY)
    edges = ["CARRIES\n2,365,574", "IN_BLOCK\n6,551", "OVERLAPS\n1,063", "ENCODES\n460"]
    for i, lbl in enumerate(edges):
        x1, x2 = xs[i] + W + 0.008, xs[i + 1] - 0.008
        ax.annotate("", xy=(x2, Y + H / 2), xytext=(x1, Y + H / 2), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=INK))
        ax.text((x1 + x2) / 2, Y + H / 2 + 0.05, lbl, ha="center", va="bottom", fontsize=8.5, color=INK)
    # self relations drawn as loops above the box
    for x, lbl in ((xs[1], "CO_OCCURS {weight, lift}\n187,030"), (xs[2], "NEXT_BLOCK\n668")):
        ax.annotate("", xy=(x + W * 0.72, Y + H + 0.01), xytext=(x + W * 0.28, Y + H + 0.01), arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-1.3", lw=1.2, color=INK))
        ax.text(x + W / 2, Y + H + 0.2, lbl, ha="center", va="bottom", fontsize=8.5, color=INK)
    # MEASURED arc below, Individual -> Protein
    ax.annotate("", xy=(xs[4] + W / 2, Y - 0.01), xytext=(xs[0] + W / 2, Y - 0.01), arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.28", lw=1.4, color=AMBER))
    ax.text(0.5, 0.05, "MEASURED {log2, z}   1,065,712   (Individual -> Protein: one edge per observed protein level; a missing value creates no edge)", ha="center", fontsize=9, color=AMBER)
    ax.text(0.5, 0.97, "Knowledge-graph schema (chr22 counts). Labels live on the Individual node, never as neighbours.", ha="center", va="top", fontsize=11, weight="bold", color=INK)
    fig.savefig(FIG / "schema_diagram.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 3. genome -> clusters schematic
def genome_schematic():
    fig, ax = plt.subplots(figsize=(14, 6.4)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    H = dict(fontsize=10, weight="bold", color=INK)
    ax.text(0.02, 0.965, "1. chromosome 22 is cut into haploblocks at recombination hotspots (669 blocks tiling 17.1-50.2 Mb)", **H)
    ax.add_patch(FancyBboxPatch((0.03, 0.845), 0.94, 0.055, boxstyle="round,pad=0.004", fc="#dfe6e8", ec=GREY))
    for xx in np.linspace(0.03, 0.97, 9)[1:-1]:
        ax.plot([xx, xx], [0.845, 0.9], color=INK, lw=1)
    ax.text(0.2, 0.815, "block 1", ha="center", fontsize=8.5, color=GREY); ax.text(0.5, 0.815, "block k", ha="center", fontsize=8.5, color=GREY); ax.text(0.85, 0.815, "block 669", ha="center", fontsize=8.5, color=GREY)
    ax.text(0.02, 0.73, "2. two phased haplotypes per person and block", **H)
    for j, (name, seqs) in enumerate([("HG00096", ["ACGTTGCA...", "ACGATGCA..."]), ("HG00097", ["ACGATGCA...", "TCGTTGCA..."]), ("NA21144", ["ACGATGCA...", "ACGATGCA..."])]):
        ax.text(0.03, 0.66 - j * 0.065, name, fontsize=9, family="monospace")
        for k, sq in enumerate(seqs):
            ax.text(0.13 + k * 0.17, 0.66 - j * 0.065, f"hap{k}: {sq}", fontsize=8.5, family="monospace", color=INK)
    ax.text(0.53, 0.73, "3. MMseqs2 groups near-identical haplotypes into clusters", **H)
    for j, (cl, members, col) in enumerate([("cluster 1", "HG00096/hap0, HG00097/hap0, ...  (1,983 carriers)", TEAL), ("cluster 2", "HG00096/hap1, HG00097/hap0, NA21144/hap0+1, ...  (1,142)", AMBER), ("cluster 3", "HG00097/hap1, ...  (559)", "#7a5af8")]):
        ax.add_patch(FancyBboxPatch((0.54, 0.645 - j * 0.065), 0.018, 0.035, boxstyle="round,pad=0.002", fc=col, ec=col))
        ax.text(0.57, 0.662 - j * 0.065, f"{cl}:  {members}", fontsize=8.5, va="center")
    ax.text(0.02, 0.42, "4. carrier matrix = the graph's person-to-cluster edges (1 if either haplotype is in the cluster)", **H)
    hdr = ["", "cluster 1", "cluster 2", "cluster 3", "..."]
    rows = [["HG00096", 1, 1, 0, "..."], ["HG00097", 1, 1, 1, "..."], ["NA21144", 0, 1, 0, "..."]]
    for c_, hname in enumerate(hdr):
        ax.text(0.07 + c_ * 0.1, 0.35, hname, fontsize=9, weight="bold", ha="center")
    for r_, row in enumerate(rows):
        for c_, v in enumerate(row):
            ax.text(0.07 + c_ * 0.1, 0.29 - r_ * 0.06, str(v), fontsize=9, ha="center", family="monospace" if c_ else None,
                    color=(TEAL if v == 1 else GREY) if isinstance(v, int) else INK)
    ax.text(0.53, 0.35, "5. filter: keep clusters with 25 <= carriers <= N-25:  248,254 -> 6,551", fontsize=9)
    ax.text(0.53, 0.29, "6. co-occurrence: clusters found together in people more than chance -> CO_OCCURS edges (lift)", fontsize=9)
    ax.text(0.53, 0.23, "7. every person carries ~928 kept clusters (2 haplotypes x 669 blocks, minus filtered ones)", fontsize=9)
    ax.text(0.53, 0.17, "8. the carrier matrix is the CARRIES edge list; the SVD of it gives the starting embeddings", fontsize=9)
    fig.savefig(FIG / "genome_to_graph.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 4. federated topology
def federated_topology():
    fig, ax = plt.subplots(figsize=(14, 6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    W, H, Y = 0.2, 0.34, 0.5
    sites = [("Site 1 (hospital)", "835 people\n774,753 CARRIES\n355,294 MEASURED"), ("Site 2 (hospital)", "835 people"), ("Site 3 (hospital)", "833 people")]
    for i, (name, detail) in enumerate(sites):
        x = 0.03 + i * 0.235
        ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0.008", fc="#ffffff", ec="#7a5af8", ls="--", lw=1.5))
        ax.text(x + W / 2, Y + H - 0.05, name, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + W / 2, Y + H / 2 + 0.01, "Individual nodes + labels\nCARRIES and MEASURED edges\n(never leave the site)", ha="center", va="center", fontsize=8.8)
        ax.text(x + W / 2, Y + 0.02, detail, ha="center", va="bottom", fontsize=7.5, color=GREY)
        ax.annotate("", xy=(x + W / 2, 0.93), xytext=(x + W / 2, Y + H + 0.01), arrowprops=dict(arrowstyle="-", lw=1.2, color=INK))
    ax.annotate("", xy=(0.86, 0.93), xytext=(0.13, 0.93), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=INK))
    ax.text(0.5, 0.955, "weights only, once per round (no rows, no protein values, no embeddings of people)", ha="center", fontsize=9, color=INK)
    box(ax, 0.76, Y, 0.21, H, "NVFlare server\nFedAvg\n\naverages the three\nstate dicts, keeps the best\nby validation balanced\naccuracy, sends it back", fc="#ffffff", fs=8.8, pad=0.008)
    ax.annotate("", xy=(0.03 + 2 * 0.235 + W + 0.008, Y + 0.08), xytext=(0.76 - 0.008, Y + 0.08), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK))
    ax.text(0.735, Y + 0.035, "global model", ha="center", fontsize=8, color=GREY)
    ax.add_patch(FancyBboxPatch((0.03, 0.1), 0.67, 0.3, boxstyle="round,pad=0.008", fc="#d9efee", ec=TEAL, lw=1.5))
    ax.text(0.365, 0.33, "Shared and public: the Cluster, Block, Gene, Protein graph", ha="center", va="center", fontsize=10, weight="bold", color=INK)
    ax.text(0.365, 0.25, "CO_OCCURS, IN_BLOCK, NEXT_BLOCK, OVERLAPS, ENCODES edges and the encoder weights;\nidentical at every site (data.haploblocks.org + UniProt)", ha="center", va="center", fontsize=8.8, color=INK)
    ax.text(0.365, 0.15, "each site trains 5 local epochs on its own people per round; 30 rounds", ha="center", va="center", fontsize=8.5, color=GREY)
    box(ax, 0.76, 0.1, 0.21, 0.3, "Result\n\nglobal model AUC 0.987-0.998\nvs central 0.992-0.995\non the same 376 held-out\npeople (three runs)", fc="#ffffff", fs=8.8, pad=0.008)
    fig.savefig(FIG / "federated_topology.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 5. person neighbourhood (real)
def person_neighbourhood():
    ctx_path = OUT / "graphrag" / "chr22" / "HG00103_context.json"
    if not ctx_path.exists():
        return
    ctx = json.loads(ctx_path.read_text())
    Gp = nx.Graph(); who = ctx["individual"]["id"]
    Gp.add_node(who, kind="person")
    for cl in ctx["most_ancestry_informative_clusters_carried"][:6]:
        Gp.add_node(cl["cluster_id"], kind="cluster", anc=cl["enriched_in"]); Gp.add_edge(who, cl["cluster_id"])
        Gp.add_node(cl["block_id"], kind="block"); Gp.add_edge(cl["cluster_id"], cl["block_id"])
        for g in cl["genes_in_block"][:3]:
            Gp.add_node(g, kind="gene"); Gp.add_edge(cl["block_id"], g)
        for pr in cl["proteins_in_block"][:3]:
            Gp.add_node(pr, kind="protein"); Gp.add_edge(cl["genes_in_block"][0] if cl["genes_in_block"] else cl["block_id"], pr)
    for pr in ctx["most_extreme_protein_levels"][:5]:
        Gp.add_node(pr["protein_id"], kind="protein"); Gp.add_edge(who, pr["protein_id"], measured=True, z=pr["harmonised_z"])
    for nb in ctx["nearest_individuals_in_gnn_embedding"][:3]:
        Gp.add_node(nb["individual_id"], kind="neighbour", anc=nb["ancestry"]); Gp.add_edge(who, nb["individual_id"], nn=True)
    pos = nx.spring_layout(Gp, seed=3, k=0.9)
    fig, ax = plt.subplots(figsize=(12, 9.5))
    kinds = {"person": ("#7a5af8", 900), "neighbour": ("#c7bfff", 500), "cluster": (TEAL, 420), "block": ("#9fd3d1", 420), "gene": (AMBER, 380), "protein": ("#f2c98a", 380)}
    for kind, (col, size) in kinds.items():
        ns = [n for n, d in Gp.nodes(data=True) if d["kind"] == kind]
        nx.draw_networkx_nodes(Gp, pos, nodelist=ns, node_color=col, node_size=size, ax=ax, edgecolors="white")
    meas = [(u, v) for u, v, d in Gp.edges(data=True) if d.get("measured")]
    nnb = [(u, v) for u, v, d in Gp.edges(data=True) if d.get("nn")]
    other = [(u, v) for u, v, d in Gp.edges(data=True) if not d.get("measured") and not d.get("nn")]
    nx.draw_networkx_edges(Gp, pos, edgelist=other, ax=ax, edge_color="#b8c2c5", width=1.2)
    nx.draw_networkx_edges(Gp, pos, edgelist=meas, ax=ax, edge_color=AMBER, width=1.6, style="dashed")
    nx.draw_networkx_edges(Gp, pos, edgelist=nnb, ax=ax, edge_color="#7a5af8", width=1.2, style="dotted")
    labels = {n: (n.replace("chr22_", "").replace("_cluster", "\nc") if Gp.nodes[n]["kind"] in ("cluster", "block") else n) for n in Gp}
    nx.draw_networkx_labels(Gp, pos, labels=labels, font_size=6.8, ax=ax, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=11, label=lbl) for lbl, col in
               [(f"this person ({who})", "#7a5af8"), ("nearest people in the GNN embedding", "#c7bfff"), ("haploblock cluster this person carries", TEAL),
                ("block containing that cluster", "#9fd3d1"), ("gene overlapping the block", AMBER), ("protein (encoded by the gene, or measured in this person)", "#f2c98a")]]
    handles += [Line2D([0], [0], color="#b8c2c5", lw=1.6, label="graph edge: CARRIES, IN_BLOCK, OVERLAPS, ENCODES"),
                Line2D([0], [0], color=AMBER, lw=1.6, ls="--", label="MEASURED: one of this person's most extreme protein levels"),
                Line2D([0], [0], color="#7a5af8", lw=1.4, ls=":", label="nearest neighbour in the 64-d GNN embedding")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=8, handletextpad=0.6, columnspacing=1.4)
    ax.set_title(f"What the decoder retrieves for {who} ({ctx['individual']['ancestry']}, {ctx['individual']['population']}); GNN prediction: {ctx['gnn_phenotype_prediction']['predicted']}", fontsize=10.5)
    ax.axis("off"); fig.savefig(FIG / "person_neighbourhood.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 6. sites composition / batch
def sites():
    synth = OUT / "proteomics_synth" / "chr22"
    meta = pd.read_csv(synth / "sample_metadata.csv"); long = pd.read_csv(synth / "measured_long.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    comp = meta.groupby(["site", "ancestry"]).size().unstack(fill_value=0)
    comp.plot(kind="bar", stacked=True, ax=axes[0], color=[ANC[a] for a in comp.columns], width=0.7)
    axes[0].set_title("people per site by ancestry (mixed by design)"); axes[0].set_ylabel("people"); axes[0].legend(frameon=False, fontsize=7); axes[0].tick_params(axis="x", rotation=0)
    prev = meta.groupby("site")["phenotype"].mean()
    axes[1].bar(prev.index, prev.values, color="#7a5af8", width=0.6); axes[1].set_ylim(0, 0.6); axes[1].set_title("case prevalence per site"); axes[1].set_ylabel("fraction cases")
    for i, v in enumerate(prev.values): axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    n_per = meta.groupby("site").size(); miss = 1 - long.groupby("site").size() / (n_per * long["protein_id"].nunique())
    axes[2].bar(miss.index, miss.values, color=AMBER, width=0.6); axes[2].set_title("missing protein values per site"); axes[2].set_ylabel("fraction missing"); axes[2].set_ylim(0, 0.12)
    for i, v in enumerate(miss.values): axes[2].text(i, v + 0.002, f"{v:.1%}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "sites_composition.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    harm = hp.harmonise(long)
    prots = long.groupby("protein_id")["log2_intensity"].median().sort_values().index[::23][:20]
    raw = long[long.protein_id.isin(prots)].groupby(["protein_id", "site"])["log2_intensity"].median().unstack().loc[prots]
    z = harm[harm.protein_id.isin(prots)].groupby(["protein_id", "site"])["z"].median().unstack().loc[prots]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    for s, col in zip(raw.columns, ["#0072b2", "#d55e00", "#009e73"]):
        axes[0].plot(range(len(prots)), raw[s], marker="o", ms=3, color=col, label=s); axes[1].plot(range(len(prots)), z[s], marker="o", ms=3, color=col, label=s)
    axes[0].set_title("median log2 intensity per site: raw (batch offsets visible)"); axes[1].set_title("after the harmoniser (robust z per site x protein)")
    axes[0].set_xlabel("20 proteins, ordered by abundance"); axes[1].set_xlabel("20 proteins, ordered by abundance"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "sites_batch_effect.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    truth = json.loads((synth / "ground_truth.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    axes[0].bar(range(20), sorted([c["beta_phenotype"] for c in truth["causal_clusters"]]), color=TEAL); axes[0].set_title("causal clusters: effect on phenotype logit"); axes[0].set_xlabel("20 clusters (sorted)")
    axes[1].bar(range(20), sorted([c["beta_cis"] for c in truth["causal_clusters"]]), color=AMBER); axes[1].set_title("cis effect on the block's protein (log2)")
    bp = np.array(truth["protein_effects"]["beta_phenotype"]); axes[2].hist(bp[bp != 0], bins=20, color="#7a5af8"); axes[2].set_title(f"phenotype effect on {int((bp != 0).sum())} responsive proteins"); axes[2].set_xlabel("beta (log2)")
    freq = [c for c in truth["causal_clusters"]]
    kg = haplokg.load_kg(OUT / "kg" / "chr22"); sup = kg["clusters"].set_index("cluster_idx")["support_frac"]
    axes[3].hist([sup.loc[c["cluster_idx"]] for c in freq], bins=10, color=GREY); axes[3].set_title("carrier frequency of causal clusters"); axes[3].set_xlabel("fraction of people")
    fig.suptitle("Ground truth of the synthetic proteome (saved in ground_truth.json)", fontsize=10); fig.tight_layout(); fig.savefig(FIG / "ground_truth_effects.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 7. training curves, saliency, ridge, results bars
def gnn_results():
    base = BREV / "gnn_v2" / "chr22"
    runs = {"genome (raw)": "phenotype_genome_raw", "proteome (MLP)": "phenotype_proteome_svd", "genome + proteome (raw)": "phenotype_both_raw", "site control": "site_both_svd"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, d in runs.items():
        h = base / d / "history.csv"
        if h.exists():
            hist = pd.read_csv(h); axes[0].plot(hist["epoch"], hist["loss"], label=name); axes[1].plot(hist["epoch"], hist["val_score"], label=name)
    axes[0].set_title("training loss"); axes[0].set_xlabel("epoch"); axes[1].set_title("validation balanced accuracy (early-stopping criterion)"); axes[1].set_xlabel("epoch"); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "training_curves.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    metrics = {}
    for name, d in {**runs, "genome (svd)": "phenotype_genome_svd", "genome + proteome (svd)": "phenotype_both_svd"}.items():
        m = base / d / "metrics.json"
        if m.exists(): metrics[name] = json.loads(m.read_text())["test"]
    order = [k for k in ["genome (svd)", "genome (raw)", "proteome (MLP)", "genome + proteome (svd)", "genome + proteome (raw)"] if k in metrics]
    fig, ax = plt.subplots(figsize=(9, 3.6)); xs = np.arange(len(order))
    ax.bar(xs - 0.18, [metrics[k].get("roc_auc", np.nan) for k in order], 0.36, color=TEAL, label="AUC")
    ax.bar(xs + 0.18, [metrics[k]["balanced_accuracy"] for k in order], 0.36, color=AMBER, label="balanced accuracy")
    for i, k in enumerate(order):
        ax.text(i - 0.18, metrics[k].get("roc_auc", 0) + 0.01, f"{metrics[k].get('roc_auc', float('nan')):.2f}", ha="center", fontsize=8)
        ax.text(i + 0.18, metrics[k]["balanced_accuracy"] + 0.01, f"{metrics[k]['balanced_accuracy']:.2f}", ha="center", fontsize=8)
    if "site control" in metrics:
        ax.axhline(metrics["site control"]["balanced_accuracy"], ls="--", color="#7a5af8", lw=1.2, label=f"site (batch) control: balanced accuracy {metrics['site control']['balanced_accuracy']:.2f}, chance 0.33")
    ax.set_xticks(xs); ax.set_xticklabels(order, fontsize=8); ax.set_ylim(0.2, 1.15); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3); ax.set_title("Synthetic phenotype on the held-out people: genome vs proteome vs both")
    fig.tight_layout(); fig.savefig(FIG / "results_modalities.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    sal = base / "phenotype_both_raw" / "saliency_top100.csv"
    if sal.exists():
        s = pd.read_csv(sal).head(20)
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.bar(range(20), s["saliency"], color=[TEAL if c else GREY for c in s["is_causal"]])
        ax.set_xticks(range(20)); ax.set_xticklabels([c.replace("chr22_", "").replace("_cluster", "\nc") for c in s["cluster_id"]], fontsize=6.5, rotation=90)
        ax.set_ylabel("saliency (gradient of case logit)"); ax.set_title(f"Top-20 clusters by saliency (teal = ground-truth causal: {int(s['is_causal'].sum())}/20, chance 1.2)")
        fig.tight_layout(); fig.savefig(FIG / "saliency_top20.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    r2 = OUT / "gnn_v2" / "chr22" / "proteome_ridge_baseline" / "protein_r2.csv"
    if r2.exists():
        t_ = pd.read_csv(r2)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.hist(t_.loc[~t_.is_cis, "test_r2"].dropna(), bins=40, color=GREY, alpha=0.7, label="440 other proteins")
        for v in t_.loc[t_.is_cis, "test_r2"].dropna(): ax.axvline(v, color=AMBER, lw=1.2)
        ax.axvline(-9, color=AMBER, lw=1.2, label="20 cis proteins (vertical lines)"); ax.set_xlim(-0.3, 0.3)
        ax.set_xlabel("test R-squared of per-protein ridge from the carrier row"); ax.set_ylabel("proteins"); ax.legend(frameon=False, fontsize=8)
        ax.set_title("Genome -> proteome: only strong cis effects are recoverable\n(4 of 20 cis proteins above R2 0.1, 0 of 440 others)", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG / "ridge_r2.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # baseline vs gnn on real labels
    bl = json.loads((OUT / "baseline" / "chr22" / "metrics.json").read_text())["targets"]
    gnn = {}
    for tgt, d in {"ancestry": "ancestry_svd", "population": "population_raw", "sex": "sex_svd"}.items():
        m = OUT / "gnn" / "chr22" / d / "metrics.json"
        if m.exists(): gnn[tgt] = json.loads(m.read_text())["test"]["balanced_accuracy"]
    fig, ax = plt.subplots(figsize=(7, 3.4)); xs = np.arange(3); tg = ["ancestry", "population", "sex"]
    ax.bar(xs - 0.18, [bl[t]["test"]["balanced_accuracy"] for t in tg], 0.36, color=GREY, label="logistic regression")
    ax.bar(xs + 0.18, [gnn.get(t, np.nan) for t in tg], 0.36, color=TEAL, label="GNN")
    ax.axhline(0.5, ls=":", color="#999"); ax.set_xticks(xs); ax.set_xticklabels(["ancestry (5)", "population (26)", "sex (control)"]); ax.set_ylim(0, 1.05); ax.set_ylabel("test balanced accuracy"); ax.legend(frameon=False, fontsize=8)
    for i, t in enumerate(tg):
        ax.text(i - 0.18, bl[t]["test"]["balanced_accuracy"] + 0.01, f"{bl[t]['test']['balanced_accuracy']:.2f}", ha="center", fontsize=8); ax.text(i + 0.18, gnn.get(t, 0) + 0.01, f"{gnn.get(t, float('nan')):.2f}", ha="center", fontsize=8)
    ax.set_title("Real labels: baseline vs GNN (sex must stay at chance)"); fig.tight_layout(); fig.savefig(FIG / "baseline_vs_gnn.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    eq = OUT / "embeddings" / "chr22" / "embedding_quality.csv"
    if eq.exists():
        e = pd.read_csv(eq); e = e[e["embedding"].isin(["svd32", "gnn_ancestry_svd", "gnn_population_svd", "gnn_sex_svd"])]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        axes[0].bar(e["embedding"], e["silhouette_ancestry"], color=TEAL); axes[0].set_title("silhouette by ancestry (higher = tighter groups)"); axes[0].tick_params(axis="x", labelsize=7)
        axes[1].bar(e["embedding"], e["knn5_test_balanced_accuracy_ancestry"], color=TEAL, label="ancestry"); axes[1].bar(e["embedding"], e["knn5_test_balanced_accuracy_sex"], color=GREY, alpha=0.6, label="sex (control)")
        axes[1].set_title("5-NN test balanced accuracy in the embedding"); axes[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); axes[1].tick_params(axis="x", labelsize=7)
        fig.tight_layout(); fig.savefig(FIG / "embedding_quality.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 8. federated vs central, inference
def federated_and_inference():
    fed10 = OUT / "federated" / "chr22" / "evaluation_10rounds.json"; fed30 = OUT / "federated" / "chr22" / "evaluation.json"
    if fed30.exists():
        d30 = json.loads(fed30.read_text()); d10 = json.loads(fed10.read_text()) if fed10.exists() else None
        fig, ax = plt.subplots(figsize=(8, 3.4))
        names, auc, bal = ["central"], [d30["central_model_same_test_people"]["roc_auc"]], [d30["central_model_same_test_people"]["balanced_accuracy"]]
        if d10: names.append("federated 10 rounds"); auc.append(d10["federated_global_model"]["auc"]); bal.append(d10["federated_global_model"]["balanced_accuracy"])
        names.append("federated 30 rounds"); auc.append(d30["federated_global_model"]["auc"]); bal.append(d30["federated_global_model"]["balanced_accuracy"])
        for s in d30["federated_per_site"]: names.append(f"fed 30, {s['site']} (own test)"); auc.append(s["auc"]); bal.append(s["balanced_accuracy"])
        xs = np.arange(len(names)); ax.bar(xs - 0.18, auc, 0.36, color=TEAL, label="AUC"); ax.bar(xs + 0.18, bal, 0.36, color=AMBER, label="balanced accuracy")
        for i in range(len(names)): ax.text(i - 0.18, auc[i] + 0.005, f"{auc[i]:.3f}", ha="center", fontsize=7); ax.text(i + 0.18, bal[i] + 0.005, f"{bal[i]:.2f}", ha="center", fontsize=7)
        ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7.5, rotation=15); ax.set_ylim(0.8, 1.06); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0)); ax.set_title("Federated global model vs central model (same held-out people)", pad=14)
        fig.tight_layout(); fig.savefig(FIG / "federated_vs_central.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    inf = BREV / "gnn" / "chr22" / "ancestry_node2vec" / "inference"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].bar(["M2 CPU", "A100"], [0.95, 0.10], color=[GREY, TEAL]); axes[0].set_ylabel("seconds per epoch"); axes[0].set_title("GNN training: one full-graph epoch")
    for i, v in enumerate([0.95, 0.10]): axes[0].text(i, v + 0.02, f"{v:.2f} s", ha="center", fontsize=8)
    ms = []
    for name, f_ in [("eager", "benchmark_none_fp32.json"), ("torch.compile", "benchmark_inductor_fp32.json")]:
        pth = inf / f_
        if pth.exists(): d = json.loads(pth.read_text()); ms.append((name, d.get("compiled_ms_per_full_graph", d.get("eager_ms_per_full_graph"))))
    if ms:
        axes[1].bar([m[0] for m in ms], [m[1] for m in ms], color=[GREY, TEAL]); axes[1].set_ylabel("ms per full-graph inference (A100)"); axes[1].set_title("inference: 2,548 people in one pass")
        for i, (n, v) in enumerate(ms): axes[1].text(i, v + 0.5, f"{v:.1f} ms", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "compute_benchmarks.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 9. more EDA: blocks entropy/dominance, informative clusters heatmap, sex by population
def eda_extra():
    kg = haplokg.load_kg(OUT / "kg" / "chr22"); bl = kg["blocks"]; ind = kg["individuals"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sc = axes[0].scatter(bl["dominance"], bl["shannon_entropy"], c=np.log10(bl["n_clusters"]), s=10, cmap="viridis"); axes[0].set_xlabel("dominance (share of the largest cluster)"); axes[0].set_ylabel("Shannon entropy of clusters"); fig.colorbar(sc, ax=axes[0], label="log10 clusters in block")
    axes[0].set_title("blocks: one common haplotype vs many rare ones")
    pop = ind.dropna(subset=["ancestry"]).groupby(["population", "sex"]).size().unstack(fill_value=0)
    pop.plot(kind="bar", stacked=True, ax=axes[1], color=["#7b3294", "#008837"], width=0.8); axes[1].set_title("sex within each population"); axes[1].tick_params(axis="x", labelsize=6.5); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "eda_blocks_populations.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    assoc = pd.read_csv(OUT / "cooccurrence" / "chr22" / "cluster_phenotype_association.csv").sort_values("ancestry_cramers_v", ascending=False).head(30)
    cols = [c for c in assoc.columns if c.startswith("carrier_frac_")]
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(assoc[cols].to_numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace("carrier_frac_", "") for c in cols]); ax.set_yticks(range(30)); ax.set_yticklabels([c.replace("chr22_", "") for c in assoc["cluster_id"]], fontsize=6.5)
    fig.colorbar(im, ax=ax, label="fraction of carriers within ancestry"); ax.set_title("The 30 most ancestry-informative clusters: who carries them")
    fig.tight_layout(); fig.savefig(FIG / "informative_clusters_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 10. ground truth vs prediction: confusion matrices
def confusion_matrices():
    runs = [("real ancestry: GNN (SVD input)", OUT / "gnn" / "chr22" / "ancestry_svd"),
            ("synthetic phenotype: genome + proteome (graph)", BREV / "gnn_v2" / "chr22" / "phenotype_both_raw"),
            ("synthetic phenotype: genome only (graph)", BREV / "gnn_v2" / "chr22" / "phenotype_genome_raw"),
            ("sex, negative control: GNN (SVD input)", OUT / "gnn" / "chr22" / "sex_svd")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    for ax, (title, run) in zip(axes.ravel(), runs):
        p = pd.read_csv(run / "test_predictions.csv")
        ct = pd.crosstab(p["true"], p["pred"]).reindex(index=sorted(p["true"].unique()), columns=sorted(p["true"].unique()), fill_value=0)
        m = ct.to_numpy(); ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=10, color="white" if m[i, j] > m.max() / 2 else INK)
        ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(ct.columns); ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(ct.index)
        ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
        acc = np.trace(m) / m.sum()
        ax.set_title(f"{title}\n{np.trace(m)} / {m.sum()} correct ({acc:.1%}) on the held-out people", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 11. where each data source enters and where each model product goes
def data_flow_map():
    fig, ax = plt.subplots(figsize=(20, 8.5)); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1); ax.axis("off")
    LAB, LABE, PAD = "#efe3f2", "#7b3294", 0.006
    hdr = dict(ha="center", fontsize=11, weight="bold", color=INK)
    # column 1: sources
    SX, SW, SH = 0.01, 0.165, 0.1
    src = [("HaploGraph nodes.csv.gz\nwho carries which cluster", "#d9efee", TEAL, 0.85), ("HaploGraph edges + block_stats\nco-occurrence, block statistics", "#d9efee", TEAL, 0.70),
           ("phenotypes_real.csv\nancestry, population, sex", LAB, LABE, 0.55), ("uniprot_chr22.bed\ngenes, proteins, coordinates", "#f6e7cf", AMBER, 0.40),
           ("proteomics matrices (3 sites)\nlog2 intensity per protein", "#f6e7cf", AMBER, 0.25), ("proteomics metadata\nsite, age, sex, case/control", LAB, LABE, 0.10)]
    for txt, fc, ec, y in src:
        box(ax, SX, y, SW, SH, txt, fc=fc, ec=ec, fs=9, pad=PAD)
    ax.text(SX + SW / 2, 0.985, "DATA SOURCES", **hdr)
    # column 2: knowledge graph
    KX, KW = 0.24, 0.21
    box(ax, KX, 0.62, KW, 0.3, "GRAPH STRUCTURE\nIndividual, Cluster, Block,\nGene, Protein\nCARRIES, CO_OCCURS, IN_BLOCK,\nNEXT_BLOCK, OVERLAPS, ENCODES,\nMEASURED (harmonised z)", fs=9, pad=PAD)
    box(ax, KX, 0.36, KW, 0.2, "NODE FEATURES\ncluster and block statistics;\nperson = SVD-32 of the carrier matrix\n(label-free) or raw row,\nplus protein z and observed mask", fs=9, pad=PAD)
    box(ax, KX, 0.10, KW, 0.2, "LABELS on the person node\nancestry, population, sex,\nsite, phenotype\ntargets and ground truth only:\nnever an edge, never a feature", fc=LAB, ec=LABE, fs=9, pad=PAD)
    ax.text(KX + KW / 2, 0.985, "KNOWLEDGE GRAPH (hetero_v2.pt)", **hdr)
    def a(x1, y1, x2, y2, color=INK, text=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=color))
        if text: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha="center", va="bottom", fontsize=8.5, color=GREY)
    R = SX + SW + PAD; L = KX - PAD
    a(R, 0.90, L, 0.86); a(R, 0.75, L, 0.78); a(R, 0.45, L, 0.70)             # structure <- nodes, edges, BED
    a(R, 0.88, L, 0.50); a(R, 0.30, L, 0.44)                                    # features <- nodes (SVD / raw), matrices
    a(R, 0.60, L, 0.24, LABE); a(R, 0.15, L, 0.18, LABE)                        # labels <- phenotypes, metadata
    # column 3: encoder
    EX, EW = 0.50, 0.17
    box(ax, EX, 0.34, EW, 0.36, "GNN ENCODER\nHeteroConv x 2, hidden 64\nSAGEConv + edge-weighted\nGraphConv\n\ntrained on TRAIN people's labels\nselected on VAL\nreported on TEST (376 people)", fc="#d9efee", ec=TEAL, fs=9, pad=PAD)
    ax.text(EX + EW / 2, 0.985, "ENCODER", **hdr)
    a(KX + KW + PAD, 0.77, EX - PAD, 0.62, text="messages"); a(KX + KW + PAD, 0.46, EX - PAD, 0.52, text="inputs"); a(KX + KW + PAD, 0.20, EX - PAD, 0.42, LABE, "loss")
    # column 4: products
    OX, OW, OH = 0.72, 0.13, 0.09
    outs = [("class probabilities\nper person", 0.86), ("64-d embedding per\nperson and cluster", 0.71), ("saliency per\nhaploblock cluster", 0.56), ("model weights\n(state dict)", 0.41), ("predictions for all\n2,548 people (infer.py)", 0.26)]
    for txt, y in outs:
        box(ax, OX, y, OW, OH, txt, fs=9, pad=PAD); a(EX + EW + PAD, 0.52, OX - PAD, y + OH / 2)
    ax.text(OX + OW / 2, 0.985, "TRAINED MODEL GIVES", **hdr)
    # column 5: consumers
    CX, CW, CH = 0.885, 0.125, 0.12
    cons = [("evaluation against\nground truth\n(test labels, ground_truth.json)", 0.79, "#ffffff", INK), ("LLM decoder\n(NIM GraphRAG)\ncited insight per person", 0.56, "#f6e7cf", AMBER),
            ("NVFlare FedAvg\nonly weights leave a site", 0.34, "#d9efee", TEAL), ("plots, Neo4j, CSVs\nfor the team", 0.12, "#ffffff", INK)]
    for txt, y, fc, ec in cons:
        box(ax, CX, y, CW, CH, txt, fc=fc, ec=ec, fs=8.8, pad=PAD)
    ax.text(CX + CW / 2, 0.985, "CONSUMERS", **hdr)
    OR = OX + OW + PAD; CL = CX - PAD
    a(OR, 0.905, CL, 0.85); a(OR, 0.605, CL, 0.82)          # probabilities, saliency -> evaluation
    a(OR, 0.755, CL, 0.64); a(OR, 0.60, CL, 0.60)           # embedding, saliency -> decoder
    a(OR, 0.455, CL, 0.40)                                  # weights -> FedAvg
    a(OR, 0.305, CL, 0.18)                                  # predictions -> plots
    ax.text(0.51, 0.02, "Phenotypes are never neighbours of the person: the encoder reaches a label only through the loss on training people. Ground truth for the synthetic phenotype is the generator's own label; saliency and ridge are scored against ground_truth.json.", ha="center", fontsize=9, color=GREY)
    fig.savefig(FIG / "data_flow_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)



# --------------------------------------------------------------------- 12. site alone vs federated, and starting-embedding comparison
def federated_site_alone():
    lo = json.loads((G / "outputs_brev" / "progenome-a100-verify" / "federated" / "chr22" / "local_only_vs_federated.json").read_text())
    sites = list(lo["local_only"])
    own = [lo["local_only"][s]["own_test"]["auc"] for s in sites]
    worst = [min(v["auc"] for k, v in lo["local_only"][s].items() if k != "own_test") for s in sites]
    fed = {r["site"]: r["auc"] for r in lo["federated_per_site"]}
    fig, ax = plt.subplots(figsize=(8, 3.6)); xs = np.arange(len(sites)); w = 0.26
    ax.bar(xs - w, own, w, color=GREY, label="site alone, own held-out people")
    ax.bar(xs, worst, w, color="#c9d3d5", label="site alone, worst transfer to another site")
    ax.bar(xs + w, [fed[s] for s in sites], w, color=TEAL, label="federated global model, same people")
    for i in range(len(sites)):
        for off, v in ((-w, own[i]), (0, worst[i]), (w, fed[sites[i]])): ax.text(xs[i] + off, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(sites); ax.set_ylim(0.9, 1.02); ax.set_ylabel("test AUC")
    ax.set_title(f"With and without federation ({lo['steps_per_site']} optimizer steps per site, A100)", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "federated_site_alone.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def init_comparison():
    V = G / "outputs_brev" / "progenome-a100-verify" / "gnn" / "chr22"
    runs = [("SVD-32", OUT / "gnn/chr22/ancestry_svd", OUT / "gnn/chr22/population_svd"), ("Node2Vec-32 (50 epochs)", V / "ancestry_node2vec", V / "population_node2vec"),
            ("free learned", OUT / "gnn/chr22/ancestry_learned", None), ("raw carrier row", None, OUT / "gnn/chr22/population_raw")]
    base = json.loads((OUT / "baseline/chr22/metrics.json").read_text())["targets"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (target, col) in zip(axes, (("ancestry", 1), ("population", 2))):
        names, vals = [], []
        for name, a, pth in runs:
            path = a if target == "ancestry" else pth
            if path is not None and (path / "metrics.json").exists():
                names.append(name); vals.append(json.loads((path / "metrics.json").read_text())["test"]["balanced_accuracy"])
        ax.bar(names, vals, color=[TEAL if n.startswith("SVD") else GREY for n in names])
        ax.axhline(base[target]["test"]["balanced_accuracy"], color=AMBER, ls="--", lw=1, label="logistic regression")
        for i, v in enumerate(vals): ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.2); ax.set_ylabel("test balanced accuracy"); ax.set_title(f"{target}: GNN by starting embedding"); ax.tick_params(axis="x", labelsize=7.5); ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "init_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for fn in (sync_pipeline_figures, workflow, schema, genome_schematic, federated_topology, person_neighbourhood, sites, gnn_results, federated_and_inference, eda_extra, confusion_matrices, data_flow_map, federated_site_alone, init_comparison):
        try:
            fn(); print("ok  ", fn.__name__)
        except Exception as exc:  # keep going; report which figure failed
            print("FAIL", fn.__name__, type(exc).__name__, str(exc)[:160])
    print(len(list(FIG.glob("*.png"))), "figures in", FIG)
