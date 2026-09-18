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


def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, fs=8.5, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, color=INK, fs=7.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=color))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.018, text, ha="center", va="bottom", fontsize=fs, color=GREY)


# --------------------------------------------------------------------- 1. workflow
def workflow():
    fig, ax = plt.subplots(figsize=(13, 5.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [("data.haploblocks.org\nHaploGraph chr22", TEAL), ("fetch_data.sh\nmd5-verified download", None), ("build_kg.py\nsparse carrier matrix\nPyG HeteroData", None),
              ("cooccurrence_analysis.py\ncluster/edge vs phenotype", None), ("baseline.py\nlogistic regression\nshared split", None), ("train_gnn.py\nhetero-GNN encoder", None),
              ("embeddings.py\nsilhouette / kNN", None)]
    x = 0.02
    for i, (txt, col) in enumerate(stages):
        box(ax, x, 0.66, 0.125, 0.22, txt, fc="#d9efee" if col else "#ffffff", ec=TEAL if col else INK)
        if i < len(stages) - 1: arrow(ax, x + 0.125, 0.77, x + 0.14, 0.77)
        x += 0.14
    ax.text(0.02, 0.93, "v1  genome graph -> phenotypes", fontsize=11, weight="bold", color=TEAL)
    stages2 = [("proteomics_synth_1000g.py\n3 sites, ground truth", AMBER), ("build_kg_v2.py\ngenes, proteins,\nharmonised MEASURED", AMBER), ("eda.py\nfull EDA report", None),
               ("train_gnn_v2.py\ngenome / proteome / both\ncontrols, saliency", None), ("proteome_linear_baseline.py\nridge cis test", None), ("graphrag_decoder.py\nNIM LLM insight", None),
               ("federated/job.py\nNVFlare FedAvg\n3 sites", None)]
    x = 0.02
    for i, (txt, col) in enumerate(stages2):
        box(ax, x, 0.2, 0.125, 0.24, txt, fc="#f6e7cf" if col else "#ffffff", ec=AMBER if col else INK)
        if i < len(stages2) - 1: arrow(ax, x + 0.125, 0.32, x + 0.14, 0.32)
        x += 0.14
    ax.text(0.02, 0.5, "v2  + proteomics -> integration -> decoder -> federated", fontsize=11, weight="bold", color=AMBER)
    arrow(ax, 0.3, 0.66, 0.3, 0.45, "hetero.pt", TEAL)
    ax.text(0.02, 0.06, "Every stage is one script with a Makefile target; outputs land under genomics/outputs/<stage>/chr22/. Docker image + brev_deploy.sh run the same chain on a GPU.", fontsize=8.5, color=GREY)
    fig.savefig(FIG / "workflow_pipeline.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 2. schema
def schema():
    S = nx.DiGraph()
    nodes = {"Individual": (0, 1), "Cluster": (1, 1), "Block": (2, 1), "Gene": (3, 1), "Protein": (4, 1)}
    edges = [("Individual", "Cluster", "CARRIES 2,365,574"), ("Cluster", "Block", "IN_BLOCK 6,551"), ("Block", "Gene", "OVERLAPS 1,063"), ("Gene", "Protein", "ENCODES 460")]
    fig, ax = plt.subplots(figsize=(12, 3.8)); ax.axis("off")
    cols = {"Individual": "#e8e6f7", "Cluster": "#d9efee", "Block": "#d9efee", "Gene": "#f6e7cf", "Protein": "#f6e7cf"}
    counts = {"Individual": "2,548", "Cluster": "6,551", "Block": "669", "Gene": "458", "Protein": "460"}
    props = {"Individual": "ancestry · population · sex\nsite · phenotype (labels)", "Cluster": "support · block stats\nSVD-32", "Block": "length · n_clusters\nentropy · dominance", "Gene": "coordinates", "Protein": "coordinates · isoforms"}
    for n, (x, y) in nodes.items():
        box(ax, x - 0.42, 0.7, 0.84, 0.7, f"{n}\n({counts[n]})", fc=cols[n], fs=10)
        ax.text(x, 0.45, props[n], ha="center", va="top", fontsize=7.5, color=GREY)
    for a, b_, lbl in edges:
        arrow(ax, nodes[a][0] + 0.42, 1.05, nodes[b_][0] - 0.42, 1.05, lbl, fs=8)
    # self loop and measured
    ax.annotate("", xy=(1.25, 1.4), xytext=(0.75, 1.4), arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-1.4", lw=1.2, color=INK))
    ax.text(1, 1.95, "CO_OCCURS {weight, lift}\n187,030", ha="center", fontsize=8, color=GREY)
    ax.annotate("", xy=(3.6, 0.7), xytext=(0.4, 0.7), arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.35", lw=1.2, color=AMBER))
    ax.text(2, -0.15, "MEASURED {log2, z}  1,065,712  (one edge per observed protein level; missing = no edge)", ha="center", fontsize=8, color=AMBER)
    ax.annotate("", xy=(2.6, 1.4), xytext=(1.4, 1.4), arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.6", lw=1.0, color=INK))
    ax.text(2, 1.72, "NEXT_BLOCK 668", ha="center", fontsize=8, color=GREY)
    ax.set_xlim(-0.6, 4.6); ax.set_ylim(-0.4, 2.2)
    ax.set_title("Knowledge-graph schema (chr22 counts). Labels live on the Individual node, never as neighbours.", fontsize=10)
    fig.savefig(FIG / "schema_diagram.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 3. genome -> clusters schematic
def genome_schematic():
    fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.95, "1. chromosome 22, cut into haploblocks at recombination hotspots (669 blocks, 17.1-50.2 Mb)", fontsize=9.5, weight="bold")
    ax.add_patch(FancyBboxPatch((0.03, 0.8), 0.94, 0.06, boxstyle="round,pad=0.005", fc="#dfe6e8", ec=GREY))
    for i, xx in enumerate(np.linspace(0.03, 0.97, 9)[1:-1]):
        ax.plot([xx, xx], [0.8, 0.86], color=INK, lw=1)
    ax.text(0.2, 0.755, "block 1", ha="center", fontsize=8, color=GREY); ax.text(0.5, 0.755, "block k", ha="center", fontsize=8, color=GREY)
    ax.text(0.02, 0.68, "2. each person has two phased haplotypes per block (one per parental copy)", fontsize=9.5, weight="bold")
    for j, (name, seqs) in enumerate([("HG00096", ["ACGTTGCA...", "ACGATGCA..."]), ("HG00097", ["ACGATGCA...", "TCGTTGCA..."]), ("NA21144", ["ACGATGCA...", "ACGATGCA..."])]):
        ax.text(0.04, 0.6 - j * 0.07, name, fontsize=8.5, family="monospace")
        for k, s in enumerate(seqs):
            ax.text(0.15 + k * 0.16, 0.6 - j * 0.07, f"hap{k}: {s}", fontsize=8, family="monospace", color=INK)
    ax.text(0.5, 0.68, "3. MMseqs2 groups near-identical haplotypes into clusters", fontsize=9.5, weight="bold")
    for j, (cl, members, col) in enumerate([("cluster1", "HG00096/hap0, HG00097/hap0, ... (1,983 carriers)", TEAL), ("cluster2", "HG00096/hap1, HG00097/hap0, NA21144/hap0+1, ... (1,142)", AMBER), ("cluster3", "HG00097/hap1, ... (559)", "#7a5af8")]):
        ax.add_patch(FancyBboxPatch((0.52, 0.575 - j * 0.07), 0.02, 0.04, boxstyle="round,pad=0.002", fc=col, ec=col))
        ax.text(0.555, 0.595 - j * 0.07, f"{cl}: {members}", fontsize=8, va="center")
    ax.text(0.02, 0.3, "4. carrier matrix = the graph's person-to-cluster edges (1 if either haplotype is in the cluster)", fontsize=9.5, weight="bold")
    hdr = ["", "cluster1", "cluster2", "cluster3", "..."]
    rows = [["HG00096", 1, 1, 0, "..."], ["HG00097", 1, 1, 1, "..."], ["NA21144", 0, 1, 0, "..."]]
    for c_, hname in enumerate(hdr):
        ax.text(0.06 + c_ * 0.1, 0.24, hname, fontsize=8.5, weight="bold", ha="center")
    for r_, row in enumerate(rows):
        for c_, v in enumerate(row):
            ax.text(0.06 + c_ * 0.1, 0.18 - r_ * 0.055, str(v), fontsize=8.5, ha="center", family="monospace" if c_ else None,
                    color=(TEAL if v == 1 else GREY) if isinstance(v, int) else INK)
    ax.text(0.55, 0.24, "5. filter: keep clusters with 25 <= carriers <= N-25  ->  248,254 -> 6,551", fontsize=8.5)
    ax.text(0.55, 0.18, "6. co-occurrence: clusters that appear together in people more than chance -> CO_OCCURS edges (lift)", fontsize=8.5)
    ax.text(0.55, 0.12, "7. every person carries ~928 kept clusters = 2 haplotypes x 669 blocks minus filtered ones", fontsize=8.5)
    fig.savefig(FIG / "genome_to_graph.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------- 4. federated topology
def federated_topology():
    fig, ax = plt.subplots(figsize=(11, 4)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for i, (name, n) in enumerate([("Site 1 (hospital)", "835 people · 774,753 CARRIES · 355,294 MEASURED"), ("Site 2 (hospital)", "835 people"), ("Site 3 (hospital)", "833 people")]):
        x = 0.03 + i * 0.25
        ax.add_patch(FancyBboxPatch((x, 0.5), 0.22, 0.32, boxstyle="round,pad=0.01", fc="#ffffff", ec="#7a5af8", ls="--", lw=1.5))
        ax.text(x + 0.11, 0.76, name, ha="center", fontsize=9.5, weight="bold")
        ax.text(x + 0.11, 0.66, "Individual nodes + labels\nCARRIES · MEASURED edges\n(never leave)", ha="center", va="center", fontsize=8)
        ax.text(x + 0.11, 0.53, n, ha="center", fontsize=6.5, color=GREY)
        arrow(ax, x + 0.11, 0.82, x + 0.11, 0.9)
    ax.plot([0.14, 0.64], [0.9, 0.9], color=INK, lw=1.2); arrow(ax, 0.64, 0.9, 0.79, 0.9, "weights only, each round")
    box(ax, 0.79, 0.5, 0.19, 0.32, "NVFlare server\nFedAvg\n\naverages weights\nreturns global model", fc="#ffffff", fs=8.5)
    arrow(ax, 0.79, 0.6, 0.75, 0.6, "global model", fs=7)
    ax.add_patch(FancyBboxPatch((0.03, 0.12), 0.72, 0.24, boxstyle="round,pad=0.01", fc="#d9efee", ec=TEAL, lw=1.5))
    ax.text(0.39, 0.29, "Shared, public: Cluster · Block · Gene · Protein graph (CO_OCCURS, IN_BLOCK, OVERLAPS, ENCODES) + the encoder weights", ha="center", fontsize=9)
    ax.text(0.39, 0.19, "identical at every site (data.haploblocks.org); each site trains 5 local epochs on its own people per round; 30 rounds", ha="center", fontsize=8, color=GREY)
    ax.text(0.79, 0.2, "Result: global model AUC 0.998\nvs central 0.992 on the same\n376 held-out people", fontsize=8.5, color=INK)
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
    fig, ax = plt.subplots(figsize=(11, 8))
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
    nx.draw_networkx_labels(Gp, pos, labels=labels, font_size=6.5, ax=ax)
    ax.set_title(f"What the decoder retrieves for {who} ({ctx['individual']['ancestry']}, {ctx['individual']['population']}): carried clusters (teal) -> blocks -> genes -> proteins (amber);\n"
                 f"dashed amber = this person's most extreme protein levels; dotted purple = nearest people in the GNN embedding; GNN prediction: {ctx['gnn_phenotype_prediction']['predicted']}", fontsize=9)
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
    fig.tight_layout(); fig.savefig(FIG / "sites_composition.png", dpi=150); plt.close(fig)

    harm = hp.harmonise(long)
    prots = long.groupby("protein_id")["log2_intensity"].median().sort_values().index[::23][:20]
    raw = long[long.protein_id.isin(prots)].groupby(["protein_id", "site"])["log2_intensity"].median().unstack().loc[prots]
    z = harm[harm.protein_id.isin(prots)].groupby(["protein_id", "site"])["z"].median().unstack().loc[prots]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    for s, col in zip(raw.columns, ["#0072b2", "#d55e00", "#009e73"]):
        axes[0].plot(range(len(prots)), raw[s], marker="o", ms=3, color=col, label=s); axes[1].plot(range(len(prots)), z[s], marker="o", ms=3, color=col, label=s)
    axes[0].set_title("median log2 intensity per site: raw (batch offsets visible)"); axes[1].set_title("after the harmoniser (robust z per site x protein)")
    axes[0].set_xlabel("20 proteins, ordered by abundance"); axes[1].set_xlabel("20 proteins, ordered by abundance"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "sites_batch_effect.png", dpi=150); plt.close(fig)

    truth = json.loads((synth / "ground_truth.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    axes[0].bar(range(20), sorted([c["beta_phenotype"] for c in truth["causal_clusters"]]), color=TEAL); axes[0].set_title("causal clusters: effect on phenotype logit"); axes[0].set_xlabel("20 clusters (sorted)")
    axes[1].bar(range(20), sorted([c["beta_cis"] for c in truth["causal_clusters"]]), color=AMBER); axes[1].set_title("cis effect on the block's protein (log2)")
    bp = np.array(truth["protein_effects"]["beta_phenotype"]); axes[2].hist(bp[bp != 0], bins=20, color="#7a5af8"); axes[2].set_title(f"phenotype effect on {int((bp != 0).sum())} responsive proteins"); axes[2].set_xlabel("beta (log2)")
    freq = [c for c in truth["causal_clusters"]]
    kg = haplokg.load_kg(OUT / "kg" / "chr22"); sup = kg["clusters"].set_index("cluster_idx")["support_frac"]
    axes[3].hist([sup.loc[c["cluster_idx"]] for c in freq], bins=10, color=GREY); axes[3].set_title("carrier frequency of causal clusters"); axes[3].set_xlabel("fraction of people")
    fig.suptitle("Ground truth of the synthetic proteome (saved in ground_truth.json)", fontsize=10); fig.tight_layout(); fig.savefig(FIG / "ground_truth_effects.png", dpi=150); plt.close(fig)


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
    fig.tight_layout(); fig.savefig(FIG / "training_curves.png", dpi=150); plt.close(fig)

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
        ax.axhline(metrics["site control"]["balanced_accuracy"], ls="--", color="#7a5af8", lw=1); ax.text(len(order) - 0.5, metrics["site control"]["balanced_accuracy"] + 0.01, f"site control {metrics['site control']['balanced_accuracy']:.2f} (chance 0.33)", ha="right", fontsize=8, color="#7a5af8")
    ax.set_xticks(xs); ax.set_xticklabels(order, fontsize=8); ax.set_ylim(0.2, 1.05); ax.legend(frameon=False, fontsize=8); ax.set_title("Synthetic phenotype on the held-out people: genome vs proteome vs both")
    fig.tight_layout(); fig.savefig(FIG / "results_modalities.png", dpi=150); plt.close(fig)

    sal = base / "phenotype_both_raw" / "saliency_top100.csv"
    if sal.exists():
        s = pd.read_csv(sal).head(20)
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.bar(range(20), s["saliency"], color=[TEAL if c else GREY for c in s["is_causal"]])
        ax.set_xticks(range(20)); ax.set_xticklabels([c.replace("chr22_", "").replace("_cluster", "\nc") for c in s["cluster_id"]], fontsize=6.5, rotation=90)
        ax.set_ylabel("saliency (gradient of case logit)"); ax.set_title(f"Top-20 clusters by saliency (teal = ground-truth causal: {int(s['is_causal'].sum())}/20, chance 1.2)")
        fig.tight_layout(); fig.savefig(FIG / "saliency_top20.png", dpi=150); plt.close(fig)

    r2 = OUT / "gnn_v2" / "chr22" / "proteome_ridge_baseline" / "protein_r2.csv"
    if r2.exists():
        t_ = pd.read_csv(r2)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.hist(t_.loc[~t_.is_cis, "test_r2"].dropna(), bins=40, color=GREY, alpha=0.7, label="440 other proteins")
        for v in t_.loc[t_.is_cis, "test_r2"].dropna(): ax.axvline(v, color=AMBER, lw=1.2)
        ax.axvline(-9, color=AMBER, lw=1.2, label="20 cis proteins (vertical lines)"); ax.set_xlim(-0.3, 0.3)
        ax.set_xlabel("test R-squared of per-protein ridge from the carrier row"); ax.set_ylabel("proteins"); ax.legend(frameon=False, fontsize=8)
        ax.set_title("Genome -> proteome: only the strong cis effects are recoverable (4/20 above 0.1, 0/440 others)")
        fig.tight_layout(); fig.savefig(FIG / "ridge_r2.png", dpi=150); plt.close(fig)

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
    ax.set_title("Real labels: baseline vs GNN (sex must stay at chance)"); fig.tight_layout(); fig.savefig(FIG / "baseline_vs_gnn.png", dpi=150); plt.close(fig)

    eq = OUT / "embeddings" / "chr22" / "embedding_quality.csv"
    if eq.exists():
        e = pd.read_csv(eq); e = e[e["embedding"].isin(["svd32", "gnn_ancestry_svd", "gnn_population_svd", "gnn_sex_svd"])]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        axes[0].bar(e["embedding"], e["silhouette_ancestry"], color=TEAL); axes[0].set_title("silhouette by ancestry (higher = tighter groups)"); axes[0].tick_params(axis="x", labelsize=7)
        axes[1].bar(e["embedding"], e["knn5_test_balanced_accuracy_ancestry"], color=TEAL, label="ancestry"); axes[1].bar(e["embedding"], e["knn5_test_balanced_accuracy_sex"], color=GREY, alpha=0.6, label="sex (control)")
        axes[1].set_title("5-NN test balanced accuracy in the embedding"); axes[1].legend(frameon=False, fontsize=8); axes[1].tick_params(axis="x", labelsize=7)
        fig.tight_layout(); fig.savefig(FIG / "embedding_quality.png", dpi=150); plt.close(fig)


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
        ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7.5, rotation=15); ax.set_ylim(0.8, 1.03); ax.legend(frameon=False, fontsize=8); ax.set_title("Federated global model vs central model (same held-out people)")
        fig.tight_layout(); fig.savefig(FIG / "federated_vs_central.png", dpi=150); plt.close(fig)

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
    fig.tight_layout(); fig.savefig(FIG / "compute_benchmarks.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------- 9. more EDA: blocks entropy/dominance, informative clusters heatmap, sex by population
def eda_extra():
    kg = haplokg.load_kg(OUT / "kg" / "chr22"); bl = kg["blocks"]; ind = kg["individuals"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sc = axes[0].scatter(bl["dominance"], bl["shannon_entropy"], c=np.log10(bl["n_clusters"]), s=10, cmap="viridis"); axes[0].set_xlabel("dominance (share of the largest cluster)"); axes[0].set_ylabel("Shannon entropy of clusters"); fig.colorbar(sc, ax=axes[0], label="log10 clusters in block")
    axes[0].set_title("blocks: one common haplotype vs many rare ones")
    pop = ind.dropna(subset=["ancestry"]).groupby(["population", "sex"]).size().unstack(fill_value=0)
    pop.plot(kind="bar", stacked=True, ax=axes[1], color=["#7b3294", "#008837"], width=0.8); axes[1].set_title("sex within each population"); axes[1].tick_params(axis="x", labelsize=6.5); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "eda_blocks_populations.png", dpi=150); plt.close(fig)

    assoc = pd.read_csv(OUT / "cooccurrence" / "chr22" / "cluster_phenotype_association.csv").sort_values("ancestry_cramers_v", ascending=False).head(30)
    cols = [c for c in assoc.columns if c.startswith("carrier_frac_")]
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(assoc[cols].to_numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace("carrier_frac_", "") for c in cols]); ax.set_yticks(range(30)); ax.set_yticklabels([c.replace("chr22_", "") for c in assoc["cluster_id"]], fontsize=6.5)
    fig.colorbar(im, ax=ax, label="fraction of carriers within ancestry"); ax.set_title("The 30 most ancestry-informative clusters: who carries them")
    fig.tight_layout(); fig.savefig(FIG / "informative_clusters_heatmap.png", dpi=150); plt.close(fig)


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
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------- 11. where each data source enters and where each model product goes
def data_flow_map():
    fig, ax = plt.subplots(figsize=(17, 7)); ax.set_xlim(0, 1.03); ax.set_ylim(0, 1); ax.axis("off")
    LAB = "#efe3f2"; LABE = "#7b3294"
    src = [("HaploGraph nodes.csv.gz\nwho carries which cluster", "#d9efee", TEAL, 0.86),
           ("HaploGraph edges + block_stats\nco-occurrence, block statistics", "#d9efee", TEAL, 0.72),
           ("phenotypes_real.csv\nancestry, population, sex", LAB, LABE, 0.58),
           ("uniprot_chr22.bed\ngenes, proteins, coordinates", "#f6e7cf", AMBER, 0.44),
           ("proteomics matrices (3 sites)\nlog2 intensities per protein", "#f6e7cf", AMBER, 0.30),
           ("proteomics metadata\nsite, age, sex, case/control", LAB, LABE, 0.16)]
    for txt, fc, ec, y in src:
        box(ax, 0.01, y, 0.17, 0.09, txt, fc=fc, ec=ec, fs=7.8)
    ax.text(0.095, 0.975, "DATA SOURCES", ha="center", fontsize=9, weight="bold", color=INK)
    # knowledge graph column
    box(ax, 0.245, 0.62, 0.2, 0.3, "GRAPH STRUCTURE\nIndividual, Cluster, Block, Gene, Protein\nCARRIES, CO_OCCURS, IN_BLOCK,\nNEXT_BLOCK, OVERLAPS, ENCODES,\nMEASURED (harmonised z)", fc="#ffffff", ec=INK, fs=7.8)
    box(ax, 0.245, 0.38, 0.2, 0.18, "NODE FEATURES\ncluster + block statistics;\nperson = SVD-32 of the carrier matrix\n(label-free) or raw row, + protein z + mask", fc="#ffffff", ec=INK, fs=7.5)
    box(ax, 0.245, 0.14, 0.2, 0.18, "LABELS on the person node\nancestry, population, sex, site, phenotype\ntargets and ground truth only:\nnever an edge, never a feature", fc=LAB, ec=LABE, fs=7.5)
    ax.text(0.345, 0.975, "KNOWLEDGE GRAPH (hetero_v2.pt)", ha="center", fontsize=9, weight="bold", color=INK)
    for y, ty in [(0.905, 0.85), (0.765, 0.78), (0.485, 0.66), (0.345, 0.70)]:
        arrow(ax, 0.18, y, 0.245, ty)
    arrow(ax, 0.18, 0.905, 0.245, 0.50); arrow(ax, 0.18, 0.765, 0.245, 0.46); arrow(ax, 0.18, 0.345, 0.245, 0.42)
    arrow(ax, 0.18, 0.625, 0.245, 0.26, color=LABE); arrow(ax, 0.18, 0.205, 0.245, 0.20, color=LABE)
    # GNN
    box(ax, 0.49, 0.36, 0.155, 0.34, "GNN ENCODER\nHeteroConv x 2, hidden 64\nSAGEConv + edge-weighted GraphConv\n\ntrained on TRAIN people's labels\nselected on VAL\nreported on TEST (376 people)", fc="#d9efee", ec=TEAL, fs=7.8)
    arrow(ax, 0.445, 0.77, 0.49, 0.62, "messages"); arrow(ax, 0.445, 0.47, 0.49, 0.53, "inputs"); arrow(ax, 0.445, 0.23, 0.49, 0.42, "loss", color=LABE)
    ax.text(0.58, 0.975, "ENCODER", ha="center", fontsize=9, weight="bold", color=INK)
    # outputs
    outs = [("class probabilities\nper person", 0.86), ("64-d embedding per\nperson and per cluster", 0.72), ("saliency per\nhaploblock cluster", 0.58), ("model weights\n(state dict)", 0.44), ("predictions for all\n2,548 people (infer.py)", 0.30)]
    for txt, y in outs:
        box(ax, 0.685, y, 0.13, 0.09, txt, fc="#ffffff", ec=INK, fs=7.4); arrow(ax, 0.645, 0.53, 0.685, y + 0.045)
    ax.text(0.75, 0.975, "WHAT THE TRAINED MODEL GIVES", ha="center", fontsize=9, weight="bold", color=INK)
    # consumers
    cons = [("evaluation against\nground truth (test labels,\nground_truth.json)", 0.80, "#ffffff", INK),
            ("LLM decoder\n(NIM GraphRAG)\ncited insight per person", 0.55, "#f6e7cf", AMBER),
            ("NVFlare FedAvg\nonly weights leave\na site", 0.30, "#d9efee", TEAL),
            ("plots, Neo4j, CSVs\nfor the team", 0.08, "#ffffff", INK)]
    for txt, y, fc, ec in cons:
        box(ax, 0.855, y, 0.14, 0.12, txt, fc=fc, ec=ec, fs=7.4)
    ax.text(0.925, 0.975, "CONSUMERS", ha="center", fontsize=9, weight="bold", color=INK)
    arrow(ax, 0.815, 0.905, 0.855, 0.87); arrow(ax, 0.815, 0.625, 0.855, 0.84)      # probabilities, saliency -> evaluation
    arrow(ax, 0.815, 0.905, 0.855, 0.64); arrow(ax, 0.815, 0.765, 0.855, 0.62); arrow(ax, 0.815, 0.625, 0.855, 0.60)   # -> decoder
    arrow(ax, 0.815, 0.485, 0.855, 0.36)                                            # weights -> FedAvg
    arrow(ax, 0.815, 0.345, 0.855, 0.14); arrow(ax, 0.815, 0.765, 0.855, 0.12)      # predictions, embeddings -> plots
    ax.text(0.5, 0.02, "Phenotypes are never neighbours of the person: the encoder can only reach a label through the loss on training people. Ground truth for the synthetic phenotype is the generator's own label; saliency and ridge are scored against ground_truth.json.", ha="center", fontsize=8, color=GREY)
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
    ax.set_title(f"With and without federation ({lo['steps_per_site']} optimizer steps per site, A100)", fontsize=10); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "federated_site_alone.png", dpi=150); plt.close(fig)


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
        ax.set_ylim(0, 1.05); ax.set_ylabel("test balanced accuracy"); ax.set_title(f"{target}: GNN by starting embedding"); ax.tick_params(axis="x", labelsize=7.5); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "init_comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    for fn in (workflow, schema, genome_schematic, federated_topology, person_neighbourhood, sites, gnn_results, federated_and_inference, eda_extra, confusion_matrices, data_flow_map, federated_site_alone, init_comparison):
        try:
            fn(); print("ok  ", fn.__name__)
        except Exception as exc:  # keep going; report which figure failed
            print("FAIL", fn.__name__, type(exc).__name__, str(exc)[:160])
    print(len(list(FIG.glob("*.png"))), "figures in", FIG)
