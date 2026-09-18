"""Knowledge graph from the published 1000G HaploGraph (data.haploblocks.org).

Inputs (see fetch_data.sh):
  nodes.csv.gz                        one row per haploblock *cluster*, one 0/1 column per
                                      1000G individual: does this individual carry the cluster
  edges_lift_above_threshold.csv.gz   cluster-cluster co-occurrence edges (weight, lift >= 5)
  block_stats.tsv                     per-haploblock statistics (length, n_clusters, entropy, ...)
  phenotypes_real.csv                 long-format ancestry / population / sex labels (1000G panel)

The graph has three node types
  individual --carries-->  cluster --in_block--> block --next_block--> block
                           cluster <--co_occurs--> cluster
and the labels live on the individual nodes.  Only the *cluster* rows are
filtered (by carrier support); every individual and every block is kept.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

BLOCK_RE = re.compile(r"^(?P<chrom>chr[0-9XY]+)_(?P<start>\d+)-(?P<end>\d+)$")
CLUSTER_RE = re.compile(r"^(?P<block>chr[0-9XY]+_\d+-\d+)_cluster(?P<num>\d+)$")
PHENOTYPES = ("ancestry", "population", "sex")


# ----------------------------------------------------------------------------- ids
def parse_block_id(block_id: str) -> tuple[str, int, int]:
    m = BLOCK_RE.match(block_id)
    if not m:
        raise ValueError(f"not a block id: {block_id!r}")
    return m["chrom"], int(m["start"]), int(m["end"])


def parse_cluster_id(cluster_id: str) -> tuple[str, int]:
    m = CLUSTER_RE.match(cluster_id)
    if not m:
        raise ValueError(f"not a cluster id: {cluster_id!r}")
    return m["block"], int(m["num"])


# ------------------------------------------------------------------------- loading
def read_node_matrix(path: Path, chunksize: int = 8192):
    """Stream nodes.csv.gz into a CSR matrix of shape (clusters, individuals), int8.

    The file is ~1.3 GB uncompressed for chr22; reading it as int8 in chunks and
    converting each chunk to sparse keeps peak memory well under 1 GB.
    """
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
    individuals = header[2:]
    dtypes = {name: np.int8 for name in individuals}
    dtypes[header[0]] = str
    dtypes[header[1]] = str

    cluster_ids: list[str] = []
    block_ids: list[str] = []
    parts = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=chunksize, engine="c"):
        cluster_ids.extend(chunk.iloc[:, 0].tolist())
        block_ids.extend(chunk.iloc[:, 1].tolist())
        parts.append(sparse.csr_matrix(chunk.iloc[:, 2:].to_numpy(dtype=np.int8)))
    if parts:
        matrix = sparse.vstack(parts, format="csr")
    else:
        matrix = sparse.csr_matrix((0, len(individuals)), dtype=np.int8)
    return cluster_ids, block_ids, individuals, matrix


def load_phenotypes(path: Path) -> pd.DataFrame:
    """Long (individual_id, phenotype, value) -> wide, indexed by individual_id."""
    long = pd.read_csv(path, dtype=str)
    wide = long.pivot_table(index="individual_id", columns="phenotype", values="value", aggfunc="first")
    for name in PHENOTYPES:
        if name not in wide.columns:
            wide[name] = np.nan
    return wide[list(PHENOTYPES)]


def load_block_stats(path: Path, chrom: str) -> pd.DataFrame:
    stats = pd.read_csv(path, sep="\t")
    stats = stats[stats["chr"] == chrom].copy()
    stats = stats.sort_values("start").reset_index(drop=True)
    stats["block_idx"] = np.arange(len(stats))
    stats["singleton_rate"] = stats["singleton_count"] / stats["n_clusters"].clip(lower=1)
    return stats.rename(columns={"block": "block_id"})


def load_edges(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ------------------------------------------------------------------------ building
def encode_labels(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Sorted class codes; missing -> -1."""
    classes = sorted(v for v in values.dropna().unique())
    lookup = {c: i for i, c in enumerate(classes)}
    codes = np.array([lookup.get(v, -1) if isinstance(v, str) else -1 for v in values], dtype=np.int64)
    return codes, classes


def build_tables(
    cluster_ids: list[str],
    block_ids: list[str],
    individual_ids: list[str],
    matrix: sparse.csr_matrix,
    phenotypes: pd.DataFrame,
    block_stats: pd.DataFrame,
    edges: pd.DataFrame,
    min_support: int = 25,
    symmetric: bool = True,
) -> dict:
    """Filter clusters by carrier support and assemble every table of the graph.

    symmetric=True mirrors the HaploGraph edge filter: a cluster is kept only if
    min(carriers, non-carriers) >= min_support, so near-universal clusters go too.
    """
    n_ind = len(individual_ids)
    support = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    keep = support >= min_support
    if symmetric:
        keep &= (n_ind - support) >= min_support

    # blocks: everything in block_stats for this chromosome, plus any block that
    # appears in nodes.csv but is missing from block_stats (should not happen)
    blocks = block_stats.copy()
    known = set(blocks["block_id"])
    extra = sorted(set(block_ids) - known, key=lambda b: parse_block_id(b)[1])
    if extra:
        rows = []
        for b in extra:
            chrom, start, end = parse_block_id(b)
            rows.append({"chr": chrom, "block_id": b, "start": start, "end": end, "block_length": end - start})
        blocks = pd.concat([blocks, pd.DataFrame(rows)], ignore_index=True)
        blocks = blocks.sort_values("start").reset_index(drop=True)
        blocks["block_idx"] = np.arange(len(blocks))
    block_index = dict(zip(blocks["block_id"], blocks["block_idx"]))

    clusters = pd.DataFrame({"cluster_id": cluster_ids, "block_id": block_ids, "support": support})
    clusters["row"] = np.arange(len(clusters))
    clusters = clusters[keep].reset_index(drop=True)
    clusters["cluster_idx"] = np.arange(len(clusters))
    clusters["block_idx"] = clusters["block_id"].map(block_index).astype(np.int64)
    clusters["cluster_num"] = [parse_cluster_id(c)[1] for c in clusters["cluster_id"]]
    clusters["support_frac"] = clusters["support"] / n_ind

    carries = matrix[clusters["row"].to_numpy()].T.tocsr()  # individuals x kept clusters
    carries.data = np.ones_like(carries.data, dtype=np.int8)

    individuals = pd.DataFrame({"individual_id": individual_ids})
    individuals["individual_idx"] = np.arange(n_ind)
    individuals = individuals.join(phenotypes, on="individual_id")
    label_maps = {}
    for name in PHENOTYPES:
        codes, classes = encode_labels(individuals[name])
        individuals[f"{name}_code"] = codes
        label_maps[name] = classes

    cluster_index = dict(zip(clusters["cluster_id"], clusters["cluster_idx"]))
    co = edges.copy()
    co["src"] = co["source"].map(cluster_index)
    co["dst"] = co["target"].map(cluster_index)
    dropped = int(co["src"].isna().sum() + co["dst"].isna().sum() - (co["src"].isna() & co["dst"].isna()).sum())
    co = co.dropna(subset=["src", "dst"]).astype({"src": np.int64, "dst": np.int64})
    co_occurs = co[["src", "dst", "weight", "lift"]].reset_index(drop=True)

    order = blocks.sort_values("start")["block_idx"].to_numpy()
    next_block = pd.DataFrame({"src": order[:-1], "dst": order[1:]})

    return {
        "individuals": individuals,
        "clusters": clusters.drop(columns=["row"]),
        "blocks": blocks,
        "carries": carries,
        "co_occurs": co_occurs,
        "next_block": next_block,
        "label_maps": label_maps,
        "n_edges_dropped": dropped,
        "min_support": min_support,
        "symmetric": symmetric,
    }


# ---------------------------------------------------------------------- features
def _zscore(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=np.float64)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[std == 0] = 1.0
    z = (values - mean) / std
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


BLOCK_FEATURES = ["log_length", "shannon_entropy", "dominance", "log_n_clusters", "singleton_rate"]


def block_feature_frame(blocks: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=blocks.index)
    out["log_length"] = np.log10(blocks["block_length"].astype(float).clip(lower=1))
    out["shannon_entropy"] = blocks.get("shannon_entropy", np.nan)
    out["dominance"] = blocks.get("dominance", np.nan)
    out["log_n_clusters"] = np.log1p(blocks.get("n_clusters", np.nan).astype(float))
    out["singleton_rate"] = blocks.get("singleton_rate", np.nan)
    return out


def cluster_feature_frame(clusters: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    bf = block_feature_frame(blocks).set_index(blocks["block_idx"])
    out = pd.DataFrame(index=clusters.index)
    out["log_support"] = np.log1p(clusters["support"].astype(float))
    out["support_frac"] = clusters["support_frac"]
    joined = bf.reindex(clusters["block_idx"].to_numpy())
    for col in BLOCK_FEATURES:
        out[f"block_{col}"] = joined[col].to_numpy()
    return out


def to_hetero_data(tables: dict):
    """Assemble a torch_geometric HeteroData (imported lazily so the tables work without torch)."""
    import torch
    from torch_geometric.data import HeteroData

    ind, cl, bl = tables["individuals"], tables["clusters"], tables["blocks"]
    data = HeteroData()

    data["individual"].num_nodes = len(ind)
    data["individual"].individual_id = list(ind["individual_id"])
    for name in PHENOTYPES:
        data["individual"][f"y_{name}"] = torch.tensor(ind[f"{name}_code"].to_numpy(), dtype=torch.long)

    data["cluster"].x = torch.from_numpy(_zscore(cluster_feature_frame(cl, bl)))
    data["cluster"].cluster_id = list(cl["cluster_id"])
    data["cluster"].support = torch.tensor(cl["support"].to_numpy(), dtype=torch.long)
    data["block"].x = torch.from_numpy(_zscore(block_feature_frame(bl)))
    data["block"].block_id = list(bl["block_id"])

    coo = tables["carries"].tocoo()
    carries = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    data["individual", "carries", "cluster"].edge_index = carries
    data["cluster", "rev_carries", "individual"].edge_index = carries.flip(0)

    in_block = torch.from_numpy(np.vstack([cl["cluster_idx"].to_numpy(), cl["block_idx"].to_numpy()]).astype(np.int64))
    data["cluster", "in_block", "block"].edge_index = in_block
    data["block", "rev_in_block", "cluster"].edge_index = in_block.flip(0)

    co = tables["co_occurs"]
    src = torch.tensor(co["src"].to_numpy(), dtype=torch.long)
    dst = torch.tensor(co["dst"].to_numpy(), dtype=torch.long)
    attr = torch.tensor(co[["weight", "lift"]].to_numpy(dtype=np.float32))
    data["cluster", "co_occurs", "cluster"].edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    data["cluster", "co_occurs", "cluster"].edge_attr = torch.cat([attr, attr])

    nb = tables["next_block"]
    a = torch.tensor(nb["src"].to_numpy(), dtype=torch.long)
    b = torch.tensor(nb["dst"].to_numpy(), dtype=torch.long)
    data["block", "next_block", "block"].edge_index = torch.stack([torch.cat([a, b]), torch.cat([b, a])])

    data.label_maps = tables["label_maps"]
    return data


# -------------------------------------------------------------------------- saving
def save_tables(tables: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables["individuals"].to_csv(out_dir / "individuals.csv", index=False)
    tables["clusters"].to_csv(out_dir / "clusters.csv", index=False)
    tables["blocks"].to_csv(out_dir / "blocks.csv", index=False)
    tables["co_occurs"].to_csv(out_dir / "co_occurs.csv", index=False)
    tables["next_block"].to_csv(out_dir / "next_block.csv", index=False)
    sparse.save_npz(out_dir / "carries.npz", tables["carries"])
    (out_dir / "label_maps.json").write_text(json.dumps(tables["label_maps"], indent=2))
    ind = tables["individuals"]
    summary = {
        "n_individuals": int(len(ind)),
        "n_individuals_labelled": int((ind["ancestry_code"] >= 0).sum()),
        "n_clusters_kept": int(len(tables["clusters"])),
        "n_blocks": int(len(tables["blocks"])),
        "n_carries_edges": int(tables["carries"].nnz),
        "n_co_occurs_edges": int(len(tables["co_occurs"])),
        "n_co_occurs_dropped_by_filter": int(tables["n_edges_dropped"]),
        "min_support": tables["min_support"],
        "symmetric_filter": tables["symmetric"],
        "clusters_per_individual_mean": float(tables["carries"].sum(axis=1).mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------------------- reloading
def load_kg(kg_dir: Path) -> dict:
    """Reload the tables written by save_tables (no torch needed)."""
    kg_dir = Path(kg_dir)
    return {
        "individuals": pd.read_csv(kg_dir / "individuals.csv"),
        "clusters": pd.read_csv(kg_dir / "clusters.csv"),
        "blocks": pd.read_csv(kg_dir / "blocks.csv"),
        "co_occurs": pd.read_csv(kg_dir / "co_occurs.csv"),
        "next_block": pd.read_csv(kg_dir / "next_block.csv"),
        "carries": sparse.load_npz(kg_dir / "carries.npz").tocsr(),
        "label_maps": json.loads((kg_dir / "label_maps.json").read_text()),
    }


def stratified_split(codes: np.ndarray, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> np.ndarray:
    """Return an array of 'train' / 'val' / 'test' / 'unlabelled' per individual.

    Stratified on `codes` (use ancestry); individuals with code -1 are never used
    for training or evaluation.  Deterministic for a given seed so the baseline
    and the GNN score the very same held-out people.
    """
    from sklearn.model_selection import train_test_split

    codes = np.asarray(codes)
    split = np.full(len(codes), "unlabelled", dtype=object)
    labelled = np.flatnonzero(codes >= 0)
    hold = val_frac + test_frac
    train_idx, hold_idx = train_test_split(labelled, test_size=hold, random_state=seed, stratify=codes[labelled])
    val_idx, test_idx = train_test_split(hold_idx, test_size=test_frac / hold, random_state=seed, stratify=codes[hold_idx])
    split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
    return split


def load_or_make_split(kg: dict, split_path: Path, seed: int = 42) -> np.ndarray:
    split_path = Path(split_path)
    if split_path.exists():
        frame = pd.read_csv(split_path)
        assert (frame["individual_id"].to_numpy() == kg["individuals"]["individual_id"].to_numpy()).all()
        return frame["split"].to_numpy()
    split = stratified_split(kg["individuals"]["ancestry_code"].to_numpy(), seed=seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"individual_id": kg["individuals"]["individual_id"], "split": split}).to_csv(split_path, index=False)
    return split
