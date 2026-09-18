"""Unit tests for haplokg on a 4-individual, 5-cluster, 2-block toy graph."""
import gzip

import numpy as np
import pandas as pd
import pytest

import haplokg

INDIVIDUALS = ["I1", "I2", "I3", "I4"]
NODE_ROWS = [
    # cluster_id,                block_id,        I1 I2 I3 I4   support
    ("chr22_100-200_cluster1", "chr22_100-200", [1, 1, 0, 0]),  # 2  keep
    ("chr22_100-200_cluster2", "chr22_100-200", [0, 0, 1, 1]),  # 2  keep
    ("chr22_100-200_cluster3", "chr22_100-200", [0, 0, 0, 1]),  # 1  singleton -> drop
    ("chr22_200-300_cluster1", "chr22_200-300", [1, 1, 1, 1]),  # 4  universal -> drop (symmetric)
    ("chr22_200-300_cluster2", "chr22_200-300", [1, 0, 1, 0]),  # 2  keep
]


@pytest.fixture
def toy(tmp_path):
    nodes = tmp_path / "nodes.csv.gz"
    with gzip.open(nodes, "wt") as fh:
        fh.write("id,high_dim_edge," + ",".join(INDIVIDUALS) + "\n")
        for cid, bid, bits in NODE_ROWS:
            fh.write(f"{cid},{bid}," + ",".join(map(str, bits)) + "\n")

    stats = tmp_path / "block_stats.tsv"
    pd.DataFrame(
        {
            "chr": ["chr22", "chr22", "chr1"],
            "block": ["chr22_200-300", "chr22_100-200", "chr1_5-9"],
            "start": [200, 100, 5],
            "end": [300, 200, 9],
            "block_length": [100, 100, 4],
            "n_haplotypes": [8, 8, 8],
            "n_clusters": [2, 3, 1],
            "max_cluster_size": [4, 2, 1],
            "singleton_count": [0, 1, 1],
            "dominance": [0.5, 0.25, 1.0],
            "shannon_entropy": [1.0, 1.5, 0.0],
        }
    ).to_csv(stats, sep="\t", index=False)

    pheno = tmp_path / "phenotypes_real.csv"
    rows = []
    for ind, anc, pop, sex in [("I1", "EUR", "GBR", "male"), ("I2", "AFR", "YRI", "female"), ("I3", "EUR", "FIN", "female")]:
        rows += [(ind, "ancestry", anc), (ind, "population", pop), (ind, "sex", sex)]
    pd.DataFrame(rows, columns=["individual_id", "phenotype", "value"]).assign(source="test").to_csv(pheno, index=False)

    edges = tmp_path / "edges.csv.gz"
    pd.DataFrame(
        {
            "source": ["chr22_100-200_cluster1", "chr22_100-200_cluster3"],
            "target": ["chr22_200-300_cluster2", "chr22_200-300_cluster2"],
            "weight": [1, 1],
            "lift": [2.0, 4.0],
        }
    ).to_csv(edges, index=False)
    return dict(nodes=nodes, stats=stats, pheno=pheno, edges=edges)


def test_parse_ids():
    assert haplokg.parse_block_id("chr22_17099658-17118145") == ("chr22", 17099658, 17118145)
    assert haplokg.parse_cluster_id("chr22_17099658-17118145_cluster219") == ("chr22_17099658-17118145", 219)
    with pytest.raises(ValueError):
        haplokg.parse_cluster_id("chr22_1-2")


def test_read_node_matrix_streams_to_sparse(toy):
    ids, blocks, individuals, matrix = haplokg.read_node_matrix(toy["nodes"], chunksize=2)
    assert individuals == INDIVIDUALS
    assert ids == [r[0] for r in NODE_ROWS]
    assert blocks == [r[1] for r in NODE_ROWS]
    assert matrix.shape == (5, 4) and matrix.dtype == np.int8
    assert np.asarray(matrix.sum(axis=1)).ravel().tolist() == [2, 2, 1, 4, 2]


def _tables(toy, **kw):
    ids, blocks, individuals, matrix = haplokg.read_node_matrix(toy["nodes"])
    return haplokg.build_tables(
        ids, blocks, individuals, matrix,
        haplokg.load_phenotypes(toy["pheno"]),
        haplokg.load_block_stats(toy["stats"], "chr22"),
        haplokg.load_edges(toy["edges"]),
        **kw,
    )


def test_symmetric_support_filter_drops_singletons_and_universal(toy):
    t = _tables(toy, min_support=2, symmetric=True)
    assert t["clusters"]["cluster_id"].tolist() == [
        "chr22_100-200_cluster1", "chr22_100-200_cluster2", "chr22_200-300_cluster2",
    ]
    assert t["carries"].shape == (4, 3)               # individuals x kept clusters
    assert t["carries"].sum() == 6                     # 2 + 2 + 2 carriers
    assert t["carries"][0].toarray().ravel().tolist() == [1, 0, 1]   # I1 carries b1c1 and b2c2


def test_non_symmetric_filter_keeps_universal(toy):
    t = _tables(toy, min_support=2, symmetric=False)
    assert "chr22_200-300_cluster1" in t["clusters"]["cluster_id"].tolist()


def test_blocks_sorted_by_position_and_linked(toy):
    t = _tables(toy, min_support=2)
    assert t["blocks"]["block_id"].tolist() == ["chr22_100-200", "chr22_200-300"]   # chr1 row filtered out
    assert t["next_block"].to_dict("records") == [{"src": 0, "dst": 1}]
    assert t["clusters"]["block_idx"].tolist() == [0, 0, 1]


def test_labels_encoded_with_missing_as_minus_one(toy):
    t = _tables(toy, min_support=2)
    ind = t["individuals"].set_index("individual_id")
    assert t["label_maps"]["ancestry"] == ["AFR", "EUR"]
    assert ind.loc["I1", "ancestry_code"] == 1 and ind.loc["I2", "ancestry_code"] == 0
    assert ind.loc["I4", "ancestry_code"] == -1 and ind.loc["I4", "sex_code"] == -1
    assert t["label_maps"]["sex"] == ["female", "male"]


def test_edges_remapped_and_filtered_endpoints_dropped(toy):
    t = _tables(toy, min_support=2)
    assert t["co_occurs"].to_dict("records") == [{"src": 0, "dst": 2, "weight": 1, "lift": 2.0}]
    assert t["n_edges_dropped"] == 1


def test_hetero_data_shapes(toy):
    torch = pytest.importorskip("torch")
    t = _tables(toy, min_support=2)
    data = haplokg.to_hetero_data(t)
    assert data["individual"].num_nodes == 4
    assert data["cluster"].x.shape == (3, 2 + len(haplokg.BLOCK_FEATURES))
    assert data["block"].x.shape == (2, len(haplokg.BLOCK_FEATURES))
    assert data["individual", "carries", "cluster"].edge_index.shape == (2, 6)
    assert data["cluster", "rev_carries", "individual"].edge_index.shape == (2, 6)
    assert data["cluster", "co_occurs", "cluster"].edge_index.shape == (2, 2)      # both directions
    assert data["cluster", "co_occurs", "cluster"].edge_attr.shape == (2, 2)
    assert data["block", "next_block", "block"].edge_index.shape == (2, 2)
    assert data["individual"].y_ancestry.tolist() == [1, 0, 1, -1]
    assert not torch.isnan(data["cluster"].x).any()
    data.validate()


def test_save_tables_roundtrip(toy, tmp_path):
    t = _tables(toy, min_support=2)
    summary = haplokg.save_tables(t, tmp_path / "kg")
    assert summary["n_clusters_kept"] == 3 and summary["n_individuals_labelled"] == 3
    reloaded = pd.read_csv(tmp_path / "kg" / "clusters.csv")
    assert len(reloaded) == 3
