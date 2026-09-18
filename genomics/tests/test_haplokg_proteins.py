"""Protein layer on top of the toy graph from test_haplokg."""
import numpy as np
import pandas as pd
import pytest

import haplokg
import haplokg_proteins as hp
from test_haplokg import toy, _tables  # noqa: F401  (fixture reuse)


@pytest.fixture
def protein_files(tmp_path):
    bed = tmp_path / "uniprot.bed"
    pd.DataFrame([
        ["chr22", 120, 180, "P00001-1", 1000, "+"],   # gene GA, inside block chr22_100-200
        ["chr22", 120, 190, "P00001-2", 0, "+"],
        ["chr22", 190, 260, "P00002", 1000, "-"],     # gene GB, spans both blocks
        ["chr22", 900, 950, "P00003", 1000, "+"],     # gene GC, in no block
    ]).to_csv(bed, sep="\t", header=False, index=False)
    symbols = tmp_path / "symbols.csv"
    pd.DataFrame({"protein_id": ["P00001", "P00002", "P00003"], "gene_symbol": ["GA", "GB", "GC"]}).to_csv(symbols, index=False)
    measured = tmp_path / "measured.csv"
    pd.DataFrame({
        "individual_id": ["I1", "I1", "I2", "I2", "I3", "ZZ"],
        "protein_id":    ["P00001", "P00002", "P00001", "P00002", "P00001", "P00001"],
        "log2_intensity": [10.0, 8.0, 12.0, 8.5, 11.0, 5.0],
        "site":          ["S1", "S1", "S1", "S1", "S2", "S2"],
    }).to_csv(measured, index=False)
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"sample_id": ["I1", "I2", "I3"], "site": ["S1", "S1", "S2"], "age": [30, 40, 50], "sex": [1, 0, 0],
                  "phenotype": [1, 0, 1]}).to_csv(meta, index=False)
    return dict(bed=bed, symbols=symbols, measured=measured, meta=meta)


def test_protein_tables(toy, protein_files):
    kg = _tables(toy, min_support=2)
    t = hp.build_protein_tables(kg, protein_files["bed"], protein_files["measured"], protein_files["meta"], protein_files["symbols"])
    assert t["proteins"]["protein_id"].tolist() == ["P00001", "P00002", "P00003"]
    assert t["proteins"].set_index("protein_id").loc["P00001", "n_isoforms"] == 2
    assert t["genes"]["gene_symbol"].tolist() == ["GA", "GB", "GC"]
    bg = t["block_gene"].sort_values(["gene_idx", "block_idx"])[["block_idx", "gene_idx"]].values.tolist()
    assert bg == [[0, 0], [0, 1], [1, 1]]                       # GA in block0; GB overlaps block0 and block1; GC nowhere
    assert t["n_unmatched_measurements"] == 1                   # ZZ is not in the graph
    assert len(t["measured"]) == 5
    z = t["measured"].set_index(["individual_idx", "protein_idx"])["z"]
    # harmonisation runs per site over *all* the site's samples (ZZ included, even though ZZ is not in the graph):
    # S2/P00001 = {11.0 (I3), 5.0 (ZZ)} -> median 8, MAD 3 -> z(I3) = 3 / (3 * 1.4826)
    assert abs(z.loc[(2, 0)] - 3 / (3 * 1.4826)) < 1e-6
    ind = t["individuals"].set_index("individual_id")
    assert ind.loc["I1", "phenotype_code"] == 1 and ind.loc["I4", "phenotype_code"] == -1
    assert t["label_maps"]["site"] == ["S1", "S2"] and ind.loc["I3", "site_code"] == 1
    assert t["abundance"].shape == (4, 3) and t["observed"].sum() == 5


def test_extend_hetero_data(toy, protein_files):
    torch = pytest.importorskip("torch")
    kg = _tables(toy, min_support=2)
    t = hp.build_protein_tables(kg, protein_files["bed"], protein_files["measured"], protein_files["meta"], protein_files["symbols"])
    data = hp.extend_hetero_data(haplokg.to_hetero_data(kg), t)
    assert data["gene"].x.shape == (3, 2) and data["protein"].x.shape == (3, 2)
    assert data["block", "overlaps", "gene"].edge_index.shape == (2, 3)
    assert data["gene", "encodes", "protein"].edge_index.shape == (2, 3)
    assert data["individual", "measured", "protein"].edge_index.shape == (2, 5)
    assert data["individual", "measured", "protein"].edge_attr.shape == (5, 2)
    assert data["individual"].y_phenotype.tolist() == [1, 0, 1, -1]
    data.validate()
