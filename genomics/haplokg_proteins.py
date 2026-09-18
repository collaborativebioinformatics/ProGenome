"""Protein layer of the knowledge graph (schema v2).

Adds to the genome graph built by haplokg:
    Block -OVERLAPS-> Gene -ENCODES-> Protein            (proteomics/uniprot_chr22.bed + gene symbols)
    Individual -MEASURED{log2, z}-> Protein             (a proteomics matrix keyed by 1000G IDs)
and per-individual phenotype / site / age columns from the proteomics sample metadata.

The harmoniser: MEASURED carries the raw log2 intensity and a z-score computed
*within each site and protein* (median / MAD), which removes per-site batch offsets
before anything crosses sites - the simplest version of the whiteboard's
"NORM: HARMONIZER" box.  Detection-limit missingness stays missing (no edge).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def load_protein_bed(bed_path: Path, symbols_path: Path | None = None) -> pd.DataFrame:
    """One row per protein (isoforms collapsed to the base accession) with its genomic span and gene symbol."""
    bed = pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2, 3, 5], names=["chrom", "start", "end", "uniprot", "strand"])
    bed["protein_id"] = bed["uniprot"].str.split("-").str[0]
    prot = bed.groupby(["chrom", "protein_id"], as_index=False).agg(
        start=("start", "min"), end=("end", "max"), strand=("strand", "first"), n_isoforms=("uniprot", "nunique"))
    symbols = {}
    if symbols_path and Path(symbols_path).exists():
        symbols = pd.read_csv(symbols_path).set_index("protein_id")["gene_symbol"].to_dict()
    prot["gene_symbol"] = [symbols.get(p, p) for p in prot["protein_id"]]
    return prot


def harmonise(long: pd.DataFrame) -> pd.DataFrame:
    """Robust z-score per (site, protein): (x - median) / (1.4826 * MAD)."""
    grp = long.groupby(["site", "protein_id"])["log2_intensity"]
    med = grp.transform("median")
    mad = (long["log2_intensity"] - med).abs().groupby([long["site"], long["protein_id"]]).transform("median") * 1.4826
    long = long.copy()
    long["z"] = ((long["log2_intensity"] - med) / mad.replace(0, np.nan)).fillna(0.0)
    return long


def build_protein_tables(kg: dict, bed_path: Path, measured_path: Path, metadata_path: Path,
                         symbols_path: Path | None = None) -> dict:
    ind, blocks = kg["individuals"], kg["blocks"]
    chrom = blocks["chr"].iloc[0]
    prot = load_protein_bed(bed_path, symbols_path)
    prot = prot[prot["chrom"] == chrom].reset_index(drop=True)

    genes = prot.groupby("gene_symbol", as_index=False).agg(start=("start", "min"), end=("end", "max"), n_proteins=("protein_id", "nunique"))
    genes["gene_idx"] = np.arange(len(genes))
    gene_index = dict(zip(genes["gene_symbol"], genes["gene_idx"]))
    prot["protein_idx"] = np.arange(len(prot))
    prot["gene_idx"] = prot["gene_symbol"].map(gene_index)

    # Block -OVERLAPS-> Gene by coordinate intersection (BED is 0-based half-open; blocks are 1-based inclusive)
    b = blocks[["block_idx", "start", "end"]].to_numpy()
    rows = []
    for g in genes.itertuples(index=False):
        hit = b[(b[:, 1] <= g.end) & (b[:, 2] >= g.start + 1)]
        for blk_idx, bs, be in hit:
            rows.append({"block_idx": int(blk_idx), "gene_idx": int(g.gene_idx),
                         "overlap_bp": int(min(be, g.end) - max(bs, g.start + 1) + 1)})
    block_gene = pd.DataFrame(rows, columns=["block_idx", "gene_idx", "overlap_bp"])
    gene_protein = prot[["gene_idx", "protein_idx"]].copy()

    # Individual -MEASURED-> Protein
    long = pd.read_csv(measured_path)
    long = long[long["protein_id"].isin(prot["protein_id"])]
    long = harmonise(long)
    ind_index = dict(zip(ind["individual_id"], ind["individual_idx"]))
    prot_index = dict(zip(prot["protein_id"], prot["protein_idx"]))
    long["individual_idx"] = long["individual_id"].map(ind_index)
    long["protein_idx"] = long["protein_id"].map(prot_index)
    unmatched = int(long["individual_idx"].isna().sum())
    long = long.dropna(subset=["individual_idx"]).astype({"individual_idx": np.int64, "protein_idx": np.int64})
    measured = long[["individual_idx", "protein_idx", "site", "log2_intensity", "z"]].reset_index(drop=True)

    # dense harmonised abundance matrix (individuals x proteins) + observed mask, for node features
    abundance = sparse.coo_matrix((measured["z"].to_numpy(), (measured["individual_idx"], measured["protein_idx"])),
                                  shape=(len(ind), len(prot))).tocsr()
    observed = sparse.coo_matrix((np.ones(len(measured), dtype=np.int8), (measured["individual_idx"], measured["protein_idx"])),
                                 shape=(len(ind), len(prot))).tocsr()

    # phenotype / site / age onto individuals
    meta = pd.read_csv(metadata_path).rename(columns={"sample_id": "individual_id"})
    ind2 = ind.merge(meta[["individual_id", "site", "age", "phenotype"]], on="individual_id", how="left")
    ind2["phenotype_code"] = ind2["phenotype"].fillna(-1).astype(np.int64)
    sites = sorted(meta["site"].dropna().unique())
    ind2["site_code"] = ind2["site"].map({s: i for i, s in enumerate(sites)}).fillna(-1).astype(np.int64)
    label_maps = dict(kg["label_maps"])
    label_maps["phenotype"] = ["control", "case"]
    label_maps["site"] = sites

    return {"individuals": ind2, "genes": genes, "proteins": prot, "block_gene": block_gene, "gene_protein": gene_protein,
            "measured": measured, "abundance": abundance, "observed": observed, "label_maps": label_maps,
            "n_unmatched_measurements": unmatched}


def save_protein_tables(t: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    t["individuals"].to_csv(out_dir / "individuals_v2.csv", index=False)
    t["genes"].to_csv(out_dir / "genes.csv", index=False)
    t["proteins"].to_csv(out_dir / "proteins.csv", index=False)
    t["block_gene"].to_csv(out_dir / "block_gene.csv", index=False)
    t["gene_protein"].to_csv(out_dir / "gene_protein.csv", index=False)
    t["measured"].to_csv(out_dir / "measured.csv", index=False)
    sparse.save_npz(out_dir / "abundance_z.npz", t["abundance"])
    sparse.save_npz(out_dir / "abundance_observed.npz", t["observed"])
    (out_dir / "label_maps_v2.json").write_text(json.dumps(t["label_maps"], indent=2))
    summary = {
        "n_genes": int(len(t["genes"])), "n_proteins": int(len(t["proteins"])),
        "n_block_gene_edges": int(len(t["block_gene"])), "n_measured_edges": int(len(t["measured"])),
        "n_individuals_with_proteomics": int((t["individuals"]["site_code"] >= 0).sum()),
        "n_unmatched_measurements": int(t["n_unmatched_measurements"]),
        "genes_without_block": int((~t["genes"]["gene_idx"].isin(t["block_gene"]["gene_idx"])).sum()),
    }
    (out_dir / "summary_v2.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_protein_tables(kg_dir: Path) -> dict:
    kg_dir = Path(kg_dir)
    return {
        "individuals": pd.read_csv(kg_dir / "individuals_v2.csv"),
        "genes": pd.read_csv(kg_dir / "genes.csv"), "proteins": pd.read_csv(kg_dir / "proteins.csv"),
        "block_gene": pd.read_csv(kg_dir / "block_gene.csv"), "gene_protein": pd.read_csv(kg_dir / "gene_protein.csv"),
        "measured": pd.read_csv(kg_dir / "measured.csv"),
        "abundance": sparse.load_npz(kg_dir / "abundance_z.npz").tocsr(),
        "observed": sparse.load_npz(kg_dir / "abundance_observed.npz").tocsr(),
        "label_maps": json.loads((kg_dir / "label_maps_v2.json").read_text()),
    }


def extend_hetero_data(data, t: dict):
    """Add gene/protein nodes and their edges to an existing HeteroData (in place) and return it."""
    import torch

    genes, prot = t["genes"], t["proteins"]
    data["individual"].y_phenotype = torch.tensor(t["individuals"]["phenotype_code"].to_numpy(), dtype=torch.long)
    data["individual"].y_site = torch.tensor(t["individuals"]["site_code"].to_numpy(), dtype=torch.long)
    gx = np.stack([np.log10(genes["end"] - genes["start"] + 1), np.log1p(genes["n_proteins"])], axis=1).astype(np.float32)
    px = np.stack([np.log10(prot["end"] - prot["start"] + 1), np.log1p(prot["n_isoforms"])], axis=1).astype(np.float32)
    data["gene"].x = torch.from_numpy((gx - gx.mean(0)) / (gx.std(0) + 1e-6))
    data["gene"].gene_symbol = list(genes["gene_symbol"])
    data["protein"].x = torch.from_numpy((px - px.mean(0)) / (px.std(0) + 1e-6))
    data["protein"].protein_id = list(prot["protein_id"])

    bg = torch.tensor(t["block_gene"][["block_idx", "gene_idx"]].to_numpy().T, dtype=torch.long)
    data["block", "overlaps", "gene"].edge_index = bg
    data["gene", "rev_overlaps", "block"].edge_index = bg.flip(0)
    gp = torch.tensor(t["gene_protein"][["gene_idx", "protein_idx"]].to_numpy().T, dtype=torch.long)
    data["gene", "encodes", "protein"].edge_index = gp
    data["protein", "rev_encodes", "gene"].edge_index = gp.flip(0)
    m = t["measured"]
    mi = torch.tensor(m[["individual_idx", "protein_idx"]].to_numpy().T, dtype=torch.long)
    attr = torch.tensor(m[["z", "log2_intensity"]].to_numpy(dtype=np.float32))
    data["individual", "measured", "protein"].edge_index = mi
    data["individual", "measured", "protein"].edge_attr = attr
    data["protein", "rev_measured", "individual"].edge_index = mi.flip(0)
    data["protein", "rev_measured", "individual"].edge_attr = attr
    data.label_maps = t["label_maps"]
    return data
