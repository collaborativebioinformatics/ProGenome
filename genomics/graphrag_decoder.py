#!/usr/bin/env python3
"""GraphRAG decoder: knowledge-graph neighbourhood + GNN outputs -> LLM -> cited insight.

    python graphrag_decoder.py --individual HG00096 --dry-run          # print the prompt, no LLM call
    python graphrag_decoder.py --individual HG00096                    # calls the NVIDIA NIM endpoint
    python graphrag_decoder.py --individual HG00096 --run phenotype_both_raw

The GNN encodes; the LLM decodes.  Retrieval is deterministic and comes only from the graph:
  * who the person is (ancestry / population / sex / site / age, from the graph; the true phenotype is withheld)
  * the GNN's prediction for them (test_predictions.csv of the chosen run) and globally salient clusters
  * the clusters they carry that are most ancestry-informative, with block, genes and proteins in that block
  * their most extreme harmonised protein levels, with the gene and whether that gene sits in a block where
    they carry a notable cluster (the genome<->proteome link)
  * their nearest neighbours in the GNN embedding space
The LLM (OpenAI-compatible NIM API; key, model and endpoint come from config.py: NVIDIA_API_KEY, NIM_MODEL, NIM_URL in
genomics/.env or ~/.progenome.env, see .env.example) must answer as JSON and may only cite ids that appear in the
context; citations are validated before anything is written.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import haplokg
import haplokg_proteins as hp

import config  # credentials and endpoints: .env / ~/.progenome.env / defaults (see .env.example)

NIM_URL = config.settings.nim_url            # any OpenAI-compatible chat-completions endpoint
DEFAULT_MODEL = config.settings.nim_model

SYSTEM = """You are a careful genomics analyst writing for a clinical research team.
You receive a structured context extracted from a knowledge graph (haploblock clusters, blocks, genes, proteins,
a graph-neural-network prediction and similar individuals). Rules:
1. Use only the context. Do not invent genes, proteins, variants, diseases or numbers.
2. Every claim that refers to a graph entity must cite its id exactly as written (cluster ids like
   chr22_17099658-17118145_cluster1, protein ids like P14174, gene symbols as given).
3. Ancestry in this graph is population structure, not a medical finding. The phenotype is a synthetic case/control
   label used to test the pipeline; say so.
4. Answer as a single JSON object with keys: summary (2-3 sentences), ancestry_assessment, phenotype_assessment,
   genome_proteome_links (list of {cluster_id, block_id, gene, protein_id, observation}), caveats (list of strings),
   cited_ids (list of every id you cited). No markdown fences, no text outside the JSON."""


def load_everything(here: Path, chrom: str, run: str):
    kg_dir = here / "outputs" / "kg" / chrom
    kg = haplokg.load_kg(kg_dir)
    pt = hp.load_protein_tables(kg_dir)
    assoc = pd.read_csv(here / "outputs" / "cooccurrence" / chrom / "cluster_phenotype_association.csv")
    run_dirs = [here / "outputs" / "gnn_v2" / chrom / run, here / "outputs_brev" / "progenome-a100" / "gnn_v2" / chrom / run]
    run_dir = next((d for d in run_dirs if (d / "metrics.json").exists()), None)
    if run_dir is None:
        raise SystemExit(f"no GNN run named {run!r} under outputs/gnn_v2 or outputs_brev/progenome-a100/gnn_v2")
    preds = pd.read_csv(run_dir / "test_predictions.csv").set_index("individual_id")
    emb = np.load(run_dir / "embedding_individual.npy")
    saliency = pd.read_csv(run_dir / "saliency_top100.csv") if (run_dir / "saliency_top100.csv").exists() else None
    return kg, pt, assoc, run_dir, preds, emb, saliency


def retrieve(individual: str, kg, pt, assoc, preds, emb, saliency, k_clusters=8, k_proteins=8, k_neighbours=5) -> dict:
    ind = pt["individuals"].set_index("individual_id")
    if individual not in ind.index:
        raise SystemExit(f"{individual} is not in the graph")
    row = ind.loc[individual]
    i = int(row["individual_idx"])
    clusters, blocks, genes, prot = kg["clusters"], kg["blocks"], pt["genes"], pt["proteins"]
    block_gene = pt["block_gene"].merge(genes[["gene_idx", "gene_symbol"]], on="gene_idx")
    gene_prot = pt["gene_protein"].merge(prot[["protein_idx", "protein_id"]], on="protein_idx")
    gene_to_prot = gene_prot.groupby("gene_idx")["protein_id"].apply(list).to_dict()
    block_to_genes = block_gene.groupby("block_idx")["gene_symbol"].apply(list).to_dict()
    block_to_gene_idx = block_gene.groupby("block_idx")["gene_idx"].apply(list).to_dict()

    carried = kg["carries"][i].indices
    a = assoc.set_index("cluster_idx")
    top = a.loc[carried].sort_values("ancestry_cramers_v", ascending=False).head(k_clusters)
    cluster_ctx = []
    for cidx, r in top.iterrows():
        blk = int(r["block_idx"]); b = blocks.loc[blocks["block_idx"] == blk].iloc[0]
        cluster_ctx.append({
            "cluster_id": r["cluster_id"], "block_id": b["block_id"], "carriers": int(r["support"]),
            "enriched_in": r["ancestry_dominant"], "cramers_v": round(float(r["ancestry_cramers_v"]), 3),
            "carrier_fraction_by_ancestry": {k.replace("carrier_frac_", ""): round(float(r[k]), 3) for k in top.columns if k.startswith("carrier_frac_")},
            "genes_in_block": block_to_genes.get(blk, []),
            "proteins_in_block": sorted({p for g in block_to_gene_idx.get(blk, []) for p in gene_to_prot.get(g, [])}),
        })
    notable_blocks = {c["block_id"] for c in cluster_ctx}

    m = pt["measured"]
    mine = m[m["individual_idx"] == i].copy()
    mine["abs_z"] = mine["z"].abs()
    mine = mine.sort_values("abs_z", ascending=False).head(k_proteins)
    prot_by_idx = prot.set_index("protein_idx")
    protein_ctx = []
    for _, r in mine.iterrows():
        p = prot_by_idx.loc[int(r["protein_idx"])]
        gblocks = block_gene.loc[block_gene["gene_idx"] == p["gene_idx"], "block_idx"].tolist()
        gblock_ids = [blocks.loc[blocks["block_idx"] == b, "block_id"].iloc[0] for b in gblocks]
        protein_ctx.append({"protein_id": p["protein_id"], "gene": p["gene_symbol"], "harmonised_z": round(float(r["z"]), 2),
                            "log2_intensity": round(float(r["log2_intensity"]), 2), "site": r["site"],
                            "encoded_in_blocks": gblock_ids,
                            "in_a_block_with_a_notable_cluster": any(b in notable_blocks for b in gblock_ids)})

    labelled = ind[ind["ancestry_code"] >= 0]
    others = labelled.index[labelled.index != individual]
    d = np.linalg.norm(emb[labelled.loc[others, "individual_idx"].to_numpy()] - emb[i], axis=1)
    nn = np.argsort(d)[:k_neighbours]
    neighbours = [{"individual_id": others[j], "distance": round(float(d[j]), 3), "ancestry": labelled.loc[others[j], "ancestry"],
                   "population": labelled.loc[others[j], "population"],
                   "predicted_phenotype": preds.loc[others[j], "pred"] if others[j] in preds.index else "n/a"} for j in nn]

    prediction = ({"predicted": preds.loc[individual, "pred"]} if individual in preds.index
                  else {"predicted": "n/a (not in the held-out test split)"})
    global_saliency = saliency.head(10)[["cluster_id", "saliency"]].to_dict("records") if saliency is not None else []
    carried_salient = [s for s in global_saliency if s["cluster_id"] in set(clusters.loc[carried, "cluster_id"])]

    return {
        "individual": {"id": individual, "ancestry": row["ancestry"], "population": row["population"], "sex": row["sex"],
                       "site": row.get("site", "n/a"), "age": int(row["age"]) if pd.notna(row.get("age", np.nan)) else "n/a",
                       "n_clusters_carried": int(len(carried))},
        "gnn_phenotype_prediction": prediction,
        "globally_salient_clusters_this_person_carries": carried_salient,
        "most_ancestry_informative_clusters_carried": cluster_ctx,
        "most_extreme_protein_levels": protein_ctx,
        "nearest_individuals_in_gnn_embedding": neighbours,
        "note": "phenotype is a synthetic case/control label with a saved ground truth; ancestry labels are real 1000G panel data",
    }


def call_nim(context: dict, model: str, temperature: float = 0.2, max_tokens: int = 4000, reasoning: str = "none") -> tuple[str, dict]:
    import requests
    key = config.require("NVIDIA_API_KEY")
    # Nemotron 3 is a reasoning model: without reasoning_effort it thinks inline and can exhaust the token budget
    # before the JSON; "none" answers directly, "low"/"high" keep the thinking in a separate reasoning field.
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "reasoning_effort": reasoning,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": "CONTEXT (JSON):\n" + json.dumps(context, indent=1) + "\n\nWrite the JSON answer now."}]}
    t0 = time.time()
    r = requests.post(NIM_URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}, json=payload, timeout=300)
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    return msg.get("content") or "", {"model": model, "reasoning_effort": reasoning, "seconds": round(time.time() - t0, 1),
                                      "usage": d.get("usage"), "reasoning_chars": len(msg.get("reasoning_content") or "")}


def validate_citations(answer: dict, context: dict) -> dict:
    blob = json.dumps(context)
    ids = set(re.findall(r"chr\d+_\d+-\d+_cluster\d+", blob)) | set(re.findall(r"chr\d+_\d+-\d+", blob))
    ids |= {p["protein_id"] for p in context["most_extreme_protein_levels"]}
    ids |= {p for c in context["most_ancestry_informative_clusters_carried"] for p in c["proteins_in_block"]}
    ids |= {g for c in context["most_ancestry_informative_clusters_carried"] for g in c["genes_in_block"]}
    ids |= {p["gene"] for p in context["most_extreme_protein_levels"]}
    ids |= {n["individual_id"] for n in context["nearest_individuals_in_gnn_embedding"]} | {context["individual"]["id"]}
    cited = set(answer.get("cited_ids", []))
    return {"cited": len(cited), "unknown_ids": sorted(cited - ids)}


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--individual", default="HG00096")
    parser.add_argument("--run", default="phenotype_both_raw", help="GNN run folder under outputs/gnn_v2/<chrom>/")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning", default="none", choices=["none", "low", "medium", "high"], help="NIM reasoning_effort")
    parser.add_argument("--dry-run", action="store_true", help="print the prompt context and stop")
    args = parser.parse_args()

    kg, pt, assoc, run_dir, preds, emb, saliency = load_everything(here, args.chrom, args.run)
    context = retrieve(args.individual, kg, pt, assoc, preds, emb, saliency)
    out_dir = here / "outputs" / "graphrag" / args.chrom
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.individual}_context.json").write_text(json.dumps(context, indent=2))
    if args.dry_run:
        print(SYSTEM); print("\nCONTEXT:"); print(json.dumps(context, indent=1)); print(f"\nwrote {out_dir}/{args.individual}_context.json")
        return 0

    raw, meta = call_nim(context, args.model, reasoning=args.reasoning)
    try:
        answer = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except json.JSONDecodeError:
        (out_dir / f"{args.individual}_raw.txt").write_text(raw)
        raise SystemExit(f"model did not return JSON; raw reply saved to {out_dir}/{args.individual}_raw.txt")
    check = validate_citations(answer, context)
    result = {"individual": args.individual, "gnn_run": run_dir.name, "llm": meta, "citation_check": check, "answer": answer}
    (out_dir / f"{args.individual}_insight.json").write_text(json.dumps(result, indent=2))
    print(f"{args.model}  reasoning={meta['reasoning_effort']}  {meta['seconds']}s  tokens {meta['usage']}\n")
    print(answer.get("summary", ""))
    print("\nancestry:", answer.get("ancestry_assessment", ""))
    print("phenotype:", answer.get("phenotype_assessment", ""))
    for link in answer.get("genome_proteome_links", []):
        print(" link:", link)
    for c in answer.get("caveats", []):
        print(" caveat:", c)
    print(f"\ncitations: {check['cited']} cited, unknown ids: {check['unknown_ids'] or 'none'}")
    print(f"wrote {out_dir}/{args.individual}_insight.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
