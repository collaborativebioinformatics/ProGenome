#!/usr/bin/env python3
"""Federated training of the ProGenome genome+proteome GNN with NVFlare FedAvg (simulation).

    python federated/job.py                       # 3 sites, 10 rounds x 5 local epochs, workspace under outputs/federated/
    python federated/job.py --rounds 20 --local-epochs 3

Sites = the 3 mixed-ancestry hospital sites of the synthetic proteomics.  Each site trains on its own
Individual nodes only; the cluster/block/gene/protein graph is public and identical everywhere; only model
weights travel.  Evaluate the resulting global model centrally with federated/evaluate_global.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.recipe import SimEnv

HERE = Path(__file__).resolve().parent            # genomics/federated
GENOMICS = HERE.parent
CO = ("cluster", "co_occurs", "cluster")
RELATIONS = [["individual", "carries", "cluster"], ["cluster", "rev_carries", "individual"],
             ["cluster", "in_block", "block"], ["block", "rev_in_block", "cluster"], list(CO), ["block", "next_block", "block"],
             ["individual", "measured", "protein"], ["protein", "rev_measured", "individual"],
             ["block", "overlaps", "gene"], ["gene", "rev_overlaps", "block"], ["gene", "encodes", "protein"], ["protein", "rev_encodes", "gene"]]


def write_model_config(kg_dir: Path, cfg_path: Path, hidden: int, layers: int, dropout: float) -> dict:
    """Derive every constructor value from the public graph so server and sites build the same model."""
    data = torch.load(kg_dir / "hetero_v2.pt", weights_only=False)
    n_clusters = data["cluster"].x.shape[0]
    n_proteins = data["protein"].x.shape[0]
    cfg = {"in_dims": {"individual": n_clusters + 2 * n_proteins,          # raw carrier row + protein z + observed mask
                       "cluster": data["cluster"].x.shape[1], "block": data["block"].x.shape[1],
                       "gene": data["gene"].x.shape[1], "protein": data["protein"].x.shape[1]},
           "relations": RELATIONS, "hidden": hidden, "out_dim": 2, "layers": layers, "dropout": dropout, "aggr": "mean"}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workspace", type=Path, default=None, help="default outputs/federated/<chrom>/workspace")
    args = parser.parse_args()

    kg_dir = GENOMICS / "outputs" / "kg" / args.chrom
    split_path = GENOMICS / "outputs" / "splits" / args.chrom / f"split_seed{args.seed}.csv"
    out_dir = GENOMICS / "outputs" / "federated" / args.chrom
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = args.workspace or out_dir / "workspace"
    cfg_path = out_dir / "model_args.json"
    cfg = write_model_config(kg_dir, cfg_path, args.hidden, args.layers, args.dropout)
    print("model config:", json.dumps(cfg["in_dims"]), f"hidden {cfg['hidden']} layers {cfg['layers']}")

    train_args = (f"--genomics-dir {GENOMICS} --kg-dir {kg_dir} --split-path {split_path} --model-config {cfg_path} "
                  f"--local-epochs {args.local_epochs} --seed {args.seed}")
    recipe = FedAvgRecipe(
        name="progenome_fedavg",
        model={"class_path": "model.ProGenomeGNN", "args": {"config_path": str(cfg_path)}},
        min_clients=args.sites,
        num_rounds=args.rounds,
        train_script=str(HERE / "client.py"),
        train_args=train_args,
        key_metric="val_balanced_accuracy",
        key_metric_mode="max",
        server_expected_format=ExchangeFormat.PYTORCH,
    )
    recipe.add_decomposers(["nvflare.app_opt.pt.decomposers.TensorDecomposer"])
    recipe.add_server_file(str(HERE / "model.py"))

    env = SimEnv(num_clients=args.sites, workspace_root=str(workspace))
    run = recipe.execute(env)
    print("status:", run.get_status())
    print("result:", run.get_result())
    (out_dir / "run_info.json").write_text(json.dumps({"rounds": args.rounds, "local_epochs": args.local_epochs, "sites": args.sites,
                                                       "workspace": str(workspace), "status": str(run.get_status())}, indent=2))


if __name__ == "__main__":
    main()
