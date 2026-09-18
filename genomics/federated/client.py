#!/usr/bin/env python3
"""NVFlare Client API training script: one hospital site of the ProGenome graph.

What stays at the site: its own Individual nodes (labels, CARRIES and MEASURED edges).  What is shared:
the public cluster / block / gene / protein graph and the model weights.  Each round the site receives the
global weights, evaluates them on its own validation and test people, trains a few local full-batch
epochs on its own training people, and sends back weights + metrics + the number of optimizer steps.

Run only through NVFlare (job.py); not a standalone trainer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import MetaKey

CO = ("cluster", "co_occurs", "cluster")
MEAS = ("individual", "measured", "protein")
RMEAS = ("protein", "rev_measured", "individual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--genomics-dir", required=True, help="the genomics/ folder (for haplokg imports)")
    parser.add_argument("--kg-dir", required=True, help="outputs/kg/<chrom> with hetero_v2.pt and the v2 tables")
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def site_data(args, site_name: str, device):
    """Load the public graph, keep only this site's individuals, build model inputs."""
    sys.path.insert(0, args.genomics_dir)
    import haplokg
    import haplokg_proteins as hp
    import json

    kg = haplokg.load_kg(args.kg_dir)
    pt = hp.load_protein_tables(args.kg_dir)
    data = torch.load(Path(args.kg_dir) / "hetero_v2.pt", weights_only=False)
    cfg = json.loads(Path(args.model_config).read_text())
    relations = [tuple(r) for r in cfg["relations"]]

    ind2 = pt["individuals"]
    sites = pt["label_maps"]["site"]                                   # e.g. ["SITE1","SITE2","SITE3"]
    site_idx = int(site_name.rsplit("-", 1)[-1]) - 1                   # site-1 -> SITE1
    my_site = sites[site_idx]
    members = np.flatnonzero((ind2["site_code"] == site_idx).to_numpy() & (ind2["phenotype_code"] >= 0).to_numpy())
    split_frame = __import__("pandas").read_csv(args.split_path)
    split = split_frame["split"].to_numpy()[members]

    # HeteroData.subgraph slices every attribute of the store; the id lists are plain Python lists, so drop them first
    for key in list(data["individual"].keys()):
        if key != "num_nodes" and not isinstance(data["individual"][key], torch.Tensor):
            del data["individual"][key]
    sub = data.subgraph({"individual": torch.as_tensor(members, dtype=torch.long)})     # other node types untouched
    x_ind = np.concatenate([kg["carries"][members].toarray().astype(np.float32),
                            pt["abundance"][members].toarray().astype(np.float32),
                            pt["observed"][members].toarray().astype(np.float32)], axis=1)
    x_dict = {"individual": torch.from_numpy(x_ind), "cluster": data["cluster"].x, "block": data["block"].x,
              "gene": data["gene"].x, "protein": data["protein"].x}
    edge_index = {rel: sub[rel].edge_index for rel in relations}
    lift = data[CO].edge_attr[:, 1]
    edge_weight = {CO: torch.log(lift) / torch.log(lift).max(), MEAS: sub[MEAS].edge_attr[:, 0], RMEAS: sub[RMEAS].edge_attr[:, 0]}
    y = sub["individual"].y_phenotype
    masks = {k: torch.from_numpy(split == k) for k in ("train", "val", "test")}
    to = lambda d: {k: v.to(device) for k, v in d.items()}
    return my_site, to(x_dict), to(edge_index), to(edge_weight), y.to(device), to(masks)


def evaluate(model, x, ei, ew, y, mask) -> dict:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits, _ = model(x, ei, ew)
    finally:
        model.train(was_training)
    prob = torch.softmax(logits, 1)[:, 1]
    yt = y[mask].cpu().numpy()
    pb = prob[mask].cpu().numpy()
    pr = (pb >= 0.5).astype(int)
    if len(yt) == 0:
        raise RuntimeError("evaluation split is empty at this site")
    out = {"balanced_accuracy": float(balanced_accuracy_score(yt, pr)), "n": int(len(yt))}
    if len(np.unique(yt)) == 2:
        out["auc"] = float(roc_auc_score(yt, pb))
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flare.init()
    site_name = flare.get_site_name()
    my_site, x, ei, ew, y, masks = site_data(args, site_name, device)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import ProGenomeGNN

    model = ProGenomeGNN(args.model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = torch.bincount(y[masks["train"]], minlength=2).float()
    class_weight = (counts.sum() / counts.clamp(min=1) / 2).to(device)      # local class balance, a training policy
    print(f"[{site_name}] {my_site}: train {int(masks['train'].sum())} val {int(masks['val'].sum())} test {int(masks['test'].sum())} on {device}", flush=True)

    while flare.is_running():
        input_model = flare.receive()
        model.load_state_dict(input_model.params)
        val = evaluate(model, x, ei, ew, y, masks["val"])
        test = evaluate(model, x, ei, ew, y, masks["test"])
        metrics = {"val_balanced_accuracy": val["balanced_accuracy"], "test_balanced_accuracy": test["balanced_accuracy"],
                   "test_auc": test.get("auc", float("nan")), "n_val": val["n"], "n_test": test["n"]}
        print(f"[{site_name}] round {input_model.current_round}: global model val bal-acc {val['balanced_accuracy']:.3f} test AUC {metrics['test_auc']:.3f}", flush=True)
        if flare.is_evaluate():
            flare.send(flare.FLModel(metrics=metrics))
            continue

        steps = 0
        model.train()
        for _ in range(args.local_epochs):                               # full-batch: one optimizer step per epoch
            optimizer.zero_grad()
            logits, _ = model(x, ei, ew)
            loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]], weight=class_weight)
            loss.backward()
            optimizer.step()
            steps += 1
        params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        flare.send(flare.FLModel(params=params, metrics=metrics, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: steps}))


if __name__ == "__main__":
    main()
