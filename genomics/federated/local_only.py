#!/usr/bin/env python3
"""What federated learning buys here: each site training ALONE on its own people, versus the federated global model.

    python federated/local_only.py                # after federated/job.py + evaluate_global.py (uses their model_args.json)

For every site the script builds the site's subgraph exactly as client.py does, trains the same model for the same
number of optimizer steps the site performed in the federated run (rounds x local epochs, default 30 x 5 = 150),
and scores it on the site's own held-out people and on the other sites' held-out people (what a lone hospital's
model would do on another hospital's patients). Compare with federated_per_site in evaluation.json, where one
global model was trained on all sites' people without any row leaving its site.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
GENOMICS = HERE.parent
sys.path.insert(0, str(GENOMICS)); sys.path.insert(0, str(HERE))
import client  # noqa: E402  (site_data / evaluate; importing does not start NVFlare)
from job import write_model_config  # noqa: E402
from model import ProGenomeGNN  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--steps", type=int, default=150, help="optimizer steps per site = rounds x local epochs of the federated run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    kg_dir = GENOMICS / "outputs" / "kg" / args.chrom
    fed_dir = GENOMICS / "outputs" / "federated" / args.chrom
    fed_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = fed_dir / "model_args.json"
    if not cfg_path.exists():
        write_model_config(kg_dir, cfg_path, hidden=64, layers=2, dropout=0.3)
    ns = argparse.Namespace(genomics_dir=str(GENOMICS), kg_dir=str(kg_dir), model_config=str(cfg_path),
                            split_path=str(GENOMICS / "outputs" / "splits" / args.chrom / f"split_seed{args.seed}.csv"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sites = {}
    for i in range(1, args.sites + 1):
        name, x, ei, ew, y, masks = client.site_data(ns, f"site-{i}", device)
        sites[name] = (x, ei, ew, y, masks)
        print(f"{name}: train {int(masks['train'].sum())} test {int(masks['test'].sum())}", flush=True)

    results = {}
    for name, (x, ei, ew, y, masks) in sites.items():
        torch.manual_seed(args.seed)
        model = ProGenomeGNN(str(cfg_path)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
        counts = torch.bincount(y[masks["train"]], minlength=2).float()
        class_weight = (counts.sum() / counts.clamp(min=1) / 2).to(device)
        model.train()
        for _ in range(args.steps):
            optimizer.zero_grad()
            logits, _ = model(x, ei, ew)
            F.cross_entropy(logits[masks["train"]], y[masks["train"]], weight=class_weight).backward()
            optimizer.step()
        scores = {"own_test": client.evaluate(model, x, ei, ew, y, masks["test"])}
        for other, (x2, ei2, ew2, y2, m2) in sites.items():
            if other != name:
                scores[f"on_{other}_test"] = client.evaluate(model, x2, ei2, ew2, y2, m2["test"])
        results[name] = scores
        print(f"{name} alone ({args.steps} steps): own test AUC {scores['own_test'].get('auc', float('nan')):.3f} "
              + "  ".join(f"{k} AUC {v.get('auc', float('nan')):.3f}" for k, v in scores.items() if k != "own_test"), flush=True)

    ev_path = fed_dir / "evaluation.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    report = {"steps_per_site": args.steps, "local_only": results,
              "federated_per_site": ev.get("federated_per_site"), "federated_global_model": ev.get("federated_global_model"),
              "central_model_same_test_people": ev.get("central_model_same_test_people")}
    (fed_dir / "local_only_vs_federated.json").write_text(json.dumps(report, indent=2))
    if ev:
        print("\nfederated global model on each site's own test people: "
              + "  ".join(f"{s['site']} AUC {s['auc']:.3f}" for s in ev["federated_per_site"]))
    print(f"wrote {fed_dir / 'local_only_vs_federated.json'}")


if __name__ == "__main__":
    main()
