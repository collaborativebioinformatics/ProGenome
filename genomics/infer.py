#!/usr/bin/env python3
"""Inference with a trained GNN: predictions + embeddings for every individual, timed.

    python infer.py --run ancestry_svd                    # eager PyTorch (CPU or GPU)
    python infer.py --run ancestry_svd --compile tensorrt # Torch-TensorRT via torch.compile (GPU)
    python infer.py --run ancestry_svd --compile inductor # torch.compile default backend

TensorRT note: the GNN's message passing is scatter/gather over 2.4M edges, which
TensorRT does not compile natively.  torch.compile(backend="torch_tensorrt")
partitions the graph, runs the dense parts (all Linear layers, norms, the head)
as TensorRT engines and falls back to PyTorch for the rest, so the speed-up is
real but bounded - the benchmark prints both numbers instead of assuming one.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import haplokg
from train_gnn import CO, HeteroGNN, pick_device, svd_embeddings


def build_inputs(kg: dict, data, report: dict, device: torch.device):
    """Recreate exactly the inputs train_gnn.py used for this run."""
    init, k = report["init"], report["embed_dim"]
    x_dict = {"cluster": data["cluster"].x, "block": data["block"].x}
    if init in ("svd", "node2vec", "raw"):
        emb_dir = Path(__file__).resolve().parent / "outputs" / "embeddings" / report["chrom"]
        cl_path = emb_dir / f"{'node2vec' if init == 'node2vec' else 'svd'}{k}_cluster.npy"
        ind_path = emb_dir / f"{'node2vec' if init == 'node2vec' else 'svd'}{k}_individual.npy"
        if cl_path.exists() and ind_path.exists():
            cl_init, ind_init = np.load(cl_path), np.load(ind_path)
        else:
            ind_init, cl_init, _ = svd_embeddings(kg["carries"], k, 42)
        x_dict["cluster"] = torch.cat([x_dict["cluster"], torch.from_numpy(cl_init)], dim=1)
        x_dict["individual"] = torch.from_numpy(kg["carries"].toarray().astype(np.float32) if init == "raw" else ind_init)
    lift = data[CO].edge_attr[:, 1]
    edge_weight = torch.log(lift) / torch.log(lift).max()
    return ({t: x.to(device) for t, x in x_dict.items()},
            {k_: v.to(device) for k_, v in data.edge_index_dict.items()}, edge_weight.to(device))


class Wrapped(torch.nn.Module):
    """Fixed-argument wrapper so torch.compile sees plain tensors, not dicts."""
    def __init__(self, model, edge_index_dict, edge_weight):
        super().__init__()
        self.model, self.edge_index_dict, self.edge_weight = model, edge_index_dict, edge_weight

    def forward(self, x_individual, x_cluster, x_block):
        logits, h = self.model({"individual": x_individual, "cluster": x_cluster, "block": x_block},
                               self.edge_index_dict, self.edge_weight)
        return logits, h["individual"]


def bench(fn, *args, warmup: int = 3, iters: int = 10, device=None) -> float:
    for _ in range(warmup):
        fn(*args)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--run", default="ancestry_svd", help="folder name under outputs/gnn/<chrom>/")
    parser.add_argument("--compile", default="none", choices=["none", "inductor", "tensorrt"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    device = pick_device(args.device)
    run_dir = here / "outputs" / "gnn" / args.chrom / args.run
    report = json.loads((run_dir / "metrics.json").read_text())
    kg_dir = here / "outputs" / "kg" / args.chrom
    kg = haplokg.load_kg(kg_dir)
    data = torch.load(kg_dir / "hetero.pt", weights_only=False)
    x_dict, edge_index_dict, edge_weight = build_inputs(kg, data, report, device)

    model = HeteroGNN({t: x.shape[1] for t, x in x_dict.items()}, report["hidden"], len(report["classes"]),
                      report["layers"], 0.0, learned_individual=data["individual"].num_nodes if report["init"] == "learned" else None,
                      aggr=report.get("aggr", "mean")).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    wrapped = Wrapped(model, edge_index_dict, edge_weight).eval()
    inputs = (x_dict["individual"], x_dict["cluster"], x_dict["block"])

    with torch.no_grad():
        eager_s = bench(wrapped, *inputs, iters=args.iters, device=device)
        logits, emb = wrapped(*inputs)
    result = {"run": args.run, "device": str(device), "eager_ms_per_full_graph": round(eager_s * 1000, 2),
              "compile": args.compile, "precision": args.precision}

    if args.compile != "none":
        if args.compile == "tensorrt":
            try:
                import torch_tensorrt  # noqa: F401
                kwargs = {"backend": "torch_tensorrt", "options": {
                    "enabled_precisions": {torch.float16 if args.precision == "fp16" else torch.float32},
                    "min_block_size": 1, "truncate_double": True}}
            except ImportError:
                print("torch_tensorrt is not installed -> falling back to the inductor backend")
                kwargs = {"backend": "inductor"}
                result["compile"] = "inductor (tensorrt unavailable)"
        else:
            kwargs = {"backend": "inductor"}
        compiled = torch.compile(wrapped, **kwargs)
        try:
            with torch.no_grad():
                compiled_s = bench(compiled, *inputs, iters=args.iters, device=device)
                c_logits, c_emb = compiled(*inputs)
            result["compiled_ms_per_full_graph"] = round(compiled_s * 1000, 2)
            result["speedup"] = round(eager_s / compiled_s, 2)
            result["max_abs_logit_diff_vs_eager"] = float((c_logits.float() - logits.float()).abs().max())
            logits, emb = c_logits, c_emb
        except Exception as exc:  # compilation problems must be visible, not silent
            result["compile_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            print("compile failed, results below are eager:", result["compile_error"])

    pred = logits.float().argmax(1).cpu().numpy()
    prob = torch.softmax(logits.float(), dim=1).max(1).values.cpu().numpy()
    classes = report["classes"]
    ind = kg["individuals"]
    out = pd.DataFrame({"individual_id": ind["individual_id"], "true": ind[report["target"]].fillna("unlabelled"),
                        "pred": [classes[i] for i in pred], "confidence": np.round(prob, 4)})
    out_dir = run_dir / "inference"
    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / "predictions_all_individuals.csv", index=False)
    np.save(out_dir / "embedding_individual.npy", emb.float().cpu().numpy())
    labelled = out["true"] != "unlabelled"
    result["accuracy_all_labelled_individuals"] = float((out.loc[labelled, "true"] == out.loc[labelled, "pred"]).mean())
    result["n_individuals"] = int(len(out))
    (out_dir / f"benchmark_{args.compile}_{args.precision}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
