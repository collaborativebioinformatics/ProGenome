#!/usr/bin/env python3
"""Genome + proteome GNN (schema v2): does integrating the two beat either alone?

Modalities ("--modality"):
    genome     the v1 graph only (individual <-> cluster <-> block, co-occurrence)
    proteome   no graph: an MLP on the individual's harmonised protein abundances (+ observed mask)
    both       the full v2 graph: genome relations + individual <-measured-> protein <-encodes- gene <-overlaps- block,
               and the abundance vector as part of the individual's input

Targets ("--target"):
    phenotype  the case/control label from the proteomics metadata (synthetic ground truth known)
    site       negative control - sites are mixed-ancestry batches, must stay at chance
    ancestry / sex   as in v1
    proteome   regression: predict every protein's harmonised abundance from the GENOME graph alone
               (genome modality) - scored as R^2 on test individuals, separately for the proteins the
               ground truth says are cis-affected by a causal cluster

With --init raw (genome/both) the individual's input is its own carrier row, which makes a
per-cluster saliency possible: gradient of the case logit w.r.t. the carrier row, averaged over
test cases, ranked -> precision@k against the ground-truth causal clusters.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch_geometric.nn import GraphConv, HeteroConv, SAGEConv

import haplokg
import haplokg_proteins as hp
from train_gnn import pick_device, svd_embeddings, zscore

CO = ("cluster", "co_occurs", "cluster")
MEAS = ("individual", "measured", "protein")
RMEAS = ("protein", "rev_measured", "individual")

GENOME_RELS = [("individual", "carries", "cluster"), ("cluster", "rev_carries", "individual"),
               ("cluster", "in_block", "block"), ("block", "rev_in_block", "cluster"), CO, ("block", "next_block", "block")]
PROTEIN_RELS = [MEAS, RMEAS, ("block", "overlaps", "gene"), ("gene", "rev_overlaps", "block"),
                ("gene", "encodes", "protein"), ("protein", "rev_encodes", "gene")]
WEIGHTED = {CO, MEAS, RMEAS}


class HeteroGNNv2(nn.Module):
    def __init__(self, in_dims: dict, relations: list, hidden: int, out_dim: int, layers: int = 2, dropout: float = 0.3, aggr: str = "mean"):
        super().__init__()
        self.proj = nn.ModuleDict({t: nn.Linear(d, hidden) for t, d in in_dims.items()})
        self.node_types = list(in_dims)
        self.convs = nn.ModuleList([HeteroConv({
            rel: (GraphConv(hidden, hidden, aggr=aggr) if rel in WEIGHTED else SAGEConv((hidden, hidden), hidden, aggr=aggr))
            for rel in relations}, aggr="sum") for _ in range(layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(hidden) for t in self.node_types}) for _ in range(layers)])
        self.head = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def encode(self, x_dict, edge_index_dict, edge_weight_dict):
        h = {t: self.proj[t](x_dict[t]) for t in self.node_types}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict, edge_weight_dict=edge_weight_dict)
            h = {t: F.dropout(F.relu(norm[t](out[t] + h[t])), p=self.dropout, training=self.training) if t in out else h[t] for t in h}
        return h

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        h = self.encode(x_dict, edge_index_dict, edge_weight_dict)
        return self.head(h["individual"]), h


class MLP(nn.Module):
    """proteome-only baseline: same capacity, no graph."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x_dict, *_):
        h = self.net(x_dict["individual"])
        return self.head(h), {"individual": h}


def cls_metrics(y, pred, prob=None) -> dict:
    out = {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)), "n": int(len(y))}
    if prob is not None and len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, prob))
    return out


def r2_per_protein(y, yhat, mask) -> np.ndarray:
    """R^2 per column over observed entries; NaN where a protein has < 5 observed test values."""
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        m = mask[:, j]
        if m.sum() >= 5:
            ss_res = ((y[m, j] - yhat[m, j]) ** 2).sum()
            ss_tot = ((y[m, j] - y[m, j].mean()) ** 2).sum()
            out[j] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--target", default="phenotype", choices=["phenotype", "site", "ancestry", "sex", "proteome"])
    parser.add_argument("--modality", default="both", choices=["genome", "proteome", "both"])
    parser.add_argument("--init", default="svd", choices=["svd", "raw"])
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=20, help="saliency precision@k vs ground-truth causal clusters")
    args = parser.parse_args()
    if args.target == "proteome" and args.modality != "genome":
        parser.error("--target proteome only makes sense with --modality genome (predict the proteome from the genome graph)")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)
    kg_dir = here / "outputs" / "kg" / args.chrom
    out_dir = here / "outputs" / "gnn_v2" / args.chrom / f"{args.target}_{args.modality}_{args.init}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  target: {args.target}  modality: {args.modality}  init: {args.init}")

    kg = haplokg.load_kg(kg_dir)
    pt = hp.load_protein_tables(kg_dir)
    data = torch.load(kg_dir / "hetero_v2.pt", weights_only=False)
    split = haplokg.load_or_make_split(kg, here / "outputs" / "splits" / args.chrom / f"split_seed{args.seed}.csv", seed=args.seed)
    ind2 = pt["individuals"]
    n_ind = len(ind2)

    # ---- individual inputs ----------------------------------------------------------
    abundance = pt["abundance"].toarray().astype(np.float32)             # harmonised z, 0 where unobserved
    observed = pt["observed"].toarray().astype(np.float32)
    parts = []
    if args.modality in ("genome", "both"):
        if args.init == "raw":
            parts.append(kg["carries"].toarray().astype(np.float32))
            raw_offset, n_clusters = 0, kg["carries"].shape[1]
        else:
            ind_svd, cl_svd, _ = svd_embeddings(kg["carries"], args.embed_dim, args.seed)
            parts.append(ind_svd)
    if args.modality in ("proteome", "both"):
        parts += [abundance, observed]
    x_ind = np.concatenate(parts, axis=1)

    x_dict = {"individual": torch.from_numpy(x_ind)}
    relations, ew = [], {}
    if args.modality in ("genome", "both"):
        cl_x = data["cluster"].x
        if args.init == "svd":
            cl_x = torch.cat([cl_x, torch.from_numpy(cl_svd)], dim=1)
        x_dict["cluster"], x_dict["block"] = cl_x, data["block"].x
        relations += GENOME_RELS
        lift = data[CO].edge_attr[:, 1]
        ew[CO] = torch.log(lift) / torch.log(lift).max()
    if args.modality == "both":
        x_dict["gene"], x_dict["protein"] = data["gene"].x, data["protein"].x
        relations += PROTEIN_RELS
        ew[MEAS] = data[MEAS].edge_attr[:, 0]                              # harmonised z as message weight
        ew[RMEAS] = data[RMEAS].edge_attr[:, 0]
    edge_index_dict = {rel: data[rel].edge_index.to(device) for rel in relations}
    ew = {k: v.to(device) for k, v in ew.items()}
    x_dict = {t: x.to(device) for t, x in x_dict.items()}

    # ---- targets --------------------------------------------------------------------
    regression = args.target == "proteome"
    if regression:
        Y = torch.from_numpy(abundance).to(device); Mobs = torch.from_numpy(observed).to(device)
        has_label = observed.sum(1) > 0
        out_dim, classes = abundance.shape[1], None
    else:
        y_np = ind2[f"{args.target}_code"].to_numpy()
        classes = pt["label_maps"][args.target]
        y = torch.from_numpy(y_np).to(device)
        has_label = y_np >= 0
        out_dim = len(classes)
    masks = {k: torch.from_numpy((split == k) & has_label).to(device) for k in ("train", "val", "test")}
    print({k: int(v.sum()) for k, v in masks.items()})

    in_dims = {t: x.shape[1] for t, x in x_dict.items()}
    if args.modality == "proteome":
        model = MLP(in_dims["individual"], args.hidden, out_dim, args.dropout).to(device)
    else:
        model = HeteroGNNv2(in_dims, relations, args.hidden, out_dim, args.layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if not regression:
        counts = torch.bincount(y[masks["train"]], minlength=out_dim).float()
        class_weight = (counts.sum() / counts.clamp(min=1) / out_dim).to(device)

    def loss_fn(logits, mask):
        if regression:
            diff = (logits - Y) ** 2 * Mobs
            return diff[mask].sum() / Mobs[mask].sum().clamp(min=1)
        return F.cross_entropy(logits[mask], y[mask], weight=class_weight)

    def evaluate(mask):
        model.eval()
        with torch.no_grad():
            logits, h = model(x_dict, edge_index_dict, ew)
        m = mask.cpu().numpy()
        if regression:
            yhat, yt, mo = logits.cpu().numpy()[m], abundance[m], observed[m].astype(bool)
            r2 = r2_per_protein(yt, yhat, mo)
            # select on the masked validation MSE (what is optimised); mean R^2 over 460 mostly
            # unpredictable proteins is too noisy to stop on
            score = -float(loss_fn(logits, mask))
            return {"mean_r2": float(np.nanmean(r2)), "val_masked_mse": -score, "n": int(m.sum())}, score, r2, h
        prob = torch.softmax(logits, 1)
        pred = prob.argmax(1)
        met = cls_metrics(y[mask].cpu().numpy(), pred[mask].cpu().numpy(), prob[mask, 1].cpu().numpy() if out_dim == 2 else None)
        return met, met["balanced_accuracy"], pred, h

    best, best_state, best_epoch, wait, history = -np.inf, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        logits, _ = model(x_dict, edge_index_dict, ew)
        loss = loss_fn(logits, masks["train"])
        loss.backward(); optimizer.step()
        val, score, _, _ = evaluate(masks["val"])
        history.append({"epoch": epoch, "loss": loss.item(), "val_score": score})
        if score > best:
            best, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch % 25 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  val {score:.3f}  (best {best:.3f} @ {best_epoch})")
        if wait >= args.patience:
            print(f"early stop at epoch {epoch}"); break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    val, _, _, _ = evaluate(masks["val"])
    test, _, extra, h = evaluate(masks["test"])
    report = {"chrom": args.chrom, "target": args.target, "modality": args.modality, "init": args.init, "device": str(device),
              "hidden": args.hidden, "layers": args.layers, "best_epoch": best_epoch, "epochs_run": len(history),
              "train_seconds": round(train_time, 1), "val": val, "test": test,
              "n_parameters": sum(p.numel() for p in model.parameters())}

    truth_path = here / "outputs" / "proteomics_synth" / args.chrom / "ground_truth.json"
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else None
    if regression:
        r2 = extra
        report["test"]["median_r2"] = float(np.nanmedian(r2))
        if truth:
            prot_ids = list(pt["proteins"]["protein_id"])
            cis = {c["cis_protein"] for c in truth["causal_clusters"]}
            is_cis = np.array([p in cis for p in prot_ids])
            report["test"]["mean_r2_cis_proteins"] = float(np.nanmean(r2[is_cis]))
            report["test"]["mean_r2_other_proteins"] = float(np.nanmean(r2[~is_cis]))
        pd.DataFrame({"protein_id": pt["proteins"]["protein_id"], "test_r2": r2}).to_csv(out_dir / "protein_r2.csv", index=False)
        print(f"\nproteome from genome graph: mean test R^2 {test['mean_r2']:.3f} (median {report['test']['median_r2']:.3f})"
              + (f"  cis-affected proteins {report['test']['mean_r2_cis_proteins']:.3f} vs others {report['test']['mean_r2_other_proteins']:.3f}" if truth else ""))
    else:
        print(f"\n{args.target} [{args.modality}/{args.init}]: TEST acc={test['accuracy']:.3f} bal-acc={test['balanced_accuracy']:.3f} "
              f"macro-F1={test['macro_f1']:.3f}" + (f" AUC={test['roc_auc']:.3f}" if "roc_auc" in test else ""))
        te = masks["test"].cpu().numpy()
        pd.DataFrame({"individual_id": ind2.loc[te, "individual_id"], "true": [classes[i] for i in y[masks["test"]].cpu().numpy()],
                      "pred": [classes[i] for i in extra[masks["test"]].cpu().numpy()]}).to_csv(out_dir / "test_predictions.csv", index=False)

        # saliency vs ground truth (raw carrier input, case/control target)
        if args.init == "raw" and args.target == "phenotype" and truth and args.modality != "proteome":
            model.eval()
            x_req = {t: x.clone() for t, x in x_dict.items()}
            x_req["individual"].requires_grad_(True)
            logits, _ = model(x_req, edge_index_dict, ew)
            cases = masks["test"] & (y == 1)
            logits[cases, 1].sum().backward()
            sal = x_req["individual"].grad[:, :n_clusters][cases].mean(0).cpu().numpy()
            order = np.argsort(sal)[::-1]
            causal_idx = {c["cluster_idx"] for c in truth["causal_clusters"]}
            topk = order[:args.top_k]
            hits = int(sum(i in causal_idx for i in topk))
            report["saliency"] = {"top_k": args.top_k, "hits_in_ground_truth": hits, "precision_at_k": hits / args.top_k,
                                  "expected_by_chance": len(causal_idx) / n_clusters * args.top_k}
            pd.DataFrame({"cluster_id": kg["clusters"]["cluster_id"].to_numpy()[order[:100]], "saliency": sal[order[:100]],
                          "is_causal": [i in causal_idx for i in order[:100]]}).to_csv(out_dir / "saliency_top100.csv", index=False)
            print(f"saliency: {hits}/{args.top_k} of the top-{args.top_k} clusters are ground-truth causal (chance {report['saliency']['expected_by_chance']:.2f})")

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    np.save(out_dir / "embedding_individual.npy", h["individual"].detach().cpu().numpy())
    torch.save(best_state, out_dir / "model.pt")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
