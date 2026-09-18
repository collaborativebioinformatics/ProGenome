#!/usr/bin/env python3
"""Heterogeneous GNN (PyTorch Geometric) on the haploblock knowledge graph.

Node classification on *individual* nodes.  Messages flow
    individual <-carries-> cluster <-co_occurs (lift-weighted)-> cluster <-in_block-> block <-next_block-> block
so an individual's representation is built from the clusters they carry, from
what those clusters co-occur with across the population, and from the block
structure along the chromosome.

Embeddings ("--init"):
    svd       (default) truncated SVD of the individual x cluster carrier matrix
              gives both individuals and clusters a shared k-dim starting embedding
    node2vec  Node2Vec pretraining on the carries + co_occurs graph (needs pyg-lib;
              available in the GPU image), falls back to svd if missing
    learned   free nn.Embedding per individual (no prior)
    raw       the individual's own 0/1 carrier row (all kept clusters) through a linear
              layer, plus SVD for clusters - closest to the logistic-regression baseline
Cluster and block nodes additionally get their z-scored statistics (support, block
length, entropy, dominance, ...).  The final hidden layer is exported as the
learned embedding of every individual and cluster.

Targets: ancestry (5), population (26), sex (2; negative control - autosomal
chromosome, so the honest answer is chance level).
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
from scipy.sparse.linalg import svds
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch_geometric.nn import GraphConv, HeteroConv, SAGEConv

import haplokg

CO = ("cluster", "co_occurs", "cluster")


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  # MPS is not enabled by default: scatter ops are still patchy there


def zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    std = a.std(axis=0)
    std[std == 0] = 1.0
    return (a - a.mean(axis=0)) / std


def svd_embeddings(carries, k: int, seed: int):
    """Shared k-dim embedding for individuals (rows) and clusters (columns)."""
    u, s, vt = svds(carries.astype(np.float32), k=k, random_state=seed)
    order = np.argsort(s)[::-1]
    u, s, vt = u[:, order], s[order], vt[order]
    return zscore(u * s), zscore(vt.T * s), s


def node2vec_embeddings(data, k: int, device: torch.device, seed: int, epochs: int = 50):
    """Optional: Node2Vec on the homogeneous (individual + cluster) graph; None if pyg-lib is missing."""
    from torch_geometric.typing import WITH_PYG_LIB
    if not WITH_PYG_LIB:            # PyG >= 2.6: random walks live in pyg-lib (torch_cluster is deprecated)
        return None
    from torch_geometric.nn import Node2Vec
    n_ind, n_cl = data["individual"].num_nodes, data["cluster"].num_nodes
    carries = data["individual", "carries", "cluster"].edge_index
    co = data[CO].edge_index
    edge_index = torch.cat([
        torch.stack([carries[0], carries[1] + n_ind]), torch.stack([carries[1] + n_ind, carries[0]]),
        co + n_ind,
    ], dim=1).to(device)
    torch.manual_seed(seed)
    model = Node2Vec(edge_index, embedding_dim=k, walk_length=20, context_size=10, walks_per_node=10,
                     num_negative_samples=1, sparse=True, num_nodes=n_ind + n_cl).to(device)
    loader = model.loader(batch_size=256, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    for epoch in range(epochs):
        total = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward(); optimizer.step(); total += loss.item()
        print(f"  node2vec epoch {epoch + 1}/{epochs} loss {total / len(loader):.4f}")
    emb = model.embedding.weight.detach().cpu().numpy()
    return zscore(emb[:n_ind]), zscore(emb[n_ind:])


class HeteroGNN(nn.Module):
    def __init__(self, in_dims: dict, hidden: int, n_classes: int, layers: int = 2, dropout: float = 0.3,
                 learned_individual: int | None = None, aggr: str = "mean"):
        super().__init__()
        self.learned = nn.Embedding(learned_individual, hidden) if learned_individual else None
        self.proj = nn.ModuleDict({t: nn.Linear(d, hidden) for t, d in in_dims.items() if d > 0})
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(HeteroConv({
                ("individual", "carries", "cluster"): SAGEConv((hidden, hidden), hidden, aggr=aggr),
                ("cluster", "rev_carries", "individual"): SAGEConv((hidden, hidden), hidden, aggr=aggr),
                ("cluster", "in_block", "block"): SAGEConv((hidden, hidden), hidden, aggr=aggr),
                ("block", "rev_in_block", "cluster"): SAGEConv((hidden, hidden), hidden, aggr=aggr),
                CO: GraphConv(hidden, hidden, aggr="mean"),               # takes edge_weight = normalised log-lift
                ("block", "next_block", "block"): SAGEConv((hidden, hidden), hidden, aggr=aggr),
            }, aggr="sum"))
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(hidden) for t in ("individual", "cluster", "block")})
                                    for _ in range(layers)])
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def encode(self, x_dict, edge_index_dict, edge_weight):
        h = {t: self.proj[t](x) for t, x in x_dict.items() if t in self.proj}
        if self.learned is not None:
            h["individual"] = self.learned.weight if "individual" not in h else h["individual"] + self.learned.weight
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict, edge_weight_dict={CO: edge_weight})
            h = {t: F.dropout(F.relu(norm[t](out[t] + h[t])), p=self.dropout, training=self.training) for t in out}
        return h

    def forward(self, x_dict, edge_index_dict, edge_weight):
        h = self.encode(x_dict, edge_index_dict, edge_weight)
        return self.head(h["individual"]), h


def metrics(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)), "n": int(len(y_true))}


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", default="chr22")
    parser.add_argument("--target", default="ancestry", choices=["ancestry", "population", "sex"])
    parser.add_argument("--init", default="svd", choices=["svd", "node2vec", "learned", "raw"])
    parser.add_argument("--aggr", default="mean", choices=["mean", "sum"], help="neighbourhood aggregation of the SAGE layers")
    parser.add_argument("--embed-dim", type=int, default=32, help="k for svd / node2vec")
    parser.add_argument("--node2vec-epochs", type=int, default=50, help="Node2Vec pretraining epochs (5 was clearly undertrained: loss still falling)")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--kg-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    kg_dir = args.kg_dir or here / "outputs" / "kg" / args.chrom
    run_name = f"{args.target}_{args.init}" + ("" if args.aggr == "mean" else f"_{args.aggr}")
    out_dir = args.out_dir or here / "outputs" / "gnn" / args.chrom / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}  target: {args.target}  init: {args.init}")

    kg = haplokg.load_kg(kg_dir)
    data = torch.load(kg_dir / "hetero.pt", weights_only=False)
    split = haplokg.load_or_make_split(kg, here / "outputs" / "splits" / args.chrom / f"split_seed{args.seed}.csv", seed=args.seed)
    y = data["individual"][f"y_{args.target}"].clone()
    classes = data.label_maps[args.target]
    masks = {name: torch.from_numpy((split == name) & (y.numpy() >= 0)) for name in ("train", "val", "test")}
    print({k: int(v.sum()) for k, v in masks.items()})

    # ---- input embeddings -----------------------------------------------------
    t0 = time.time()
    emb_dir = here / "outputs" / "embeddings" / args.chrom
    emb_dir.mkdir(parents=True, exist_ok=True)
    ind_init = cl_init = None
    if args.init in ("svd", "node2vec", "raw"):
        ind_svd, cl_svd, sing = svd_embeddings(kg["carries"], args.embed_dim, args.seed)
        np.save(emb_dir / f"svd{args.embed_dim}_individual.npy", ind_svd)
        np.save(emb_dir / f"svd{args.embed_dim}_cluster.npy", cl_svd)
        ind_init, cl_init = ind_svd, cl_svd
        print(f"svd k={args.embed_dim} done ({time.time() - t0:.1f}s), top singular values {np.round(sing[:5], 1)}")
    if args.init == "node2vec":
        n2v = node2vec_embeddings(data, args.embed_dim, device, args.seed, epochs=args.node2vec_epochs)
        if n2v is None:
            print("pyg-lib not available -> using svd embeddings instead")
        else:
            ind_init, cl_init = n2v
            np.save(emb_dir / f"node2vec{args.embed_dim}_individual.npy", ind_init)
            np.save(emb_dir / f"node2vec{args.embed_dim}_cluster.npy", cl_init)

    x_dict = {"cluster": data["cluster"].x, "block": data["block"].x}
    if cl_init is not None:
        x_dict["cluster"] = torch.cat([x_dict["cluster"], torch.from_numpy(cl_init)], dim=1)
    if args.init == "raw":
        ind_init = kg["carries"].toarray().astype(np.float32)          # 2,548 x 6,551 -> 67 MB, fine
    if ind_init is not None:
        x_dict["individual"] = torch.from_numpy(ind_init)
    in_dims = {t: x.shape[1] for t, x in x_dict.items()}
    lift = data[CO].edge_attr[:, 1]
    edge_weight = torch.log(lift) / torch.log(lift).max()

    x_dict = {t: x.to(device) for t, x in x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    edge_weight = edge_weight.to(device)
    y = y.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}

    model = HeteroGNN(in_dims, args.hidden, len(classes), args.layers, args.dropout,
                      learned_individual=data["individual"].num_nodes if args.init == "learned" else None, aggr=args.aggr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = torch.bincount(y[masks["train"]], minlength=len(classes)).float()
    class_weight = (counts.sum() / counts.clamp(min=1) / len(classes)).to(device)   # rebalance minority classes
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def evaluate(mask):
        model.eval()
        with torch.no_grad():
            logits, h = model(x_dict, edge_index_dict, edge_weight)
        pred = logits.argmax(1)
        return metrics(y[mask].cpu().numpy(), pred[mask].cpu().numpy()), pred, h

    best, best_state, best_epoch, wait, history = -1.0, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        logits, _ = model(x_dict, edge_index_dict, edge_weight)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]], weight=class_weight)
        loss.backward(); optimizer.step()
        val, _, _ = evaluate(masks["val"])
        history.append({"epoch": epoch, "loss": loss.item(), "val_balanced_accuracy": val["balanced_accuracy"]})
        if val["balanced_accuracy"] > best:
            best, best_epoch, wait = val["balanced_accuracy"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss {loss:.4f}  val bal-acc {val['balanced_accuracy']:.3f}  (best {best:.3f} @ {best_epoch})")
        if wait >= args.patience:
            print(f"early stop at epoch {epoch}"); break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    val, _, _ = evaluate(masks["val"])
    test, pred, h = evaluate(masks["test"])
    print(f"\n{args.target}: TEST acc={test['accuracy']:.3f} bal-acc={test['balanced_accuracy']:.3f} macro-F1={test['macro_f1']:.3f}"
          f"  (best epoch {best_epoch}, {train_time:.0f}s, {train_time / len(history):.2f}s/epoch)")
    baseline_path = here / "outputs" / "baseline" / args.chrom / "metrics.json"
    if baseline_path.exists():
        b = json.loads(baseline_path.read_text())["targets"].get(args.target, {}).get("test")
        if b:
            print(f"logistic-regression baseline: acc={b['accuracy']:.3f} bal-acc={b['balanced_accuracy']:.3f} macro-F1={b['macro_f1']:.3f}")

    report = {"chrom": args.chrom, "target": args.target, "init": args.init, "device": str(device), "classes": classes,
              "hidden": args.hidden, "layers": args.layers, "aggr": args.aggr, "embed_dim": args.embed_dim, "best_epoch": best_epoch,
              "epochs_run": len(history), "train_seconds": train_time, "val": val, "test": test,
              "n_parameters": sum(p.numel() for p in model.parameters())}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    ind = kg["individuals"]
    te = masks["test"].cpu().numpy()
    pd.DataFrame({"individual_id": ind.loc[te, "individual_id"].to_numpy(),
                  "true": [classes[i] for i in y[masks["test"]].cpu().numpy()],
                  "pred": [classes[i] for i in pred[masks["test"]].cpu().numpy()]}).to_csv(out_dir / "test_predictions.csv", index=False)
    np.save(out_dir / "embedding_individual.npy", h["individual"].cpu().numpy())
    np.save(out_dir / "embedding_cluster.npy", h["cluster"].cpu().numpy())
    torch.save(best_state, out_dir / "model.pt")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
