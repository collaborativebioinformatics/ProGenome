"""Model shared by the NVFlare server and every site.

Architecture is identical to `train_gnn_v2.HeteroGNNv2` (kept here as a self-contained copy so the
server app can import it without the rest of the pipeline).  All constructor values come from one JSON
file written by job.py, so server and clients build byte-identical state dicts.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GraphConv, HeteroConv, SAGEConv

WEIGHTED = {("cluster", "co_occurs", "cluster"), ("individual", "measured", "protein"), ("protein", "rev_measured", "individual")}


class ProGenomeGNN(nn.Module):
    def __init__(self, config_path: str):
        super().__init__()
        cfg = json.loads(Path(config_path).read_text())
        in_dims: dict = cfg["in_dims"]
        relations = [tuple(r) for r in cfg["relations"]]
        hidden, out_dim, layers, dropout, aggr = cfg["hidden"], cfg["out_dim"], cfg["layers"], cfg["dropout"], cfg["aggr"]
        self.node_types = list(in_dims)
        self.proj = nn.ModuleDict({t: nn.Linear(d, hidden) for t, d in in_dims.items()})
        self.convs = nn.ModuleList([HeteroConv({
            rel: (GraphConv(hidden, hidden, aggr=aggr) if rel in WEIGHTED else SAGEConv((hidden, hidden), hidden, aggr=aggr))
            for rel in relations}, aggr="sum") for _ in range(layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(hidden) for t in self.node_types}) for _ in range(layers)])
        self.head = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict, edge_weight_dict):
        h = {t: self.proj[t](x_dict[t]) for t in self.node_types}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict, edge_weight_dict=edge_weight_dict)
            h = {t: F.dropout(F.relu(norm[t](out[t] + h[t])), p=self.dropout, training=self.training) if t in out else h[t] for t in h}
        return self.head(h["individual"]), h
