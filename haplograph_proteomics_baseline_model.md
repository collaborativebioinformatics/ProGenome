# Haplograph + proteomics: a simple baseline model

Scope: the step right after haploblock clustering. Input is the haplograph edge list
(`edges_lift_above_threshold.csv.gz`) plus a proteomics BED file. Goal: a first,
deliberately simple model before investing in a GNN.

## 1. What the haplograph data represents

Example rows (`source,target,weight,lift`):

```
chr22_17420471-17463955_cluster123,chr22_17573553-17596894_cluster65,7,5.50494
chr22_17420471-17463955_cluster123,chr22_17795362-17851807_cluster742,5,5.30833
chr22_17420471-17463955_cluster123,chr22_23355114-23374984_cluster99,9,10.0579
```

- **Nodes**: `(chrom, start, end, cluster_id)` — a haploblock region plus its
  MMseqs2 cluster assignment.
- **weight**: co-occurrence count — how many phased haplotypes carry both
  cluster assignments.
- **lift**: the classic association-rule-mining statistic,
  `lift(A,B) = P(A ∩ B) / (P(A) * P(B))`. Values > 1 mean the two
  haploblock-clusters co-occur more than chance predicts.
- The file name (`edges_lift_above_threshold`) confirms this graph is already
  pruned to statistically linked pairs only.
- Some pairs span tens of megabases (17.4 Mb → 23.3 Mb on chr22) — this is
  **long-range co-occurrence**, not physical LD/proximity, which is exactly
  the kind of signal a graph structure captures that a per-block model can't.

## 2. Attach proteomics to the graph via interval overlap

Parse node IDs into genomic coordinates, then join against the proteomics BED
file. `bioframe` keeps everything in pandas — no need to shell out to the
`bedtools` binary.

```python
import pandas as pd
import bioframe as bf

edges = pd.read_csv("edges_lift_above_threshold.csv.gz")  # source,target,weight,lift

def parse_node(node_id):
    chrom, rest = node_id.split("_", 1)
    coords, cluster = rest.rsplit("_", 1)
    start, end = coords.split("-")
    return chrom, int(start), int(end), cluster

nodes = pd.DataFrame({"node_id": pd.unique(edges[["source", "target"]].values.ravel())})
nodes[["chrom", "start", "end", "cluster"]] = nodes["node_id"].apply(
    lambda n: pd.Series(parse_node(n))
)

proteomics = pd.read_csv(
    "proteomics.bed", sep="\t",
    names=["chrom", "start", "end", "protein_id", "abundance"],
)

overlaps = bf.overlap(nodes, proteomics, how="left", suffixes=("", "_prot"))

node_features = (
    overlaps.groupby("node_id")
    .agg(n_proteins=("protein_id_prot", "count"),
         mean_abundance=("abundance_prot", "mean"))
    .fillna(0)
    .reset_index()
)
```

## 3. Simple model: predict lift from endpoint protein features

Well-posed question for a first pass: **do the proteins sitting at two loci
explain why those loci are statistically linked (their lift score)?** No GNN
needed yet — a plain Ridge regression on the two endpoints' protein features
is the right amount of model for a chr22 POC.

```python
df = (
    edges
    .merge(node_features.add_suffix("_src"), left_on="source", right_on="node_id_src")
    .merge(node_features.add_suffix("_tgt"), left_on="target", right_on="node_id_tgt")
)

X = df[["n_proteins_src", "mean_abundance_src",
        "n_proteins_tgt", "mean_abundance_tgt", "weight"]]
y = df["lift"]

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = Ridge(alpha=1.0).fit(X_train, y_train)
print("R^2:", model.score(X_test, y_test))
```

**Known blind spot:** the edge list was already filtered to "lift above
threshold," so this only ever sees positive examples. It can learn what
predicts a *higher* lift among already-linked pairs, but not whether two loci
link at all — that needs negative sampling (random non-edge pairs), which the
current file doesn't give us.

## 4. Natural v2, once the baseline is validated

Swap Ridge for a 2-layer `GCNConv`/`SAGEConv`, passing `lift` directly as
`edge_weight` (PyG's `GCNConv` accepts edge weights natively) and
`node_features` as `x`. Same data, but the model propagates protein signal
*through* the graph structure instead of only looking at direct pairs. Hold
off on this until the Ridge baseline shows there's signal worth the added
complexity.

## Open question for the team

If per-individual carrier data becomes available (which node/cluster each
person "activates" at each block), the target variable changes entirely —
that moves this from "explain the graph's own edge weights" to "predict
phenotype from the graph," which is a different model design.
