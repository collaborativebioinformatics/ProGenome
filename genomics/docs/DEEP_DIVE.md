# ProGenome deep dive — every step, with the formulas, shapes and numbers

Companion to `METHODS.md` (the short account) and `../README.md` (how to run). This file explains *how*
each stage works at the level of tensors and equations, and *why* it was built that way. chr22, seed 42.

---

## 0. Vocabulary

| term | meaning here |
|---|---|
| haploblock (block) | a recombination-defined stretch of chromosome; 669 on chr22, median 29.7 kb, tiling 17.1–50.2 Mb with no gaps |
| haplotype | one of a person's two copies of a block's sequence (phased: we know which variants sit on the same copy) |
| cluster | a group of haplotypes of one block that MMseqs2 called similar; a person *carries* a cluster if at least one of their two haplotypes is in it |
| carrier matrix | `M ∈ {0,1}^{N×C}`: `M[i,c] = 1` iff person `i` carries cluster `c`; N = 2,548 people, C = 6,551 kept clusters |
| support | number of carriers of a cluster, `Σ_i M[i,c]` |
| lift | `P(A∩B) / (P(A)·P(B))` for two clusters: >1 means they travel together more than chance |
| phenotype | a property of a person used as a label: real (ancestry, population, sex) or synthetic (case/control) |
| site | a hospital; owns its people's rows of `M` and their proteomics; never sees other sites' rows |

---

## 1. Inputs, byte by byte

### 1.1 `nodes.csv.gz` (HaploGraph)

```
id,high_dim_edge,HG00096,HG00097,...,NA21144          # 2 + 2,548 columns
chr22_17099658-17118145_cluster1,chr22_17099658-17118145,1,0,...,1
```

248,254 rows (one per cluster of any size), 2,548 person columns. Uncompressed 1.3 GB, gzipped 9 MB.
Read in 8,192-row chunks with pandas, dtype `int8` for every person column, each chunk turned into a
`scipy.sparse.csr_matrix` and stacked (`haplokg.read_node_matrix`). Peak memory stays under 1 GB; the
full dense int8 matrix would be 632 MB and int64 would be 5 GB.

### 1.2 The cluster filter

`support[c] = Σ_i M[i,c]`; keep `c` iff `25 ≤ support[c] ≤ N − 25`. Rationale: a cluster carried by 2,540
of 2,548 people is as uninformative as one carried by 8 — the "informative side" has 8 people either way
(this mirrors the HaploGraph's own `MIN_CLUSTER_SUPPORT=25` symmetric edge filter). 248,254 → 6,551.
176,903 of the dropped clusters are singletons (one haplotype). The filter reproduces the edge file's node
set exactly: 0 of 187,030 edges lose an endpoint.

### 1.3 Edges

`source,target,weight,lift`. `weight` = number of people carrying both; `lift` as above, ≥ 5 by
construction. Stored both directions in PyG (374,060 directed edges) with `edge_attr = [weight, lift]`.
We do **not** use the raw `edges.csv` (1 GB for chr6): its weights are dominated by one near-universal
haplotype ("mega-hub"), which the lift normalisation removes.

### 1.4 Phenotypes

`phenotypes_real.csv` long format `individual_id,phenotype,value,source` → pivoted wide. Labels encoded as
integers by sorted class name (`AFR=0, AMR=1, EAS=2, EUR=3, SAS=4`; `female=0, male=1`; 26 populations);
`-1` for the 45 people without labels (they stay in the graph as unlabelled nodes and are excluded from
every loss and metric).

### 1.5 Proteins

`uniprot_chr22.bed` (BED12 from the UCSC UniProt track): 917 isoform rows like `Q9BXF3-1`, `Q9BXF3-2`.
Isoforms are collapsed to the base accession (`Q9BXF3`), giving 460 proteins with the union span of their
isoforms; gene symbols from `gene_symbol_cache.csv` (UniProt lookup) give 458 genes.

---

## 2. Graph schema and how each edge is computed

| edge | computation | count |
|---|---|---|
| `Individual —CARRIES→ Cluster` | non-zeros of `M` after the filter (`carries = M[kept].T`), coordinates `(i, c)` | 2,365,574 |
| `Cluster —IN_BLOCK→ Block` | parsed from the cluster id (`chr22_<start>-<end>_clusterN` → block `chr22_<start>-<end>`) | 6,551 |
| `Block —NEXT_BLOCK→ Block` | blocks sorted by `start`; consecutive pairs | 668 |
| `Cluster —CO_OCCURS→ Cluster` | edge file, ids mapped to kept-cluster indices | 187,030 |
| `Block —OVERLAPS→ Gene` | `block.start ≤ gene.end` and `block.end ≥ gene.start+1` (BED is 0-based half-open; blocks 1-based inclusive) | 1,063 |
| `Gene —ENCODES→ Protein` | from the BED (one gene → up to 2 proteins) | 460 |
| `Individual —MEASURED→ Protein` | one edge per observed (person, protein) measurement; attributes `[z, log2]`; missing = no edge | 1,065,712 |

Node features (all z-scored, NaN → 0):

- `cluster.x ∈ ℝ^{6551×7}`: `log1p(support)`, `support/N`, block `log10(length)`, block entropy, block dominance, block `log1p(n_clusters)`, block singleton rate.
- `block.x ∈ ℝ^{669×5}`: the five block statistics.
- `gene.x`, `protein.x ∈ ℝ^{·×2}`: `log10(span)`, `log1p(n_proteins | n_isoforms)`.
- `individual.x`: set by the model (§5), not stored in the graph.

The join key everywhere is the 1000G sample id. Phenotype labels are stored as `individual.y_<name>` tensors —
node **properties**, never nodes or edges. This matters: if `Individual —HAS→ Phenotype` existed, a two-layer
GNN would read the label from its neighbour and report ~100 %.

---

## 3. The statistics (what "co-occurrence with phenotypes" means numerically)

### 3.1 Cluster × phenotype (`cooccurrence_analysis.py`)

For each cluster `c` and a k-class label, the 2×k table of (carrier, non-carrier) × class. With one-hot
`A ∈ {0,1}^{N×k}` and the carrier matrix `M`: observed carriers per class `O₁ = Aᵀ M ∈ ℝ^{k×C}` in one sparse
product; non-carriers `O₀ = n_class − O₁`; expected `E₁ = n_class · support / N`, `E₀ = n_class − E₁`;
`χ² = Σ (O₁−E₁)²/E₁ + Σ (O₀−E₀)²/E₀` with `k−1` degrees of freedom; Cramér's `V = √(χ² / (N·min(1, k−1))) = √(χ²/N)`
because one dimension is binary. p-values → Benjamini–Hochberg q-values. A cluster's *dominant* ancestry is
the class with the largest enrichment `frac_class / frac_overall`.

Result: ancestry 6,470/6,551 clusters at q < 0.05 (median V 0.20, max 0.81); population 6,378; **sex 0**
(median V 0.013, max 0.03). Sex is the negative control: any method that found sex signal on an autosome
would be fitting noise.

### 3.2 Edge × phenotype

For every lift edge, both endpoints' 5-vector of enrichment deviations `(frac_a/frac_overall − 1)` is
L2-normalised and the cosine similarity taken: mean **0.86** for real edges vs **0.22** for a null made by
permuting the target column (keeps each source's degree). 86 % of edges join two clusters with the same
dominant ancestry (42 % under the null). Similarity rises with lift (quartiles 0.84, 0.84, 0.85, 0.90).
Reading: the graph structure *is* population structure; the median endpoint distance is tens of Mb, i.e.
these are not physical linkage but co-inheritance within ancestries.

---

## 4. Baseline: logistic regression on `M`

`LogisticRegression(C ∈ {0.01, 0.1, 1}, max_iter=5000)` on the sparse float carrier matrix, one model per
label; C chosen on validation balanced accuracy; test scores: ancestry 0.977, population 0.614 (26 classes,
60–113 people each), sex 0.463. The split: `train_test_split` stratified on ancestry, 70/15/15 over the
2,503 labelled people, seed 42, written once to `outputs/splits/chr22/split_seed42.csv` and reused by every
model, so all held-out numbers are on the same 376 people.

---

## 5. Embeddings: how a node becomes a vector

### 5.1 Truncated SVD (`--init svd`, default)

`M ≈ U Σ Vᵀ` with `k = 32` (`scipy.sparse.linalg.svds`): `U ∈ ℝ^{2548×32}`, `Σ` diagonal (top values 1025, 287,
198, 112, 101), `V ∈ ℝ^{6551×32}`. Person embedding `UΣ`, cluster embedding `VΣ`, each column z-scored.
Both live in one space: `(UΣ)(VΣ)ᵀ ≈ MΣ`, so a person is close to the clusters they carry and two people who
share clusters are close to each other. No labels are used, so nothing can leak into the test split.

### 5.2 Node2Vec (`--init node2vec`)

Random walks (length 20, 10 per node, context 10) on the homogeneous graph of people + clusters with
CARRIES and CO_OCCURS edges, skip-gram objective with 1 negative sample; `pyg-lib` provides the walks
(`torch_geometric.nn.Node2Vec` requires `pyg-lib ≥ 0.6`; `torch_cluster` is deprecated). Runs in the GPU image.

### 5.3 Learned and raw

`--init learned`: an `nn.Embedding(2548, 64)` per person — no prior, and it reached only 0.585 on ancestry.
`--init raw`: the person's own carrier row (6,551 zeros/ones) through the first linear layer — the most
information but 6,551 × 64 weights in that layer.

### 5.4 What the GNN adds to an embedding

`embeddings.py` scores each space by silhouette (how tight and separated the ancestry groups are) and by 5-NN
balanced accuracy on test people. SVD-32: silhouette 0.06, 5-NN 0.90. GNN hidden layer (ancestry run):
**0.70 / 0.97**. That 0.06 → 0.70 is the GNN's contribution: it reorganises the space so that groups are
compact, which is what downstream retrieval (nearest neighbours for the decoder) and clustering need.

---

## 6. The GNN, layer by layer

Hidden size `d = 64`, two layers.

1. **Projection**: for each node type `t`, `h_t⁽⁰⁾ = W_t x_t + b_t`, `W_t ∈ ℝ^{d×in_t}`.
   `in_individual` = 32 (SVD) or 6,551 (raw), + 460 + 460 in v2 (protein z and observed mask);
   `in_cluster` = 7 (+32 with SVD), `in_block` = 5, `in_gene` = `in_protein` = 2.
2. **Message passing** (`HeteroConv`, one operator per relation, results summed per target type):
   - `SAGEConv` on `carries`, `rev_carries`, `in_block`, `rev_in_block`, `next_block`, `overlaps`, `encodes` (each with its reverse):
     `h_v' = W₁ h_v + W₂ · mean_{u ∈ N_r(v)} h_u` — the node keeps its own state and adds the mean of its neighbours under relation `r`.
   - `GraphConv` on `co_occurs` and `measured`/`rev_measured`:
     `h_v' = W₁ h_v + W₂ · mean_{u} w_{uv} h_u` with `w = log(lift)/max log(lift)` for co-occurrence and `w = z` (the harmonised protein level, signed) for measured — a protein that is high in this person pushes with positive weight, one that is low with negative weight.
   - Sum over relations, then `h ← Dropout(ReLU(LayerNorm(h' + h)))` (residual keeps the projection signal alive through both layers).
3. **Head**: `logits = W_out h_individual⁽²⁾ + b`, `W_out ∈ ℝ^{k×64}`.

What two rounds mean for a person `i`: round 1 pulls in the clusters `i` carries (and, in v2, `i`'s
proteins); round 2 pulls in what those clusters co-occur with, their blocks, and what genes/proteins sit in
those blocks. So `h_i⁽²⁾` summarises "my haplotypes, their population context, and my proteome" in 64 numbers.

**Loss**: cross-entropy over training people only, with class weights `w_k = N_train / (k · n_k)` so rare
classes (AMR, small populations) are not ignored. **Optimiser**: Adam, lr 5·10⁻³, weight decay 5·10⁻⁴.
**Early stopping**: validation balanced accuracy, patience 30, best weights restored. Full-batch: one
epoch = one forward/backward over the whole graph (2.4 M + 0.37 M + 1.07 M edges): 0.95 s on the M2 CPU,
0.10 s on the A100.

**Metrics**: accuracy; balanced accuracy = mean per-class recall (robust to the 26-class imbalance);
macro-F1; ROC-AUC for the binary phenotype; all computed only on test people with a label.

Parameters: ≈105 k with SVD input; ≈0.5 M with raw input (dominated by the 6,551×64 projection).

---

## 7. The synthetic proteome: exact generative model

Per person `i` (real 1000G id, real sex, random age 18–85, site assigned by shuffling within each ancestry
and splitting into 3):

- Causal clusters: 20 kept clusters drawn from blocks that encode a protein, with carrier frequency 5–60 %.
  `β_c ~ N(0, 1.5²)`. Logit `η_i = Σ_c β_c M[i,c] + 0.02·(age_i − mean age)`; intercept set at the
  `1−0.35` quantile of `η`; `case_i ~ Bernoulli(σ(η_i − intercept))` → 38 % cases.
- Protein `p` for person `i`: `y_ip = b_p + β^{age}_p·age_z + β^{sex}_p·sex_i + β^{pheno}_p·case_i + Σ_{c: cis(c)=p} β^{cis}_c M[i,c] + s_{site(i),p} + ε_bio + ε_tech`
  with `b_p ~ U(6,16)` log2, `β^{age}, β^{sex} ~ N(0, 0.5²)`, `β^{pheno}_p ~ N(0, 0.5²)` for a random 15 % of proteins and 0 otherwise (70 responsive proteins), `β^{cis}_c ~ N(0, 1²)` for the one protein in the causal cluster's block, site shift `s ~ N(0, 0.3²)`, `ε_bio ~ N(0, 0.8²)`, `ε_tech ~ N(0, 0.3²)`.
- Missingness: `P(missing_ip) = 0.15 · (b_max − b_p)/(b_max − b_min)` — the least abundant proteins are missing most often (MNAR at the detection limit).

Everything is written to `ground_truth.json`, so a model's saliency can be scored against the causal
clusters and a regression against the cis proteins.

**Harmoniser** (`haplokg_proteins.harmonise`): within each `(site, protein)`,
`z = (y − median) / (1.4826 · MAD)`. Robust to outliers, computed from each site's own samples only, removes
`s_{site,p}` exactly (between-site median shift 0.48 → 0.00 log2 in the EDA) while preserving within-site
biology (227 proteins keep a phenotype association at FDR 5 %).

---

## 8. Integration experiments: what each row of the table is

| run | person input | relations | AUC |
|---|---|---|---|
| genome / svd | `UΣ` (32) | genome relations | 0.635 |
| genome / raw | carrier row (6,551) | genome relations | 0.601 |
| proteome | `[z, mask]` (920) → 2-layer MLP, **no graph** | — | 0.962 |
| both / svd | `[UΣ, z, mask]` | all 12 relations | 0.993 |
| both / raw | `[row, z, mask]` | all 12 relations | 0.992 |
| site (both) | as both | all | bal. acc. 0.30 (chance 0.33) |
| ancestry (both) | as both | all | bal. acc. 0.90 |

Why genome-only is low: the phenotype's genomic part is 20 clusters with noisy logistic link → the Bayes
rate is far from 1; why proteome-only is high: 70 proteins each shift by ~N(0,0.5) with noise sd ~0.85, and
the MLP sums the evidence; why both is higher still: the graph brings the genomic evidence in and the
`measured` edges let protein evidence propagate. The site row is the guarantee that none of this is batch.

**Saliency**: `∂ logit_case / ∂ x_individual`, averaged over test cases, restricted to the carrier-row
coordinates, ranked; 3–4 of the top 20 are ground-truth causal (chance 20 × 20/6551 = 0.06).

**Genome → proteome**: predicting all 460 z-scores from `h_i⁽²⁾` gives R² ≈ 0 even for cis proteins,
although their linear ceiling (`r²` between carrying the causal cluster and the protein's z) is 0.54 for the
strongest and 0.17 on average. A per-protein ridge on the carrier row (`Ridge(alpha=1000)`, multi-output)
recovers 4/20 cis proteins at test R² > 0.1 and 0/440 others. Interpretation: a 64-d embedding trained for a
classification target does not preserve single-cluster cis effects; that needs sparse per-protein models or
a pQTL edge prior (UKB-PPP).

---

## 9. Decoder: retrieval → prompt → validated JSON

Retrieval for person `i` (all from the graph, deterministic): profile (ancestry, population, sex, site, age;
true phenotype withheld), GNN prediction from `test_predictions.csv`, the globally salient clusters `i`
carries, the 8 carried clusters with the highest ancestry Cramér's V (each with block, carrier fractions by
ancestry, genes and proteins in the block), the 8 proteins with the largest |z| (with their encoding block
and whether that block holds one of the notable clusters), and the 5 nearest labelled people by Euclidean
distance in `h⁽²⁾`. Serialised as JSON (~4.5 k tokens).

Model: `nvidia/nemotron-3-super-120b-a12b` via the OpenAI-compatible NIM endpoint, `temperature 0.2`,
`max_tokens 4000`, `reasoning_effort="none"`. Without that flag the model emits its chain of thought in the
content and hits the token cap before the JSON (measured: 1,200 tokens of reasoning, no answer);
with `"low"` the reasoning goes to a separate field and the answer still arrives, at 2–4× the latency.
The system prompt forbids new entities and requires ids verbatim; `cited_ids` are checked against the
set of ids present in the context (25 cited, 0 unknown for HG00103, 13 s).

---

## 10. Federated learning with NVFlare: the mechanics

**Partition.** Site `s` gets the people with `site_code == s` (835 / 835 / 833, mixed ancestry). Its graph is
`HeteroData.subgraph({'individual': members})`: the person nodes are re-indexed to that site, their CARRIES
and MEASURED edges kept, every other node type (cluster, block, gene, protein) and their edges kept whole —
those are public. Site 1, for example: 835 people, 774,753 CARRIES, 355,294 MEASURED edges. Person input
= raw carrier row + protein z + mask — computed from the site's own rows only; no cross-site preprocessing
(the harmoniser is already per site). Cluster/block/gene/protein features are public statistics.

**Model config.** `job.py` derives every constructor value from the public graph (input dims, the 12
relations, hidden 64, 2 layers, dropout 0.3) into `model_args.json`; the server builds
`model.ProGenomeGNN(config_path)` from it and each client builds the same, so state-dict keys and shapes
match by construction.

**Round loop (Client API, `client.py`).** `flare.init()` → `flare.get_site_name()` → build the site graph
once → `while flare.is_running(): m = flare.receive(); model.load_state_dict(m.params); evaluate the received
global model on the site's own val/test people; if it is an evaluate-only task, send metrics; else train 5
full-batch epochs (5 optimizer steps) on the site's training people with local class weights; send
FLModel(params=state_dict on CPU, metrics, meta[NUM_STEPS_CURRENT_ROUND]=5)`.

**Server.** `FedAvgRecipe` (NVFlare 2.9, `nvflare.app_opt.pt.recipes.fedavg`): after each round the global
weights are the step-weighted average `θ ← Σ_s n_s θ_s / Σ_s n_s` with `n_s = NUM_STEPS_CURRENT_ROUND`
(equal here — full-batch — so a plain mean; a real deployment would weight by local sample count via
`aggregation_weights`). `key_metric="val_balanced_accuracy"` selects `best_FL_global_model.pt`;
`FL_global_model.pt` is the final round. Tensor-native transport: `server_expected_format=PYTORCH` +
`TensorDecomposer`. `SimEnv(num_clients=3)` runs the three sites as threads on one machine;
`recipe.execute(env)` materialises the job under `outputs/federated/chr22/workspace/`.

**What crosses the site boundary.** Per round per site: one state dict (~0.5 M floats) and five scalars.
No rows of `M`, no protein values, no embeddings of people.

**Result (10 rounds × 5 local epochs, ~70 s on the M2).** Global model scored centrally on the same 376
held-out people as the central model: AUC **0.955** vs central 0.992; per site (their own test people)
AUC 0.977 / 0.968 / 0.938. Per-round curves show the global model still improving at round 9, i.e. 50 local
steps is short of the central run's ~80 epochs. **30 rounds × 5 local epochs (~4 min on the M2): AUC 0.998 /
balanced accuracy 0.963 centrally on the same 376 people — equal to the central model (0.992 / 0.969); per site
AUC 1.000 / 0.999 / 0.997.** Federated training loses nothing here because the sites are i.i.d. draws of the same
population (mixed ancestry by construction); with ancestry-pure sites the averaging would have to fight client drift.

---

## 11. Compute and engineering facts

- Image: `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime` (system-managed Python 3.12 → `PIP_BREAK_SYSTEM_PACKAGES=1`), torch_geometric 2.8.0.post1, pyg-lib 0.9 (the only extension PyG still ships; `torch_scatter` has no wheel for torch 2.14 and PyG no longer needs it), nx-cugraph (NetworkX dispatch to cuGraph with `NX_CUGRAPH_AUTOCONFIG=True`), torch-tensorrt 2.14.
- GPU: A100 80 GB (Crusoe, $1.98/h) — image built natively in ~4 min; epoch 0.10 s; whole v1 pipeline < 2 min. L4 24 GB (GCP) also ran it.
- Inference: eager 36 ms per full graph; `torch.compile` (inductor) 4.9 ms, max |Δlogit| 1.4·10⁻⁶; Torch-TensorRT (`torch.compile(backend="torch_tensorrt")`) did not finish partitioning/compiling this scatter-heavy hetero-GNN in 3 h and is reported as not applicable.
- Graph display: Neo4j 5.26 community (2,548 + 6,551 + 669 nodes, 2.37 M + 187 k + … relationships loaded with batched `UNWIND ... MERGE` in ~2 min); NetworkX 3.6 statistics and GraphML export.
- Packaging: `make setup/run/run-v2/eda/decode/federated/docker/brev`, pinned `requirements.txt`, 11 unit tests on a toy graph, secrets only via environment.

---

## 12. Caveats, stated plainly

1. The phenotype is synthetic. The genome ↔ proteome ↔ phenotype numbers show the *pipeline* recovers a
   planted signal; they are not biology. Ancestry/population/sex results are on real labels.
2. chr22 only; ancestry on one chromosome is almost linearly separable, so the GNN cannot beat logistic
   regression on accuracy there — its value is the embedding space and the integration.
3. The GNN does not recover cis genotype→protein effects; the ridge does for the strong ones.
4. Federated evaluation is on the same held-out people as central; in a real deployment each site would
   report its own held-out score and no central evaluation would exist.
5. TensorRT is not part of the inference claim; `torch.compile` is.
