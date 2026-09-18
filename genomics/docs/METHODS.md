# ProGenome — what the goal is and what we built (technical account)

Written for the team and the manuscript's methods paragraphs. Every number is from `genomics/outputs*/`
on chromosome 22, seed 42, unless stated otherwise.

## 1. The goal, precisely

The README asks three questions. Translated into things a program can do:

| RQ | question | operational form | status |
|---|---|---|---|
| 1 | How can haploblock genomics connect to genes and proteomics in one graph model? | a typed graph in which one **person** node links to the haplotype **clusters** they carry, clusters sit in **blocks**, blocks overlap **genes**, genes encode **proteins**, and the same person links to measured **protein levels** | built (v2 schema) |
| 2 | Can a GNN combine genomic + proteomic information to identify phenotype groups? | train a heterogeneous GNN on that graph to predict a phenotype from the person's neighbourhood; show that genome + proteome beats either alone, and that the embedding separates groups | built; result on synthetic ground truth |
| 3 | Can it be trained across institutions without moving individual-level data? | split the person nodes by site, keep the shared reference graph identical everywhere, exchange only model weights (FedAvg) | designed; next step (NVFlare) |

The mission sentence — "each institution retains its individual-level data locally and trains the same
graph-based model; only model updates are exchanged" — is a statement about *where node types live*. That is
why the whole design starts from making the person a separate node type.

## 2. The data and where it comes from

**HaploGraph (data.haploblocks.org, built at MDxCORE/Rigshospitalet, September 2026, for this hackathon).**
Upstream pipeline (haploblocks.org, Kubica et al. 2025): recombination-rate peaks define *haploblocks*
(recombination-defined regions); each 1000 Genomes individual's two phased haplotypes are extracted per block
from the GRCh38 phased VCF; haplotype sequences of a block are clustered with MMseqs2; each haplotype gets a
*cluster id*; a *hash* encodes strand/chromosome/block/cluster/variants. We did **not** re-run those steps —
we consumed their published outputs and cross-checked them against each other (669 blocks in the boundaries
file = 669 in block_stats = 669 in the node matrix; per-block cluster counts identical for all 669;
248,254 clusters both ways).

Files consumed for chr22:

- `nodes.csv.gz` — 248,254 clusters × 2,548 individuals, 0/1 = individual carries that cluster (on either haplotype).
- `edges_lift_above_threshold.csv.gz` — 187,030 cluster–cluster edges with `weight` (individuals carrying both) and
  `lift = P(A∩B) / (P(A)·P(B))`, pre-filtered to lift ≥ 5 (the raw `edges.csv` is dominated by one
  population-frequency "mega-hub", per the HaploGraph README).
- `block_stats.tsv` — per block: coordinates, length, number of clusters, largest-cluster share (dominance),
  Shannon entropy, singleton count.
- `phenotypes_real.csv` — ancestry (5 super-populations), population (26), sex, for 2,503 of the 2,548 people.
  **1000G has no other phenotype.** Height (whiteboard) therefore had to become a simulated label.
- `proteomics/uniprot_chr22.bed` — 917 UniProt isoform rows → 460 proteins, 458 gene symbols, with genomic spans.
- Proteomics per site: today synthetic (below); planned real sources: Wu et al. 2013 (TMT proteomics on 95 HapMap
  LCLs, 53 CEU/33 YRI/9 EAS, ids are 1000G ids) and UKB-PPP cis-pQTL summary statistics (AWS Open Data).

Why chr22: smallest autosome with a complete HaploGraph (9 MB of graph), so the whole loop runs on a laptop in
minutes; nothing in the code is chromosome-specific (`CHROM=chr21 make run`).

## 3. Knowledge-graph construction (`build_kg.py`, `haplokg.py`)

1. Stream `nodes.csv.gz` in 8,192-row chunks as `int8`, convert each chunk to a CSR sparse matrix, stack →
   clusters × individuals (1.3 GB CSV → 2.8 M non-zeros in memory).
2. **Cluster filter**: keep clusters with `25 ≤ carriers ≤ N−25` (symmetric, like the HaploGraph edge filter).
   248,254 → 6,551 clusters; 176,903 singletons and the near-universal clusters carry no population signal.
   The filter reproduces the edge file's node set exactly (0 of 187,030 edges dropped).
3. Transpose → `carries` = individuals × kept clusters (2,548 × 6,551, 2,365,574 ones; ~928 per person = two
   haplotypes × 669 blocks minus filtered clusters).
4. Join phenotypes on `individual_id`; encode labels as integers, −1 for the 45 unlabelled people.
5. Map edge endpoints to cluster indices; blocks sorted by position → `NEXT_BLOCK` edges.
6. PyG `HeteroData`: node types `individual`, `cluster` (7 z-scored features: log support, support fraction,
   block log-length, entropy, dominance, log n_clusters, singleton rate), `block` (5 features); edge types
   `carries`/`rev_carries`, `in_block`/`rev_in_block`, `co_occurs` (both directions, `edge_attr` = [weight, lift]),
   `next_block` (both directions). Labels `y_ancestry`, `y_population`, `y_sex` on `individual`.

**Where the phenotype connects:** the phenotype file and the node matrix share the sample id (`HG00096` is a
column header in one and a row key in the other). Phenotypes become node *properties*, never neighbours — if
the label were a neighbour, the GNN would read it directly (that is how a 100 % accuracy is produced by mistake).

## 4. Statistics before any model (`cooccurrence_analysis.py`, `eda.py`)

- For every cluster, a 2 × k contingency test of carrier-status vs ancestry / population / sex; Cramér's V =
  √(χ²/N) for a 2 × k table; Benjamini–Hochberg FDR. Result: 6,470/6,551 clusters (98.8 %) are ancestry-associated
  at FDR 5 % (median V 0.20, max 0.81); 6,378 population-associated; **0 sex-associated** (median V 0.013).
  The sex result is the negative control: chr22 is autosomal, so a method that finds sex signal is broken.
- For every lift edge, the cosine similarity of the two endpoints' ancestry-enrichment profiles (5-vector of
  carrier-fraction / overall-fraction − 1): 0.86 for real edges vs 0.22 for degree-preserving shuffled pairs;
  86 % of edges join clusters enriched in the same ancestry (42 % expected); higher-lift quartiles are more
  similar (0.84 → 0.90). Interpretation: the co-occurrence graph is largely population structure — long-range
  edges (median endpoint distance in the tens of Mb) are ancestry, not physical linkage.
- Graph statistics (`graph_explore.py`, NetworkX): 6,551 nodes, 187,030 edges, largest connected component
  4,327, 2,207 clusters without a lift edge, degree max 358; the top hubs are all AFR-enriched low-support
  clusters — the mega-hub artefact per node.
- EDA (`outputs/eda/chr22/EDA.md`): provenance table, population/sex tables, block length (median 29.7 kb,
  5–95 % 8.9–151 kb), clusters per block (median 214), singleton rate (median 0.62), entropy along the
  chromosome, support distributions, edge lift/distance/degree, gene/protein counts, proteomics missingness
  (7.4 % overall, rising to 16 % for the least abundant proteins — MNAR at the detection limit), between-site
  shift before/after harmonisation, and the cis genotype→protein correlations.

## 5. Baseline (`baseline.py`)

L2 logistic regression on the sparse 0/1 carrier matrix, one model per target, C chosen on validation,
scored on the untouched test split (stratified 70/15/15 over the 2,503 labelled people; the split file is
shared by every later model). Test balanced accuracy: ancestry 0.977, population 0.614 (26 classes), sex 0.463
(chance). This sets the bar the GNN must at least match.

## 6. Embeddings (`train_gnn.py --init`, `embeddings.py`)

Nodes need a starting vector. Options implemented and compared:

- **SVD-32** (default): truncated SVD of the carrier matrix `M ≈ UΣVᵀ`; rows of `UΣ` embed people, rows of
  `VΣ` embed clusters, in one shared space. Unsupervised → cannot leak labels.
- **Node2Vec** on the individual–cluster + cluster–cluster graph (pyg-lib random walks; needs the GPU image).
- **learned**: a free `nn.Embedding` per person.
- **raw**: the person's own 6,551-long carrier row through a linear layer.

Finding: with free learned embeddings the GNN reaches only 0.585 balanced accuracy on ancestry; with SVD
initialisation 0.974. The starting embedding matters more than the architecture. `embeddings.py` measures
each space by silhouette (by ancestry) and 5-NN accuracy: SVD-32 0.06 / 0.90; the ancestry-GNN's hidden layer
**0.70 / 0.97** — the GNN's real contribution is a much better-organised embedding, not higher accuracy.

## 7. The GNN (`train_gnn.py`, `train_gnn_v2.py`)

Per node type a linear projection to hidden size 64, then two rounds of `HeteroConv` (sum over relations):

- `SAGEConv` (mean or sum aggregation) on `carries`, `rev_carries`, `in_block`, `rev_in_block`, `next_block`,
  and in v2 `overlaps`, `encodes` (both directions);
- `GraphConv` with edge weights on `co_occurs` (weight = normalised log lift) and in v2 on `measured` /
  `rev_measured` (weight = harmonised protein z-score).

After each round: LayerNorm, residual connection, ReLU, dropout 0.3. A linear head maps the person's 64-d
vector to class logits. Loss: class-weighted cross-entropy on training people only; Adam 5e-3, weight decay
5e-4; early stopping on validation balanced accuracy (patience 30). Full-batch: one epoch is one pass over
the whole graph (2.4 M carries edges) — 0.95 s on an M2 CPU, 0.10 s on the A100.

What a round of message passing means here: a person's vector becomes a summary of the clusters they carry;
a cluster's vector becomes a summary of its carriers, of the clusters it co-occurs with (weighted by lift) and
of its block; after two rounds a person "sees" the clusters that co-occur with theirs and the block context —
and in v2 the proteins they express and the genes those proteins come from.

Results (test, balanced accuracy): ancestry 0.974 (baseline 0.977), population 0.437 with SVD-32 input →
0.611 with the raw carrier row (baseline 0.614), sex 0.503 (chance). Honest reading: on chr22 alone these
labels are essentially linear in the carrier matrix, so the GNN matches but does not beat logistic regression;
it wins on the embedding space.

## 8. Schema v2: joining proteomics (`proteomics_synth_1000g.py`, `haplokg_proteins.py`, `build_kg_v2.py`)

Synthetic proteomics keyed to the **real 1000G ids** (2,503 people, 460 chr22 proteins), with a saved ground
truth so recovery can be scored: 3 sites assigned at random *within* each ancestry (mixed-ancestry sites →
`site` is a pure batch label); age, sex (real), a binary phenotype whose logit is a weighted sum over 20
"causal" clusters (+ small age term), calibrated to 38 % cases; each causal cluster also shifts one protein
encoded in its block (cis effect, β ~ N(0, 1)); ~15 % of proteins respond to the phenotype (β ~ N(0, 0.5));
age and sex effects; per-site batch shift (sd 0.3 log2); missingness increasing toward the detection limit
(15 % for the least abundant protein). The first version shifted *all* proteins with the phenotype, which made
every model score 1.0 — the same failure mode as an earlier team result; that is why the signal is now sparse.

Graph additions: `Gene` and `Protein` nodes from the UniProt BED (isoforms collapsed); `Block —OVERLAPS→ Gene`
by coordinate intersection (1,063 edges; 29 genes fall outside any block, 117 span two); `Gene —ENCODES→ Protein`;
`Individual —MEASURED{log2, z}→ Protein` (1,065,712 edges). **Harmoniser**: robust z per (site, protein),
`(x − median)/(1.4826·MAD)`, computed over each site's own samples before anything crosses sites; it removes the
between-site median shift entirely (0.48 → 0.00) while 227 proteins keep a phenotype association at FDR 5 %.
Detection-limit missingness stays missing — no edge — never imputed as zero.

## 9. Integration experiments (`train_gnn_v2.py`, A100)

Same split, same test people, held-out AUC for the synthetic phenotype:

| modality | input to the person node | AUC | bal. acc. |
|---|---|---|---|
| genome | SVD-32 or raw carrier row; genome relations only | 0.60–0.63 | 0.57–0.62 |
| proteome | harmonised z (460) + observed mask (460); MLP, no graph | 0.96 | 0.96 |
| **both** | SVD/raw + z + mask; all relations incl. `measured`, `overlaps`, `encodes` | **0.99** | **0.97** |
| site (batch control, both) | — | — | 0.30 (chance 0.33) |
| ancestry (both) | — | — | 0.90 |

So the graph adds information on top of the proteome. Saliency (gradient of the case logit w.r.t. the raw
carrier row, averaged over test cases) puts 3–4 of the 20 ground-truth causal clusters in its top 20
(chance 1.2). Genome → proteome as a regression from the person's embedding fails (R² ≈ 0), while a
per-protein ridge on the carrier row recovers the strongest cis effects (4/20 cis proteins at test R² > 0.1,
0/440 others; ceiling from the data itself: r² up to 0.54). Conclusion for the paper: the GNN is the
integration/embedding tool; cis discovery wants sparse per-protein models or a pQTL edge prior.

## 10. Decoder (`graphrag_decoder.py`)

For one person, deterministic retrieval from the graph: profile (true phenotype withheld), the GNN prediction,
globally salient clusters they carry, their 8 most ancestry-informative clusters with block/genes/proteins,
their 8 most extreme protein levels with the encoding block and whether it holds a notable cluster, and their
5 nearest neighbours in the GNN embedding. Serialised as JSON (~4.5 k tokens) to an OpenAI-compatible NIM
endpoint — `nvidia/nemotron-3-super-120b-a12b`, `reasoning_effort=none` (without it the reasoning model thinks
inline and exhausts the token budget before the JSON) — under a system prompt that forbids inventing entities
and requires ids to be cited verbatim; the reply's `cited_ids` are checked against the context (25 cited, 0
unknown for HG00103). Output: summary, ancestry and phenotype assessments, genome↔proteome links, caveats.

## 11. Compute, packaging, deployment

CUDA image `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime` + torch_geometric 2.8.0 + pyg-lib + nx-cugraph +
torch-tensorrt 2.14 (the PyG docs' deprecation of torch_cluster in favour of pyg-lib was the cause of the first
build failure). Runs on CPU too. Neo4j 5.26 community via docker compose for browsing the graph. Brev:
`brev_deploy.sh` creates/uses an instance, uploads code + data, builds natively, runs, copies outputs back;
executed on an L4 (GCP) and an A100 80 GB (Crusoe). Inference: 36 ms per full graph eager → 4.9 ms with
`torch.compile` (identical logits); Torch-TensorRT compilation of this scatter-heavy hetero-GNN did not
finish in 3 h and is not claimed. Everything is `make`-driven from a clone; secrets via environment only.

## 12. What the federated step (RQ3) will do

Sites = the 3 mixed-ancestry sites; each holds its Individual nodes with their CARRIES and MEASURED edges;
the cluster/block/gene/protein graph and the encoder weights are shared. NVFlare FedAvg: each round every
site trains `train_gnn_v2.py`'s model on its subgraph for a few epochs and sends weights; the server averages
by sample count and returns the global model. Report: central vs federated on the same held-out people, plus
the `site` control. Exchanging embeddings of people would be a leak; exchanging weights is not.
