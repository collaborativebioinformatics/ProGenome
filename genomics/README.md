# genomics/ — haploblock knowledge graph → phenotypes → proteomics → GNN → LLM

[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.8-3C2179)](https://pyg.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![NVIDIA FLARE](https://img.shields.io/badge/NVIDIA%20FLARE-2.9%20FedAvg-76B900?logo=nvidia&logoColor=white)](https://github.com/NVIDIA/NVFlare)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-Nemotron%203%20Super-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![NVIDIA Brev](https://img.shields.io/badge/NVIDIA%20Brev-A100%2080GB-76B900?logo=nvidia&logoColor=white)](https://brev.nvidia.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.26%20community-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.6%20%2B%20nx--cugraph-1B6AC6)](https://networkx.org/)
[![Data](https://img.shields.io/badge/Data-1000G%20HaploGraph%20chr22-0E7C7B)](https://data.haploblocks.org/haplograph/1000G/)
[![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![Hackathon](https://img.shields.io/badge/Nordic%20Biobank%20x%20NVIDIA-Federated%20Learning%20Hackathon%202026-5A9E3F)](https://github.com/collaborativebioinformatics/ProGenome)

The genome side of ProGenome, plus the join to proteomics. It takes the **published 1000 Genomes
HaploGraph** (built for this hackathon at Rigshospitalet, <https://data.haploblocks.org/haplograph/1000G/>),
connects it to the **real 1000G labels** (ancestry, population, sex) and to **per-site proteomics keyed by the
same sample IDs**, trains a heterogeneous **PyTorch Geometric GNN**, and decodes a person's graph neighbourhood
with an **LLM (NVIDIA NIM)** into a cited insight. Everything runs from a clone; chr22 takes ~10 min on a laptop CPU
and ~2 min on an A100.

Presentation slides: [Team 3: ProGenome (Google Slides)](https://docs.google.com/presentation/d/13wHiF-xPHeDyy7XYMmTSwxAYUswgSutH7x7dZ02SOOs/edit) · Results page: [RESULTS.md](RESULTS.md) ·
Architecture page (data flow, schema, federated topology, stack): `docs/architecture.html` ·
Mermaid source for Lucidchart / GitHub: `docs/architecture.mmd`.

## Quick start

```bash
git clone https://github.com/collaborativebioinformatics/ProGenome.git
cd ProGenome && git checkout modelling && cd genomics
make setup        # .venv with torch (CPU, or CUDA if nvidia-smi works) + pinned deps, runs the unit tests   (~30 s with uv)
make run          # v1: fetch -> knowledge graph -> co-occurrence analysis -> baseline -> graph plots -> GNN -> embeddings  (~6 min CPU)
make run-v2       # v2: proteomics on 1000G IDs -> genes/proteins in the graph -> EDA -> genome+proteome GNN -> ridge -> decoder  (~19 min CPU)
make federated ROUNDS=30 LOCAL_EPOCHS=5   # NVFlare FedAvg over 3 sites + central scoring + each site alone  (~4 min CPU)
make decode WHO=HG00103                   # LLM insight for one person via NVIDIA NIM (needs an API key, step 4 below)
make neo4j-load   # browse the graph at http://localhost:7474  (neo4j / progenome)
make docker       # CUDA image (pytorch 2.14 + cu12.6, PyG 2.8, pyg-lib, nx-cugraph, torch-tensorrt); runs on CPU too
make brev         # create/use a Brev GPU instance, build there, run everything, copy outputs back
make help         # every target
```

## Implementation guide: from clone to every result

Everything below was executed end-to-end on 18 Sept 2026 from an empty folder on a laptop (Apple M2, CPU only, 30 min)
and on an A100 through the Docker image (14 min). Numbers you should see are in the *Results* section; the tolerance
between runs is stated there.

### 0. What you need

| Need | Details |
|---|---|
| OS, Python | macOS or Linux; Python 3.10-3.13 (3.13 tested). `uv` is optional and makes `make setup` take 30 s instead of minutes. |
| Tools | git, curl, make, bash. `md5sum` or macOS `md5` for the download check. |
| Resources | ~8 GB RAM, ~2 GB disk (15 MB download, ~400 MB of outputs). No GPU needed for chr22. |
| Network | data.haploblocks.org (the graph), download.pytorch.org and PyPI (setup), integrate.api.nvidia.com (decoder only). |
| Optional GPU | NVIDIA driver with CUDA 12.6: `setup.sh` picks CUDA wheels automatically. Docker + NVIDIA Container Toolkit for the image. |
| Optional cloud | NVIDIA Brev CLI and account for `make brev`. |
| Optional LLM | An NVIDIA API key (free tier is enough) for `make decode`; without it the decoder prints the prompt and stops. |

### 1. Clone and set up

```bash
git clone https://github.com/collaborativebioinformatics/ProGenome.git
cd ProGenome && git checkout modelling && cd genomics
make setup
```

`setup.sh` creates `./.venv`, installs torch 2.14.0 from the CPU index (or cu126 if `nvidia-smi` works), the pinned
`requirements.txt`, then runs the 11 unit tests and prints the torch / PyG / CUDA versions. Force a variant with
`TORCH_INDEX=cpu bash setup.sh` or `PYTHON=python3.12 bash setup.sh`. Every later command uses `.venv/bin/python` through
the Makefile, so nothing needs activating.

### 2. Genome side (v1): `make run`

Runs `fetch_data.sh` (md5-verified download of the chr22 HaploGraph, phenotypes and block statistics into `data/`),
`build_kg.py`, `cooccurrence_analysis.py`, `baseline.py`, `graph_explore.py`, `train_gnn.py` for ancestry, population
and sex, and `embeddings.py`. Check these files when it finishes:

| File | What to expect |
|---|---|
| `outputs/kg/chr22/summary.json` | 2,548 individuals, 6,551 kept clusters, 2,365,574 CARRIES, 187,030 CO_OCCURS, 0 edges dropped |
| `outputs/cooccurrence/chr22/summary.json` | 6,470 ancestry-associated clusters, 0 sex-associated, edge cosine 0.86 vs 0.22 |
| `outputs/baseline/chr22/metrics.json` | test balanced accuracy ancestry 0.977, population 0.614, sex 0.463 |
| `outputs/gnn/chr22/ancestry_svd/metrics.json` | 0.974 (population_svd 0.437, sex_svd ~0.50); `history.csv`, `test_predictions.csv`, embeddings |
| `outputs/embeddings/chr22/embedding_quality.json` | silhouette by ancestry 0.06 (SVD) -> 0.70 (GNN) |

Options: `CHROM=chr21 make run` (any chromosome on the server), `INIT=raw make gnn` (raw carrier row; population 0.611),
`INIT=node2vec` (needs pyg-lib, i.e. the GPU image; `--node2vec-epochs`, default 50).

### 3. Proteomics integration (v2): `make run-v2`

Runs `proteomics_synth_1000g.py` (synthetic proteomics on the real 1000G ids, 3 sites, `ground_truth.json`),
`build_kg_v2.py`, `eda.py`, `train_gnn_v2.py` for genome / proteome / both (SVD and raw input), the site and ancestry
controls, `proteome_linear_baseline.py`, and the decoder in dry-run mode (plus one real call if a key is configured).

| File | What to expect |
|---|---|
| `outputs/kg/chr22/summary_v2.json` | 458 genes, 460 proteins, 1,063 block-gene overlaps, 1,065,712 MEASURED edges |
| `outputs/eda/chr22/EDA.md` | the 8-section report with tables and plots |
| `outputs/gnn_v2/chr22/phenotype_{genome,proteome,both}_svd/metrics.json` | AUC ~0.64 / 0.96 / 0.99 |
| `outputs/gnn_v2/chr22/phenotype_both_raw/metrics.json` | AUC ~0.99, `saliency.hits_in_ground_truth` 4 of 20, `saliency_top100.csv` |
| `outputs/gnn_v2/chr22/site_both_svd/metrics.json` | balanced accuracy ~0.30-0.34 (chance 0.33: the batch control) |
| `outputs/gnn_v2/chr22/proteome_ridge_baseline/metrics.json` | 4 of 20 cis proteins with test R2 > 0.1, 0 of 440 others |

### 4. Configuration, credentials and the LLM decoder (NVIDIA NIM)

All settings enter the code through **`config.py`**; scripts never read credentials on their own. Precedence:
variables exported in your shell > `genomics/.env` > `~/.progenome.env` > defaults.

```bash
cp .env.example .env      # template with every variable and a comment; .env is git-ignored
make config               # prints what is in effect and where each value came from (secrets masked)
```

| Variable | Used by | Default |
|---|---|---|
| `NVIDIA_API_KEY` | `graphrag_decoder.py` (`make decode`, the real call in `run_v2.sh`) | none: required for the decoder |
| `NIM_MODEL`, `NIM_URL` | decoder model id and endpoint (any OpenAI-compatible chat server) | `nvidia/nemotron-3-super-120b-a12b`, `https://integrate.api.nvidia.com/v1/chat/completions` |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | `neo4j_load.py` | `bolt://localhost:7687`, `neo4j`, `progenome` (matches docker-compose.yml) |
| `HAPLOBLOCKS_BASE`, `CHROM` | `fetch_data.sh`, every stage | `https://data.haploblocks.org`, `chr22` |
| `PROGENOME_DATA_DIR`, `PROGENOME_OUTPUTS_DIR` | move data / outputs elsewhere | `genomics/data`, `genomics/outputs` |
| `BREV_INSTANCE`, `BREV_TYPE` | `brev_deploy.sh` | `progenome-gpu`, `g2-standard-4:nvidia-l4:1` |

Getting and using the LLM key:

1. Sign in at <https://build.nvidia.com>, open any model page, click *Get API Key*; it starts with `nvapi-`. The free tier
   covers this (one call is ~5,600 tokens, 4-13 s).
2. Put it in `.env` (`NVIDIA_API_KEY=nvapi-...`) or in `~/.progenome.env` outside the repository (`chmod 600`).
3. `make decode WHO=HG00103` (any 1000G id in the graph; people in the held-out split also get the GNN's prediction).
   Output: `outputs/graphrag/chr22/HG00103_insight.json` (summary, ancestry and phenotype assessments, genome-proteome
   links, caveats, cited ids, citation check, token usage) and `HG00103_context.json` (exactly what the model was given).
   `.venv/bin/python graphrag_decoder.py --individual HG00103 --dry-run` prints the prompt without calling anything;
   `--reasoning low|medium|high` turns on Nemotron's separate reasoning field (slower); `--run` picks another GNN run folder.
4. The call is `POST $NIM_URL` with `Authorization: Bearer $NVIDIA_API_KEY` and body `{model, messages, temperature 0.2,
   max_tokens 4000, reasoning_effort "none"}`. A local NIM container, vLLM or any OpenAI-compatible server works by
   changing `NIM_URL` and `NIM_MODEL`; a local NIM on the GPU box keeps patient context on site.

### 5. Federated learning (NVFlare): `make federated ROUNDS=30 LOCAL_EPOCHS=5`

Runs `federated/job.py` (FedAvgRecipe, 3 simulated sites as threads, only weights exchanged), `federated/evaluate_global.py`
(scores `FL_global_model.pt` on the same 376 held-out people as the central model) and `federated/local_only.py`
(each site alone, same number of steps). Expect in `outputs/federated/chr22/evaluation.json` a global AUC of 0.987-0.998
against a central 0.992-0.995, and in `local_only_vs_federated.json` site-alone AUCs of 0.986-0.998. The NVFlare
workspace (logs, per-round metrics, global models) is under `outputs/federated/chr22/workspace/`. Ten rounds are not
enough (AUC 0.955); thirty converge. To run across real machines use NVFlare POC mode with the same `client.py` and
`model.py`; each site needs the shared graph files (`outputs/kg/chr22/`), its own people and the split file.

### 6. Browse the graph: `make neo4j-load`

Starts Neo4j 5.26 community with docker compose and loads the graph (a few minutes for the 2.4 M CARRIES edges). Open
<http://localhost:7474> (user `neo4j`, password `progenome`) and try
`MATCH (i:Individual {id:'HG00096'})-[:CARRIES]->(c:Cluster)-[:IN_BLOCK]->(b:Block) RETURN i,c,b LIMIT 50`.
`make neo4j-down` stops it and keeps the volume.

### 7. GPU: Docker image and Brev

* **Docker is the whole solution in one image.** `make docker` builds `progenome-genomics` from the repository root
  (`docker build -f genomics/Dockerfile ..`) on `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime` with PyG 2.8, pyg-lib,
  nx-cugraph, torch-tensorrt and NVFlare; the two proteomics inputs are baked in and the 11 unit tests run inside the build
  (~10 GB, ~4 min on an A100). Only `data/` (15 MB, downloaded on first run) and `outputs/` are bind-mounted, and `.env`
  is passed in when present. Then, on a Linux box with the NVIDIA Container Toolkit (or CPU-only without `--gpus`):
  ```bash
  make docker-run          # v1 chain           make docker-run-v2      # v2 chain incl. decoder if .env has a key
  make docker-federated    # NVFlare + scoring  make docker-shell       # interactive shell in the image
  docker run --rm --gpus all -v $PWD/data:/app/genomics/data -v $PWD/outputs:/app/genomics/outputs progenome-genomics -c "python train_gnn.py --target ancestry --init node2vec"
  ```
  `docker compose up -d neo4j` adds the graph browser next to it. Not in the image: the report toolchain (Node.js, tectonic)
  and the Brev CLI. On Apple silicon the image builds under emulation and runs CPU-only: use `make setup` there instead.
* **Brev**: `brew install brevdev/homebrew-brev/brev` (or the installer at <https://docs.nvidia.com/brev/cli/getting-started>),
  `brev login --api-key <key from brev.nvidia.com, account settings>`, then `make brev`. `brev_deploy.sh` creates the
  instance if missing (default `g2-standard-4:nvidia-l4:1`, an L4 on GCP; `BREV_INSTANCE=progenome-a100 BREV_TYPE=a100-80gb.1x`
  for an A100 where your account has a provider), waits for it, uploads the code, builds the image natively, runs the v1
  chain and the inference benchmark, and copies everything to `outputs_brev/<instance>/`. Stop billing with
  `brev stop <instance>` (delete with `brev delete`).
* **Inference benchmark**: `.venv/bin/python infer.py --run ancestry_svd --compile inductor` (eager vs `torch.compile`,
  reports the max logit difference); `--compile tensorrt` requires torch-tensorrt (in the image) and did not finish on this
  hetero-GNN.

### 8. The report and the docs

**Presenting or reviewing? Start with [RESULTS.md](RESULTS.md)**: methods and results on one GitHub page with every figure.
`docs/report/ProGenome_KT.pdf` (+ `.docx`, `.tex`) is the full knowledge-transfer document; `docs/DEEP_DIVE.md` has the
formulas and shapes, `docs/METHODS.md` the short account, `docs/architecture.html` / `.mmd` the diagrams. `make report`
regenerates the figures from `outputs*/` and rebuilds the three formats (needs Node.js with the `docx` package on
`NODE_PATH`, and `tectonic` or another LaTeX engine).

### 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `no pyg_lib wheel for this torch build -> Node2Vec falls back to SVD` | Expected on CPU / macOS; `--init node2vec` needs the GPU image. SVD is the default and the better start anyway. |
| torch will not install | Python must be 3.10-3.13: `PYTHON=python3.12 bash setup.sh`; CPU-only box: `TORCH_INDEX=cpu`. |
| `MISMATCH` in the md5 check | A partial download: `rm -rf data && make data`. |
| `NVIDIA_API_KEY is not set` | `cp .env.example .env` and fill it in (or `~/.progenome.env`, or export it); `make config` shows what is loaded. |
| Decoder: `model did not return JSON` | Rare; the raw reply is saved as `<id>_raw.txt`; rerun, or use `--reasoning none` (the default). |
| `brev create` fails with `cloudCredId or workspaceGroupId must be specified` | That instance type has no provider on your account; use the default L4 or a type listed by `brev ls --types`. |
| `brev exec` hangs | Not used: the deploy script talks to the box over `ssh <instance>` (alias written by `brev refresh`). |
| Docker `externally-managed-environment` | Already handled (`PIP_BREAK_SYSTEM_PACKAGES=1` in the Dockerfile). |
| Docker on Apple silicon is very slow / no GPU | Expected (amd64 emulation); use the venv path locally and the image on Linux/Brev. |
| NVFlare run shows `status: None` | Normal for the simulator; look at `evaluation.json` and the workspace logs instead. |
| Different numbers than documented | Counts and statistics must be identical; model scores vary by up to ~0.01 AUC between runs (GPU kernels, FedAvg). |

## How the pieces connect

The HaploGraph file `nodes.csv.gz` has **one row per haploblock cluster and one 0/1 column per 1000G individual**.
`phenotypes_real.csv` and the proteomics matrices are keyed by the **same sample IDs** (`HG00096`, …). That single
string is the whole join:

```
Individual ─CARRIES→ Cluster ─IN_BLOCK→ Block ─OVERLAPS→ Gene ─ENCODES→ Protein
Individual ─MEASURED{log2, z}→ Protein            Cluster ─CO_OCCURS{weight, lift}→ Cluster
```

Phenotypes are *properties* of the Individual node — training targets, never neighbours — so the GNN cannot read
the label off the graph. Individuals being their own node type is what makes the graph federatable: the
cluster/block/gene/protein graph is public and identical at every site; a site holds only its Individual nodes and
their CARRIES / MEASURED edges.

| node / edge (chr22) | from | count |
|---|---|---|
| `Individual` {ancestry, population, sex, site, phenotype} | `phenotypes_real.csv` (2,503 labelled) + proteomics metadata | 2,548 |
| `Cluster` {support, block stats} | `nodes.csv.gz`, symmetric support filter ≥ 25 | 6,551 |
| `Block` {length, n_clusters, entropy, dominance} | `block_stats.tsv` | 669 |
| `Gene` · `Protein` | `../proteomics/uniprot_chr22.bed` (isoforms collapsed) | 458 · 460 |
| `CARRIES` | `nodes.csv.gz` | 2,365,574 |
| `CO_OCCURS` (weight, lift ≥ 5) | `edges_lift_above_threshold.csv.gz` | 187,030 |
| `IN_BLOCK` · `NEXT_BLOCK` · `OVERLAPS` · `ENCODES` | ids / coordinates | 6,551 · 668 · 1,063 · 460 |
| `MEASURED` (harmonised: robust z per site × protein; LOD-missing = no edge) | proteomics matrices | 1,065,712 |

**Controls** are part of the design: `sex` (chr22 is autosomal → must be chance) and `site` (sites are
mixed-ancestry batches → must be chance). Every model is scored on the same seeded 70/15/15 split over individuals.

## Results (chr22, seed 42, held-out test set)

Graph ↔ phenotype (`cooccurrence_analysis.py`): 98.8 % of clusters are ancestry-associated (FDR 5 %), **0 %**
sex-associated; co-occurring clusters share ancestry profiles (cosine 0.86 vs 0.22 for shuffled pairs).

| target | classes | logistic regression | GNN |
|---|---|---|---|
| ancestry (real) | 5 | 0.977 bal-acc | 0.974 (SVD init) |
| population (real) | 26 | 0.614 | 0.611 (`--init raw`) |
| sex — control | 2 | 0.463 | 0.503 |

Genome + proteome (`train_gnn_v2.py`, synthetic phenotype with saved ground truth; AUC on the same test people):

| modality | AUC | balanced accuracy |
|---|---|---|
| genome only | 0.60 | 0.57 |
| proteome only (MLP) | 0.96 | 0.96 |
| **genome + proteome (graph)** | **0.99** | **0.97** |
| site — batch control | — | 0.30 (chance 0.33) |

Genome → proteome: the cis effects are in the data (per-cluster/protein r² up to 0.54, `eda.py` §8) and a per-protein
ridge recovers the strong ones (4/20 cis proteins with test R² > 0.1, 0/440 others; `proteome_linear_baseline.py`),
but the GNN's 64-d embedding does not — the GNN is the integration/embedding tool, cis discovery wants sparse
per-protein models or a pQTL edge prior (UKB-PPP on AWS Open Data is the planned source).

Embeddings: the ancestry-GNN's 64-d space has silhouette 0.70 by ancestry vs 0.06 for plain SVD (`embeddings.py`).
SVD initialisation is essential (free learned embeddings: 0.585). Node2Vec (`--init node2vec`, pyg-lib, 50 pretraining
epochs, A100) is a working but weaker start: ancestry 0.950, population 0.292, sex 0.489; the same 0.70 silhouette.

Compute (A100 80 GB via Brev): GNN epoch 0.10 s (0.95 s on an M2 CPU); inference over the full graph 36 ms eager →
**4.9 ms with `torch.compile`** (7.5×, identical logits). Torch-TensorRT compilation of this scatter-heavy hetero-GNN
did not finish in 3 h and is not reported.

Decoder (`graphrag_decoder.py`, Nemotron 3 Super, `reasoning_effort=none`): ~13 s per person, 25 cited ids, 0 invented.

Federated (`federated/job.py`, NVFlare 2.9 FedAvg, 3 mixed-ancestry sites, each training only on its own people;
only weights exchanged; `make federated`): after 30 rounds × 5 local epochs the global model scores **AUC 0.998 /
balanced accuracy 0.963** on the same 376 held-out people as the central model (0.992 / 0.969); per site 1.000 /
0.999 / 0.997. Ten rounds were not enough (0.955); thirty converge. ~4 min on the M2 CPU.
Site-alone comparison (`federated/local_only.py`, same 150 steps per site): a lone site reaches own-test AUC 0.998 / 0.986 / 0.988 and transfers to the other sites at worst 0.949 / 0.969 / 0.997; the federated model scores 0.999 / 1.000 / 0.968 on the same people. With only 50 steps per site (10 rounds) lone sites reach 0.88-0.92 while the federated model reaches
0.99. Federation costs nothing when a site has enough data and budget, helps when it does not, and in both cases solves the
constraint (one model trained on everyone, no row leaves a site). A larger gain is expected with ancestry-pure sites, the next experiment.

Reproduction (18 Sept): a simulated fresh clone on the laptop (`make setup && make run && make run-v2 && make federated`)
and the Docker image rebuilt on the A100 both reproduce every count exactly and every score within run-to-run noise:
phenotype AUC both 0.994 (laptop) / 0.995 (A100), federated 0.996 / 0.987 vs central 0.994 / 0.995,
inference 36.3 -> 4.9 ms. Details: `docs/report/ProGenome_KT.pdf`, section 13.

## Pipeline

| step | script | writes |
|---|---|---|
| 1 | `fetch_data.sh` | `data/` — HaploGraph files, phenotypes, block stats (md5-verified) |
| 2 | `build_kg.py` (`haplokg.py`) | `outputs/kg/<chr>/` tables, sparse `carries.npz`, PyG `hetero.pt` |
| 3 | `cooccurrence_analysis.py` | `outputs/cooccurrence/<chr>/` |
| 4 | `baseline.py` | `outputs/baseline/<chr>/` + shared split `outputs/splits/` |
| 5 | `graph_explore.py` (NetworkX; cuGraph via `nx-cugraph` in the image) | `outputs/graph/<chr>/` stats, GraphML, plots |
| 6 | `train_gnn.py` (`--init svd|node2vec|learned|raw`) | `outputs/gnn/<chr>/<target>_<init>/` |
| 7 | `embeddings.py` | `outputs/embeddings/<chr>/` |
| v2.1 | `proteomics_synth_1000g.py` | `outputs/proteomics_synth/<chr>/` (+ `ground_truth.json`) |
| v2.2 | `build_kg_v2.py` (`haplokg_proteins.py`) | genes/proteins/measured tables, `hetero_v2.pt` |
| v2.3 | `eda.py` | `outputs/eda/<chr>/EDA.md` + tables + plots |
| v2.4 | `train_gnn_v2.py` (`--modality genome|proteome|both`, `--target phenotype|site|ancestry|sex|proteome`) | `outputs/gnn_v2/<chr>/` |
| v2.5 | `proteome_linear_baseline.py` | `outputs/gnn_v2/<chr>/proteome_ridge_baseline/` |
| v2.6 | `graphrag_decoder.py` | `outputs/graphrag/<chr>/<id>_insight.json` |
| v2.7 | `federated/job.py` → `federated/evaluate_global.py` | `outputs/federated/<chr>/` NVFlare workspace, `evaluation.json` |
| v2.8 | `federated/local_only.py` | `outputs/federated/<chr>/local_only_vs_federated.json` (each site alone vs the global model) |
| — | `infer.py` (`--compile none|inductor|tensorrt`) | inference benchmark |
| — | `neo4j_load.py` + `docker-compose.yml` | Neo4j browser |
| — | `Dockerfile`, `run_all.sh`, `run_v2.sh`, `brev_deploy.sh`, `Makefile`, `setup.sh` | packaging / deploy |

Tests: `make test` (toy graph: parsing, filtering, edge remapping, protein layer, HeteroData).

## GNN

`HeteroConv`, 2 layers, hidden 64, LayerNorm + residual, dropout 0.3: `SAGEConv` on carries / in_block / next_block /
overlaps / encodes (both directions), `GraphConv` with edge weights on co_occurs (normalised log lift) and measured
(harmonised z). Individual input = SVD-32 of the carrier matrix (or the raw carrier row) + the harmonised protein
vector and its observed mask; clusters and blocks add their z-scored statistics. Class-weighted cross-entropy,
Adam 5e-3, early stopping on validation balanced accuracy. With `--init raw`, a gradient saliency ranks clusters and
is scored against the ground-truth causal clusters (precision@20 = 0.20 vs 0.06 by chance).

## Next

* **Federated, next level:** weight FedAvg by site size (`aggregation_weights`), per-site held-out reporting only
  (no central test set), NVFlare POC mode across real machines (the L4 and A100 as two sites).
* **Real proteomics:** Wu et al. 2013 LCL proteomics (95 HapMap individuals with 1000G IDs) for a real join; UKB-PPP
  cis-pQTLs (AWS Open Data) as `Cluster → Protein` propensity edges.
* More chromosomes: `CHROM=chr21 make run`.
