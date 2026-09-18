# Clone-and-go entry points for the genomics pipeline.  `make help` lists them.
CHROM ?= chr22
INIT  ?= svd
PY    := .venv/bin/python

.PHONY: help setup data kg analysis baseline graph gnn embeddings run run-v2 eda decode federated test docker docker-run docker-run-v2 docker-federated docker-shell config neo4j neo4j-load neo4j-down brev report clean

help:            ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

setup:           ## create .venv with torch (CPU or CUDA) + pinned deps, run unit tests
	bash setup.sh

data:            ## download the HaploGraph, phenotypes and block stats for $(CHROM)
	bash fetch_data.sh $(CHROM)

kg: data         ## build the knowledge graph (tables, sparse carriers, PyG HeteroData)
	$(PY) build_kg.py --chrom $(CHROM)

analysis: kg     ## cluster/edge/block vs phenotype co-occurrence tests + plots
	$(PY) cooccurrence_analysis.py --chrom $(CHROM)

baseline: kg     ## logistic-regression baseline, writes the shared train/val/test split
	$(PY) baseline.py --chrom $(CHROM)

graph: analysis  ## NetworkX stats, GraphML export and plots of the graph
	$(PY) graph_explore.py --chrom $(CHROM)

gnn: baseline    ## train the hetero-GNN on ancestry, population and sex (control)
	for t in ancestry population sex; do $(PY) train_gnn.py --chrom $(CHROM) --target $$t --init $(INIT); done

embeddings: gnn  ## evaluate/plot SVD and GNN embeddings
	$(PY) embeddings.py --chrom $(CHROM)

run:             ## the whole pipeline (same as run_all.sh)
	PYTHON=$(PY) CHROM=$(CHROM) INIT=$(INIT) bash run_all.sh

run-v2: run      ## schema v2: proteomics on 1000G IDs, genes/proteins, genome+proteome GNN, EDA, decoder dry-run
	PYTHON=$(PY) CHROM=$(CHROM) bash run_v2.sh

eda:             ## exploratory data analysis report -> outputs/eda/$(CHROM)/EDA.md
	$(PY) eda.py --chrom $(CHROM)

decode:          ## GraphRAG insight for one person via NVIDIA NIM (needs NVIDIA_API_KEY; WHO=HG00103)
	$(PY) graphrag_decoder.py --chrom $(CHROM) --individual $(or $(WHO),HG00103) --run phenotype_both_raw

federated:       ## NVFlare FedAvg over 3 sites (simulator), score the global model centrally, compare with each site alone
	$(PY) federated/job.py --chrom $(CHROM) --rounds $(or $(ROUNDS),10) --local-epochs $(or $(LOCAL_EPOCHS),5)
	$(PY) federated/evaluate_global.py --chrom $(CHROM)
	$(PY) federated/local_only.py --chrom $(CHROM) --steps $$(( $(or $(ROUNDS),10) * $(or $(LOCAL_EPOCHS),5) ))

test:            ## unit tests
	$(PY) -m pytest tests -q

DOCKER_RUN = docker run --rm $$(command -v nvidia-smi >/dev/null 2>&1 && echo --gpus all) $$( [ -f .env ] && echo --env-file .env ) \
	  -e CHROM=$(CHROM) -e INIT=$(INIT) -v $(PWD)/data:/app/genomics/data -v $(PWD)/outputs:/app/genomics/outputs progenome-genomics

docker:          ## build the CUDA image from the repository root (runs on CPU too); proteomics inputs are baked in
	docker build -f Dockerfile -t progenome-genomics ..

docker-run:      ## v1 pipeline inside the image (GPU if available; .env passed if present)
	$(DOCKER_RUN) run_all.sh

docker-run-v2:   ## v2 chain inside the image (after docker-run)
	$(DOCKER_RUN) run_v2.sh

docker-federated: ## NVFlare FedAvg inside the image (after docker-run-v2)
	$(DOCKER_RUN) -c "python federated/job.py --chrom $(CHROM) --rounds $(or $(ROUNDS),30) --local-epochs $(or $(LOCAL_EPOCHS),5) && python federated/evaluate_global.py --chrom $(CHROM) && python federated/local_only.py --chrom $(CHROM) --steps $$(( $(or $(ROUNDS),30) * $(or $(LOCAL_EPOCHS),5) ))"

docker-shell:    ## interactive shell inside the image with data/ and outputs/ mounted
	$(DOCKER_RUN:--rm=--rm -it)

config:          ## show the effective configuration (.env / ~/.progenome.env / defaults), secrets masked
	$(PY) config.py

neo4j:           ## start Neo4j community at http://localhost:7474 (neo4j / progenome)
	docker compose up -d neo4j

neo4j-load: neo4j ## load the knowledge graph into Neo4j
	$(PY) neo4j_load.py --chrom $(CHROM)

neo4j-down:      ## stop Neo4j (keeps its data volume)
	docker compose down

brev:            ## create/use a Brev GPU instance, build there, run with Node2Vec + TensorRT benchmark, copy outputs back
	bash brev_deploy.sh

report:          ## rebuild docs/report (figures -> .tex + .docx -> .pdf); needs node with the docx package and tectonic
	$(PY) docs/report/make_report_figures.py
	cd docs/report && node build_report.js && (command -v tectonic >/dev/null && tectonic ProGenome_KT.tex || echo "tectonic not installed: compile ProGenome_KT.tex with pdflatex/xelatex")

clean:           ## remove generated outputs (keeps downloaded data)
	rm -rf outputs
