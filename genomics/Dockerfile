# ProGenome genomics pipeline: knowledge graph -> co-occurrence analysis -> GNN (PyTorch Geometric).
# CUDA image for NVIDIA Brev / any GPU box; the same image runs on CPU when no GPU is present.
#
#   Build from the REPOSITORY ROOT so the two proteomics inputs are baked in (make docker does this):
#     docker build -f genomics/Dockerfile -t progenome-genomics .
#   Run (data/ and outputs/ bind-mounted; .env passed for the decoder):
#     docker run --rm --gpus all --env-file genomics/.env -v $PWD/genomics/data:/app/genomics/data -v $PWD/genomics/outputs:/app/genomics/outputs progenome-genomics run_all.sh
#     docker run --rm --gpus all -v ... progenome-genomics run_v2.sh
#     docker run --rm --gpus all -v ... progenome-genomics -c "python train_gnn.py --target ancestry --init node2vec"
ARG BASE=pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
FROM ${BASE}
ARG PYG_WHEELS=https://data.pyg.org/whl/torch-2.14.0+cu126.html

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 NX_CUGRAPH_AUTOCONFIG=True
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app/genomics
COPY genomics/requirements.txt .
RUN pip install -r requirements.txt \
 && (pip install pyg_lib -f ${PYG_WHEELS} \
     || echo "no pyg_lib wheel for this torch build -> Node2Vec falls back to SVD") \
 && (pip install nx-cugraph-cu12 --extra-index-url https://pypi.nvidia.com \
     || echo "nx-cugraph not installed -> NetworkX runs on CPU") \
 && (pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu126 \
     || echo "torch-tensorrt not installed -> infer.py --compile tensorrt falls back to inductor")

COPY proteomics/uniprot_chr22.bed /app/proteomics/uniprot_chr22.bed
COPY proteomics/synthetic_proteomics_chr22/gene_symbol_cache.csv /app/proteomics/synthetic_proteomics_chr22/gene_symbol_cache.csv
COPY genomics/ /app/genomics/
RUN python -m pytest tests -q

# data/ and outputs/ are bind-mounted at run time (see docker-compose.yml)
ENTRYPOINT ["bash"]
CMD ["run_all.sh"]
