#!/usr/bin/env bash
# Deploy and run the pipeline on an NVIDIA Brev GPU instance, then pull the outputs back.
#   bash brev_deploy.sh                          # instance $BREV_INSTANCE (default progenome-gpu); created if missing
#   BREV_INSTANCE=progenome-a100 bash brev_deploy.sh
#   STAGE=infer bash brev_deploy.sh              # only the inference benchmark on an existing run
#   INIT=node2vec bash brev_deploy.sh            # Node2Vec instead of SVD starting embeddings (slightly weaker: ancestry 0.95 vs 0.97)
# Needs: brev CLI logged in (brev login --api-key ...).  `brev refresh` writes an SSH alias named after the
# instance; this script drives the box with plain ssh/scp through that alias (brev exec is interactive-prone).
set -euo pipefail
cd "$(dirname "$0")"
. ./load_env.sh                                        # BREV_INSTANCE / BREV_TYPE / INIT from .env
INSTANCE="${BREV_INSTANCE:-progenome-gpu}"
TYPE="${BREV_TYPE:-g2-standard-4:nvidia-l4:1}"     # L4 24 GB (works without extra cloud credentials)
STAGE="${STAGE:-all}"                               # all | infer
INIT="${INIT:-svd}"                                  # svd is the documented best; INIT=node2vec needs pyg_lib (in the image), 50 pretraining epochs
REMOTE=/home/ubuntu/progenome
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
SCP="scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

if ! brev ls 2>/dev/null | grep -qE "^\s*${INSTANCE}\s"; then
  echo "== creating ${INSTANCE} (${TYPE})"
  brev create "${INSTANCE}" --type "${TYPE}" --min-disk 100 --timeout 900
fi
echo "== waiting for ${INSTANCE} shell"
until brev ls 2>/dev/null | grep -qE "^\s*${INSTANCE}\s+RUNNING\s+\S+\s+READY"; do sleep 15; done
brev refresh >/dev/null 2>&1 || true
$SSH "${INSTANCE}" 'nvidia-smi -L; docker --version'

if [ "$STAGE" = all ]; then
  echo "== upload code + data"
  # genomics/ plus the two small proteomics inputs the v2 schema needs (protein BED, gene symbols)
  tar czf /tmp/genomics_upload.tgz --exclude=outputs --exclude=outputs_brev --exclude='__pycache__' --exclude=.venv --exclude=.pytest_cache \
      -C .. genomics proteomics/uniprot_chr22.bed proteomics/synthetic_proteomics_chr22/gene_symbol_cache.csv
  $SCP /tmp/genomics_upload.tgz "${INSTANCE}:/tmp/genomics_upload.tgz"
  $SSH "${INSTANCE}" "mkdir -p ${REMOTE} && tar xzf /tmp/genomics_upload.tgz -C ${REMOTE} && ls ${REMOTE}/genomics | head -3"

  echo "== build image on the GPU box (native, no emulation)"
  $SSH "${INSTANCE}" "cd ${REMOTE} && docker build -f genomics/Dockerfile -t progenome-genomics . 2>&1 | grep -E '^#[0-9]+ (DONE|ERROR)|passed|failed|error|Successfully|naming to' | tail -20"

  echo "== run the full pipeline on the GPU (INIT=${INIT})"
  $SSH "${INSTANCE}" "cd ${REMOTE}/genomics && mkdir -p outputs && docker run --rm --gpus all -e INIT=${INIT} \
      -v ${REMOTE}/genomics/data:/app/genomics/data -v ${REMOTE}/genomics/outputs:/app/genomics/outputs progenome-genomics run_all.sh"
fi

echo "== inference benchmark: eager vs Torch-TensorRT"
RUN_NAME=$($SSH "${INSTANCE}" "ls ${REMOTE}/genomics/outputs/gnn/chr22 2>/dev/null | grep -E '^ancestry_(node2vec|svd)$' | head -1")
$SSH "${INSTANCE}" "cd ${REMOTE}/genomics && docker run --rm --gpus all \
    -v ${REMOTE}/genomics/data:/app/genomics/data -v ${REMOTE}/genomics/outputs:/app/genomics/outputs progenome-genomics \
    -c 'python infer.py --run ${RUN_NAME} --compile none && python infer.py --run ${RUN_NAME} --compile inductor && python infer.py --run ${RUN_NAME} --compile tensorrt --precision fp16'"

echo "== copy outputs back to outputs_brev/${INSTANCE}/"
mkdir -p "outputs_brev/${INSTANCE}"
$SCP -r "${INSTANCE}:${REMOTE}/genomics/outputs/." "outputs_brev/${INSTANCE}/"
echo "done. Stop billing when finished:  brev stop ${INSTANCE}   (delete: brev delete ${INSTANCE})"
