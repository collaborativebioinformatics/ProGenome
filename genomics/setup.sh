#!/usr/bin/env bash
# One-shot local environment: creates ./.venv with torch (CPU by default, CUDA if a GPU is present) + pinned deps.
#   bash setup.sh            # auto: CUDA wheels if nvidia-smi works, else CPU wheels
#   TORCH_INDEX=cpu bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"
TORCH_VERSION="${TORCH_VERSION:-2.14.0}"
if [ -z "${TORCH_INDEX:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then TORCH_INDEX=cu126; else TORCH_INDEX=cpu; fi
fi
PY="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.13 .venv >/dev/null
  uv pip install --python .venv/bin/python "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
  uv pip install --python .venv/bin/python -r requirements.txt
  if [ "$TORCH_INDEX" != cpu ]; then
    uv pip install --python .venv/bin/python pyg_lib -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_INDEX}.html" || echo "(no pyg_lib wheel -> Node2Vec falls back to SVD)"
  fi
else
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python -c "import torch, torch_geometric; print('torch', torch.__version__, '| pyg', torch_geometric.__version__, '| cuda', torch.cuda.is_available())"
.venv/bin/python -m pytest tests -q
echo "ready: source .venv/bin/activate  (or just use: make run)"
