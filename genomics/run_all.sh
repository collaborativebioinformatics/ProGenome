#!/usr/bin/env bash
# Whole pipeline for one chromosome.  Env: CHROM (chr22), INIT (svd|node2vec|learned), TARGETS.
set -euo pipefail
cd "$(dirname "$0")"
. ./load_env.sh
CHROM="${CHROM:-chr22}"
INIT="${INIT:-svd}"
TARGETS="${TARGETS:-ancestry population sex}"
PY="${PYTHON:-python}"

echo "== 1. fetch";            bash fetch_data.sh "$CHROM"
echo "== 2. knowledge graph";  $PY build_kg.py --chrom "$CHROM"
echo "== 3. co-occurrence";    $PY cooccurrence_analysis.py --chrom "$CHROM"
echo "== 4. baseline";         $PY baseline.py --chrom "$CHROM"
echo "== 5. graph display";    $PY graph_explore.py --chrom "$CHROM"
for target in $TARGETS; do
  echo "== 6. GNN ($target, init=$INIT)"; $PY train_gnn.py --chrom "$CHROM" --target "$target" --init "$INIT"
done
echo "== 7. embeddings";       $PY embeddings.py --chrom "$CHROM"
echo "done -> outputs/"
