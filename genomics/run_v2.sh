#!/usr/bin/env bash
# Schema v2 chain: proteomics on 1000G IDs -> genes/proteins in the graph -> genome+proteome GNN ablations -> EDA.
# Run after run_all.sh (needs outputs/kg/<chrom> and the shared split).  Env: CHROM (chr22), PYTHON.
set -euo pipefail
cd "$(dirname "$0")"
. ./load_env.sh                                        # NVIDIA_API_KEY etc. from .env / ~/.progenome.env
CHROM="${CHROM:-chr22}"
PY="${PYTHON:-python}"

echo "== v2.1 synthetic proteomics on 1000G IDs (3 mixed-ancestry sites, saved ground truth)"
$PY proteomics_synth_1000g.py --chrom "$CHROM"
echo "== v2.2 genes + proteins + MEASURED edges (harmonised) -> hetero_v2.pt"
$PY build_kg_v2.py --chrom "$CHROM"
echo "== v2.3 EDA report"
$PY eda.py --chrom "$CHROM"
echo "== v2.4 phenotype: genome vs proteome vs both"
for m in genome proteome both; do $PY train_gnn_v2.py --chrom "$CHROM" --target phenotype --modality "$m"; done
for m in genome both; do $PY train_gnn_v2.py --chrom "$CHROM" --target phenotype --modality "$m" --init raw; done   # raw carrier row: saliency vs ground truth
echo "== v2.5 controls: site (batch) and ancestry on the full graph"
$PY train_gnn_v2.py --chrom "$CHROM" --target site --modality both
$PY train_gnn_v2.py --chrom "$CHROM" --target ancestry --modality both
echo "== v2.6 genome -> proteome: per-protein ridge baseline (the honest cis test)"
$PY proteome_linear_baseline.py --chrom "$CHROM"
echo "== v2.7 GraphRAG decoder (dry run; set NVIDIA_API_KEY to call the LLM)"
WHO=$(awk -F, 'NR==2{print $1}' "outputs/gnn_v2/$CHROM/phenotype_both_raw/test_predictions.csv")
$PY graphrag_decoder.py --chrom "$CHROM" --individual "$WHO" --run phenotype_both_raw --dry-run > /dev/null
if [ -n "${NVIDIA_API_KEY:-}" ]; then $PY graphrag_decoder.py --chrom "$CHROM" --individual "$WHO" --run phenotype_both_raw; fi
echo "done -> outputs/{proteomics_synth,kg,eda,gnn_v2,graphrag}/$CHROM"
