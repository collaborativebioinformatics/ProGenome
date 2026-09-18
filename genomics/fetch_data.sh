#!/usr/bin/env bash
# Download the chr22 HaploGraph (pre-built 1000G haploblock-cluster graph),
# the real 1000G phenotype labels and the per-block statistics into data/.
# Source: https://data.haploblocks.org (built for this hackathon, see
# haplograph/1000G/README.txt on that server). Idempotent; verifies md5.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/load_env.sh"                                  # HAPLOBLOCKS_BASE, CHROM from .env if set
CHR="${1:-${CHROM:-chr22}}"
BASE="${HAPLOBLOCKS_BASE:-https://data.haploblocks.org}"
DATA="$HERE/data"

mkdir -p "$DATA/haplograph/$CHR" "$DATA/haploblocks"

fetch() {  # fetch <url> <dest>
  if [ -s "$2" ]; then echo "have   $2"; else echo "fetch  $1"; curl -sSL --fail --max-time 600 -o "$2" "$1"; fi
}

for f in nodes.csv.gz edges_lift_above_threshold.csv.gz islands.csv.gz top_edges_by_lift.csv.gz; do
  fetch "$BASE/haplograph/1000G/$CHR/$f" "$DATA/haplograph/$CHR/$f"
done
fetch "$BASE/haplograph/1000G/phenotypes_real.csv" "$DATA/haplograph/phenotypes_real.csv"
fetch "$BASE/haplograph/1000G/README.txt"          "$DATA/haplograph/README.txt"
fetch "$BASE/haplograph/1000G/checksums.md5"       "$DATA/haplograph/checksums.md5"
fetch "$BASE/bidirectional_blast_samples/block_stats.tsv" "$DATA/haploblocks/block_stats.tsv"
fetch "$BASE/haploblock_hashes/1000G/$CHR/${CHR}_haploblock_boundaries_${CHR}.tsv" \
      "$DATA/haploblocks/${CHR}_haploblock_boundaries_${CHR}.tsv"

# md5 verification of the graph files (checksums.md5 uses ./chrN/file paths)
cd "$DATA/haplograph"
if command -v md5sum >/dev/null; then
  grep -E "^\S+  \./$CHR/(nodes|edges_lift_above_threshold|islands|top_edges_by_lift)\.csv\.gz$" checksums.md5 | md5sum -c -
else  # macOS without coreutils
  grep -E "\./$CHR/(nodes|edges_lift_above_threshold|islands|top_edges_by_lift)\.csv\.gz$" checksums.md5 | while read -r sum path; do
    [ "$sum" = "$(md5 -q "$path")" ] && echo "$path: OK" || { echo "$path: MISMATCH"; exit 1; }
  done
fi
echo "done -> $DATA"
