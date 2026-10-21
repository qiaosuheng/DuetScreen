#!/usr/bin/env bash
# Full training on a prepared record cache (see README for the cache schema).
set -euo pipefail
ASSETS=${ASSETS:-/path/to/duetscreen-assets}
WORK=${WORK:-$(pwd)/runs}
mkdir -p "$WORK"
docker run --rm --gpus all \
  -v "$ASSETS:/assets:ro" -v "$WORK:/work" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl \
  duetscreen-release:1.0 \
  --records /work/records.pkl \
  --output-dir /work/seed42 \
  --ntv3-model /assets/NTv3_8M_pre \
  --seed 42 --epochs 20 --patience 5 --batch-size 16 --max-seq-len 2048
