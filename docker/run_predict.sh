#!/usr/bin/env bash
# Predict with a trained checkpoint. Mount your data and assets read-only.
set -euo pipefail
ASSETS=${ASSETS:-/path/to/duetscreen-assets}      # NTv3_8M_pre/, protein_seg_cache_35m.pkl, checkpoints
DATA=${DATA:-$(pwd)/examples}                       # folder holding the FASTA
OUT=${OUT:-$(pwd)/out}
mkdir -p "$OUT"
docker run --rm --gpus all \
  -v "$ASSETS:/assets:ro" -v "$DATA:/data:ro" -v "$OUT:/out" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl \
  --entrypoint python duetscreen-release:1.0 \
  -m duetscreen.predict \
  --fasta /data/quickstart.fasta \
  --model /assets/checkpoints/seed42/best_nucleotide_v3_mamba_model.pth \
  --output-dir /out --threshold 0.5 --batch-size 8
