#!/usr/bin/env bash
# One-step training sanity check (forward + backward) inside the container.
set -euo pipefail
ASSETS=${ASSETS:-/path/to/duetscreen-assets}
WORK=${WORK:-$(pwd)/train_smoke}
mkdir -p "$WORK"
docker run --rm --gpus all \
  -v "$ASSETS:/assets:ro" -v "$WORK:/work" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl \
  duetscreen-release:1.0 \
  --smoke \
  --records /assets/mini_train_cache.pkl \
  --output-dir /work/smoke \
  --ntv3-model /assets/NTv3_8M_pre
