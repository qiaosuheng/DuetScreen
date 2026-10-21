#!/usr/bin/env bash
# One-epoch toy training + prediction check.
#
# Unpack DuetScreen_ToySubset.zip so that this folder contains data/ and cache/.
# Run from the repository root:
#   ASSETS=assets ./toy/run_toy.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-duetscreen-release:1.0}"
ASSETS="${ASSETS:-$ROOT/assets}"
DOCKER="${DOCKER:-docker}"

[ -d "$ASSETS/NTv3_8M_pre" ] || { echo "missing $ASSETS/NTv3_8M_pre"; exit 1; }
[ -f "$HERE/cache/variants_data_toy.pkl" ] || { echo "missing toy cache; unpack DuetScreen_ToySubset.zip into $HERE"; exit 1; }
[ -f "$HERE/cache/protein_seg_cache_toy.pkl" ] || { echo "missing toy protein cache"; exit 1; }
"$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not found; build it with: docker build -t $IMAGE ."; exit 1; }

echo "== 1/2 one-epoch training"
"$DOCKER" run --rm --gpus all \
  -v "$HERE:/toy" -v "$ASSETS/NTv3_8M_pre:/assets/NTv3_8M_pre:ro" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/toy/cache/protein_seg_cache_toy.pkl \
  --entrypoint python "$IMAGE" -m duetscreen.train \
  --records /toy/cache/variants_data_toy.pkl --output-dir /toy/out/seed42_toy \
  --ntv3-model /assets/NTv3_8M_pre --seed 42 --epochs 1 --patience 1 \
  --batch-size 16 --max-seq-len 2048 --external-test test=/toy/data/test_toy.fasta

echo "== 2/2 toy prediction"
"$DOCKER" run --rm --gpus all \
  -v "$HERE:/toy" -v "$ASSETS/NTv3_8M_pre:/assets/NTv3_8M_pre:ro" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/toy/cache/protein_seg_cache_toy.pkl \
  --entrypoint python "$IMAGE" -m duetscreen.predict \
  --fasta /toy/data/test_toy.fasta \
  --model /toy/out/seed42_toy/best_nucleotide_v3_mamba_model.pth \
  --ntv3-model /assets/NTv3_8M_pre --output-dir /toy/out/test_pred \
  --threshold 0.5 --batch-size 8

echo
echo "DONE"
echo "checkpoint: $HERE/out/seed42_toy/best_nucleotide_v3_mamba_model.pth"
echo "predictions: $HERE/out/test_pred/predictions_threshold_0.5.csv"
