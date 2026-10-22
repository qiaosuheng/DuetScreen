#!/usr/bin/env bash
# Predict with a checkpoint you trained with run_full_training.sh / run_train.sh.
#
#   MODEL=./runs/seed42/best_nucleotide_v3_mamba_model.pth \
#   DATA=./examples ASSETS=./assets ./docker/run_predict.sh
#
# ASSETS must contain NTv3_8M_pre/. The protein cache is optional at prediction
# time; set CACHE=/path/to/protein_seg_cache_35m.pkl to reuse one.
set -euo pipefail

# Git Bash on Windows rewrites /container/paths into C:/... ; keep them literal.
export MSYS_NO_PATHCONV=1 2>/dev/null || true

IMAGE="${IMAGE:-duetscreen-release:1.0}"
MODEL="${MODEL:-}"
ASSETS="${ASSETS:-./assets}"
DATA="${DATA:-./examples}"
OUT="${OUT:-$(pwd)/out}"
CACHE="${CACHE:-}"
FASTA="${FASTA:-quickstart.fasta}"
DOCKER="${DOCKER:-docker}"

[ -n "$MODEL" ] || { echo "set MODEL=/path/to/best_nucleotide_v3_mamba_model.pth"; exit 1; }
[ -f "$MODEL" ] || { echo "checkpoint not found: $MODEL"; exit 1; }
[ -d "$ASSETS/NTv3_8M_pre" ] || { echo "missing $ASSETS/NTv3_8M_pre"; exit 1; }
mkdir -p "$OUT"

ENV_ARGS=()
VOLS=(
  -v "$(cd "$ASSETS" && pwd):/assets:ro"
  -v "$(cd "$(dirname "$MODEL")" && pwd):/model:ro"
  -v "$(cd "$DATA" && pwd):/data:ro"
  -v "$(cd "$OUT" && pwd):/out"
)
if [ -n "$CACHE" ]; then
  ENV_ARGS=(-e "DUETSCREEN_PROTEIN_SEG_CACHE=/cache/$(basename "$CACHE")")
  VOLS+=(-v "$(cd "$(dirname "$CACHE")" && pwd):/cache:ro")
fi

"$DOCKER" run --rm --gpus all \
  "${VOLS[@]}" \
  "${ENV_ARGS[@]}" \
  --entrypoint python "$IMAGE" \
  -m duetscreen.predict \
  --fasta "/data/$FASTA" \
  --model "/model/$(basename "$MODEL")" \
  --ntv3-model /assets/NTv3_8M_pre \
  --output-dir /out --threshold "${THRESHOLD:-0.5}" --batch-size "${BATCH:-8}"
