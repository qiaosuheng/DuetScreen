#!/usr/bin/env bash
# Evaluate a checkpoint on a labelled FASTA and print ACC / F1 / AUROC.
#
# Run from the repository root:
#   MODEL=runs/full_seed42/seed42/best_nucleotide_v3_mamba_model.pth \
#   DATA=data/mixed/test.fasta CACHE=assets/protein_seg_cache_35m.pkl \
#   ./run_eval.sh
set -euo pipefail

# Git Bash on Windows rewrites /container/paths into C:/... ; keep them literal.
export MSYS_NO_PATHCONV=1 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-duetscreen-release:1.0}"
ASSETS="${ASSETS:-$HERE/assets}"
MODEL="${MODEL:-}"
DATA="${DATA:-}"
CACHE="${CACHE:-$ASSETS/protein_seg_cache_35m.pkl}"
OUT="${OUT:-$HERE/out_eval}"
THRESHOLD="${THRESHOLD:-0.5}"
BATCH="${BATCH:-64}"
DOCKER="${DOCKER:-docker}"

[ -n "$MODEL" ] || { echo "set MODEL=/path/to/best_nucleotide_v3_mamba_model.pth"; exit 1; }
[ -f "$MODEL" ] || { echo "checkpoint not found: $MODEL"; exit 1; }
[ -n "$DATA" ] || { echo "set DATA=/path/to/test.fasta"; exit 1; }
[ -f "$DATA" ] || { echo "FASTA not found: $DATA"; exit 1; }
[ -d "$ASSETS/NTv3_8M_pre" ] || { echo "missing $ASSETS/NTv3_8M_pre"; exit 1; }
[ -f "$CACHE" ] || { echo "protein cache not found: $CACHE (set CACHE=...)"; exit 1; }
"$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not found; run ./run_quickstart.sh first"; exit 1; }
mkdir -p "$OUT"

"$DOCKER" run --rm --gpus all \
  -v "$(cd "$(dirname "$MODEL")" && pwd):/model:ro" \
  -v "$(cd "$(dirname "$DATA")" && pwd):/data:ro" \
  -v "$(cd "$ASSETS" && pwd):/assets:ro" \
  -v "$(cd "$(dirname "$CACHE")" && pwd):/cache:ro" \
  -v "$(cd "$OUT" && pwd):/out" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE="/cache/$(basename "$CACHE")" \
  --entrypoint python "$IMAGE" -m duetscreen.predict \
  --fasta "/data/$(basename "$DATA")" \
  --model "/model/$(basename "$MODEL")" \
  --ntv3-model /assets/NTv3_8M_pre \
  --output-dir /out --threshold "$THRESHOLD" --batch-size "$BATCH"

RESULT="$OUT/detailed_results.json"
if [ ! -f "$RESULT" ]; then
  echo "FAIL: $RESULT was not written (prediction failed)"
  exit 1
fi
if ! grep -q '"accuracy"' "$RESULT"; then
  echo "FAIL: no metrics in $RESULT; check that the FASTA headers carry pos_/neg_ or |label:1 labels"
  exit 1
fi

echo
echo "Results: $RESULT"
echo "Predictions: $OUT/predictions_threshold_${THRESHOLD}.csv"
