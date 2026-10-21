#!/usr/bin/env bash
# Full retraining of the released DuetScreen configuration.
#
# Steps (each skipped when its output already exists):
#   1. build the training record cache from the released train/val split
#   2. make sure the protein-segment cache covers every record
#   3. train 20 epochs (early stop patience 5) and keep the best checkpoint
#
# Usage:
#   DATA=/path/to/split ./run_full_training.sh          # split holds train.fasta / val.fasta
#   SEED=43 EPOCHS=20 ./run_full_training.sh
#
# Expected hardware: one GPU with >= 8 GB free; allow several hours.
# The resulting checkpoint is trained from scratch on this machine, so it will
# not be bit-identical to the released seeds; it follows the same recipe.
set -uo pipefail
export MSYS_NO_PATHCONV=1 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="duetscreen-release:1.0"
DATA="${DATA:-$HERE/data}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-16}"
WORK="$HERE/runs/full_seed${SEED}"
DOCKER="${DOCKER:-docker}"

say() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }

[ -d "$DATA" ] || fail "DATA=$DATA not found (needs train.fasta and val.fasta)"
[ -f "$DATA/train.fasta" ] && [ -f "$DATA/val.fasta" ] || fail "$DATA must hold train.fasta and val.fasta"

if [ -z "${ASSETS:-}" ]; then
  for c in "$HERE/assets" "$HERE/../assets"; do
    [ -d "$c/NTv3_8M_pre" ] && ASSETS="$c" && break
  done
fi
[ -n "${ASSETS:-}" ] || fail "NT-V3 encoder folder not found; set ASSETS=/path/to/assets containing NTv3_8M_pre"
ASSETS="$(cd "$ASSETS" && pwd)"
mkdir -p "$WORK"

"$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1 || fail "image $IMAGE missing; run ./run_quickstart.sh once to build it"

say "1/3 record cache"
if [ -f "$WORK/variants_data.pkl" ]; then
  echo "    reusing $WORK/variants_data.pkl"
else
  "$DOCKER" run --rm --entrypoint python \
    -v "$DATA:/data:ro" -v "$WORK:/work" "$IMAGE" \
    -m duetscreen.build_cache --train-fasta /data/train.fasta --val-fasta /data/val.fasta \
    --output /work/variants_data.pkl --seed "$SEED" \
    || fail "cache build failed"
fi

say "2/3 protein cache coverage"
COVER=(run --rm --entrypoint python -v "$WORK:/work:ro" -v "$ASSETS:/assets:ro" -v "$HERE/tools:/tools:ro")
if "$DOCKER" "${COVER[@]}" "$IMAGE" /tools/check_cache_coverage.py \
     --records /work/variants_data.pkl --cache /assets/protein_seg_cache_35m.pkl 2>/dev/null; then
  echo "    shipped protein cache covers the records"
else
  echo "    building protein cache for $WORK (GPU)"
  "$DOCKER" run --rm --gpus all --entrypoint python \
    -v "$WORK:/work" -v "$ASSETS:/assets:ro" "$IMAGE" \
    -m duetscreen.protein_cache --input /work/variants_data.pkl \
    --output /work/protein_seg_cache.pkl \
    --esm2 facebook/esm2_t12_35M_UR50D --device cuda --batch-size 16 \
    || fail "protein cache build failed"
  export PROTEIN_CACHE=/work/protein_seg_cache.pkl
fi

say "3/3 training (${EPOCHS} epochs, seed ${SEED})"
CACHE_ENV=(-e DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl)
[ -n "${PROTEIN_CACHE:-}" ] && CACHE_ENV=(-e DUETSCREEN_PROTEIN_SEG_CACHE="$PROTEIN_CACHE")
"$DOCKER" run --rm --gpus all \
  -v "$WORK:/work" -v "$ASSETS:/assets:ro" \
  "${CACHE_ENV[@]}" "$IMAGE" \
  --records /work/variants_data.pkl --output-dir "/work/seed${SEED}" \
  --ntv3-model /assets/NTv3_8M_pre --seed "$SEED" \
  --epochs "$EPOCHS" --patience 5 --batch-size "$BATCH" --max-seq-len 2048 \
  || fail "training failed"

say "DONE"
echo "checkpoint: $WORK/seed${SEED}/best_nucleotide_v3_mamba_model.pth"
echo "training log: $WORK/seed${SEED}/progress.jsonl"
echo "select with:  python -m duetscreen.predict --model <checkpoint> ..."
