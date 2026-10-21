#!/usr/bin/env bash
# One-command verification of the DuetScreen release.
#
# It checks the environment, builds the image if needed, runs the 8-record
# prediction with the released seed-42 checkpoint, compares the scores against
# the recorded reference values, and runs the one-step training check.
#
# Usage:
#   ./run_quickstart.sh                 # auto: uses ./assets or ../assets
#   ASSETS=/path/to/assets ./run_quickstart.sh
#   NO_GPU=1 ./run_quickstart.sh        # prediction on CPU, training skipped
#
# Exit code 0 = every executed check passed.
set -uo pipefail

# Git Bash on Windows rewrites /container/paths into C:/... ; keep them literal.
export MSYS_NO_PATHCONV=1 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="duetscreen-release:1.0"
OUT="$HERE/out_quickstart"
WORK="$HERE/work_quickstart"
DOCKER="${DOCKER:-docker}"
GPU_FLAG="--gpus all"
[ "${NO_GPU:-0}" = "1" ] && GPU_FLAG=""

say() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }

say "1/5 environment"
command -v "$DOCKER" >/dev/null 2>&1 || fail "docker command not found"
"$DOCKER" info >/dev/null 2>&1 || fail "Docker daemon is not running (start Docker Desktop / dockerd first)"
"$DOCKER" version --format '{{.Server.Version}}' | sed 's/^/docker server: /'

say "2/5 assets"
if [ -z "${ASSETS:-}" ]; then
  for candidate in "$HERE/assets" "$HERE/../assets" "$HERE/../06_Docker/assets"; do
    [ -d "$candidate/checkpoints/seed42" ] && ASSETS="$candidate" && break
  done
fi
if [ -z "${ASSETS:-}" ] || [ ! -d "$ASSETS/checkpoints/seed42" ]; then
  ZIP=$(ls "$HERE"/../DuetScreen_Docker_Assets.zip "$HERE"/../06_Docker/DuetScreen_Docker_Assets.zip 2>/dev/null | head -1)
  [ -n "$ZIP" ] || fail "assets not found: set ASSETS=/path/to/assets or put DuetScreen_Docker_Assets.zip next to this folder"
  say "    unzipping $ZIP"
  mkdir -p "$HERE/assets" && unzip -q -o "$ZIP" -d "$HERE/assets"
  ASSETS="$HERE/assets"
fi
ASSETS="$(cd "$ASSETS" && pwd)"
ls "$ASSETS" | sed 's/^/    /'

say "3/5 image"
if ! "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1; then
  say "    building $IMAGE (about 20 minutes the first time)"
  ( cd "$HERE" && "$DOCKER" build -t "$IMAGE" . ) || fail "docker build failed (see output above)"
else
  echo "    $IMAGE already present"
fi

say "4/5 prediction + score check"
rm -rf "$OUT" "$WORK"; mkdir -p "$OUT" "$WORK"
RUN_ARGS=(run --rm $GPU_FLAG \
  -v "$ASSETS:/assets:ro" -v "$OUT:/out" -v "$WORK:/work" \
  -v "$HERE/tools:/tools:ro" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl \
  --entrypoint python "$IMAGE")
"$DOCKER" "${RUN_ARGS[@]}" -m duetscreen.predict \
  --fasta /assets/smoke.fasta \
  --model /assets/checkpoints/seed42/best_nucleotide_v3_mamba_model.pth \
  --output-dir /out --threshold 0.5 --batch-size 4 \
  --ntv3-model /assets/NTv3_8M_pre || fail "prediction run failed"
"$DOCKER" "${RUN_ARGS[@]}" /tools/verify_results.py \
  --predictions /out/predictions_threshold_0.5.csv \
  --expected /tools/expected_smoke.json || fail "predictions do not match the reference values"

say "5/5 training check"
if [ -z "$GPU_FLAG" ]; then
  echo "    SKIPPED (training requires a GPU; rerun without NO_GPU=1)"
else
  "$DOCKER" run --rm $GPU_FLAG \
    -v "$ASSETS:/assets:ro" -v "$WORK:/work" \
    -e DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl \
    "$IMAGE" --smoke --records /assets/mini_train_cache.pkl \
    --output-dir /work/smoke --ntv3-model /assets/NTv3_8M_pre \
    || fail "training smoke failed"
  "$DOCKER" run --rm --entrypoint python \
    -v "$WORK:/work" -v "$HERE/tools:/tools:ro" "$IMAGE" /tools/verify_results.py \
    --smoke /work/smoke/smoke.json || fail "training smoke report check failed"
fi

say "ALL CHECKS PASSED"
echo "predictions: $OUT/predictions_threshold_0.5.csv"
echo "training report: $WORK/smoke/smoke.json"
