#!/usr/bin/env bash
# Full training inside the container: views -> protein cache -> training.
#
# Only the dataset and the NT-V3 encoder are required; caches are built on the
# first run and reused afterwards.
#
#   ASSETS=./assets DATA=./data/mixed ./docker/run_train.sh
#   SEED=43 EPOCHS=20 ./docker/run_train.sh
set -euo pipefail

IMAGE="${IMAGE:-duetscreen-release:1.0}"
ASSETS="${ASSETS:-./assets}"          # must contain NTv3_8M_pre/
DATA="${DATA:-./data/mixed}"          # train.fasta / val.fasta
WORK="${WORK:-$(pwd)/runs}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-16}"
DOCKER="${DOCKER:-docker}"

[ -d "$ASSETS/NTv3_8M_pre" ] || { echo "missing $ASSETS/NTv3_8M_pre"; exit 1; }
[ -f "$DATA/train.fasta" ] && [ -f "$DATA/val.fasta" ] || { echo "missing $DATA/train.fasta or val.fasta"; exit 1; }
mkdir -p "$WORK"

CMD="
set -e
if [ ! -f /work/variants_data.pkl ]; then
  python -m duetscreen.build_cache --train-fasta /data/train.fasta --val-fasta /data/val.fasta \
    --output /work/variants_data.pkl --seed $SEED
fi
if [ ! -f /work/protein_seg_cache.pkl ]; then
  python -m duetscreen.protein_cache --input /work/variants_data.pkl --output /work/protein_seg_cache.pkl \
    --esm2 facebook/esm2_t12_35M_UR50D --device cuda --batch-size 16
fi
DUETSCREEN_PROTEIN_SEG_CACHE=/work/protein_seg_cache.pkl python -m duetscreen.train \
  --records /work/variants_data.pkl --output-dir /work/seed$SEED \
  --ntv3-model /assets/NTv3_8M_pre --seed $SEED \
  --epochs $EPOCHS --patience 5 --batch-size $BATCH --max-seq-len 2048
"

"$DOCKER" run --rm --gpus all \
  -v "$(cd "$ASSETS" && pwd):/assets:ro" \
  -v "$(cd "$DATA" && pwd):/data:ro" \
  -v "$(cd "$WORK" && pwd):/work" \
  --entrypoint bash "$IMAGE" -lc "$CMD"
