#!/usr/bin/env bash
# Environment check for the DuetScreen release.
#
# Verifies Docker, the NT-V3 encoder folder, builds the image if needed and
# checks that the package imports. It does not download data or weights.
#
# Usage:
#   ./run_quickstart.sh
#   ASSETS=/path/to/assets ./run_quickstart.sh     # folder containing NTv3_8M_pre/
#
# Exit code 0 = environment is ready for training.
set -uo pipefail

export MSYS_NO_PATHCONV=1 2>/dev/null || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="duetscreen-release:1.0"
DOCKER="${DOCKER:-docker}"

say() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }

say "1/4 Docker"
command -v "$DOCKER" >/dev/null 2>&1 || fail "docker command not found"
"$DOCKER" info >/dev/null 2>&1 || fail "Docker daemon is not running (start Docker Desktop / dockerd first)"
"$DOCKER" version --format '{{.Server.Version}}' | sed 's/^/    docker server: /'

say "2/4 encoder"
if [ -z "${ASSETS:-}" ]; then
  for candidate in "$HERE/assets" "$HERE/../assets"; do
    [ -d "$candidate/NTv3_8M_pre" ] && ASSETS="$candidate" && break
  done
fi
[ -n "${ASSETS:-}" ] || fail "NT-V3 folder not found: set ASSETS=/path/to/assets containing NTv3_8M_pre"
[ -d "$ASSETS/NTv3_8M_pre" ] || fail "$ASSETS/NTv3_8M_pre not found"
ASSETS="$(cd "$ASSETS" && pwd)"
echo "    $ASSETS/NTv3_8M_pre"

say "3/4 image"
if ! "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "    building $IMAGE (about 20 minutes the first time)"
  ( cd "$HERE" && "$DOCKER" build -t "$IMAGE" . ) || fail "docker build failed"
else
  echo "    $IMAGE already present"
fi

say "4/4 package import"
"$DOCKER" run --rm --entrypoint python "$IMAGE" -c \
  "import torch, duetscreen; print('    torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('    duetscreen import OK')" \
  || fail "package import failed"

say "READY"
cat <<'EOF'
Next steps:
  1. download the benchmark from https://doi.org/10.5281/zenodo.23534656
  2. place train.fasta / val.fasta under data/mixed/
  3. run: DATA=data/mixed ./run_full_training.sh
EOF
