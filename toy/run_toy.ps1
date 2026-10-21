# One-epoch toy training + prediction check (Windows PowerShell 5.1+).
#
# Unpack DuetScreen_ToySubset.zip so that this folder contains data\ and cache\.
# Run from the repository root:
#   .\toy\run_toy.ps1 -Assets assets
param([string]$Assets = "")

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$image = "duetscreen-release:1.0"
if (-not $Assets) { $Assets = Join-Path $root "assets" }

function Fail($m) { Write-Host "FAIL: $m"; exit 1 }

if (-not (Test-Path (Join-Path $Assets "NTv3_8M_pre"))) { Fail "missing NTv3_8M_pre under $Assets" }
if (-not (Test-Path (Join-Path $here "cache\variants_data_toy.pkl"))) { Fail "missing toy cache; unpack DuetScreen_ToySubset.zip into $here" }
if (-not (Test-Path (Join-Path $here "cache\protein_seg_cache_toy.pkl"))) { Fail "missing toy protein cache" }
docker image inspect $image *> $null
if ($LASTEXITCODE -ne 0) { Fail "image $image not found; build it with: docker build -t $image ." }

Write-Host "`n== 1/2 one-epoch training"
docker run --rm --gpus all `
  -v "$($here):/toy" -v "$($Assets)\NTv3_8M_pre:/assets/NTv3_8M_pre:ro" `
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/toy/cache/protein_seg_cache_toy.pkl `
  --entrypoint python $image -m duetscreen.train `
  --records /toy/cache/variants_data_toy.pkl --output-dir /toy/out/seed42_toy `
  --ntv3-model /assets/NTv3_8M_pre --seed 42 --epochs 1 --patience 1 `
  --batch-size 16 --max-seq-len 2048 --external-test test=/toy/data/test_toy.fasta
if ($LASTEXITCODE -ne 0) { Fail "toy training failed" }

Write-Host "`n== 2/2 toy prediction"
docker run --rm --gpus all `
  -v "$($here):/toy" -v "$($Assets)\NTv3_8M_pre:/assets/NTv3_8M_pre:ro" `
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/toy/cache/protein_seg_cache_toy.pkl `
  --entrypoint python $image -m duetscreen.predict `
  --fasta /toy/data/test_toy.fasta `
  --model /toy/out/seed42_toy/best_nucleotide_v3_mamba_model.pth `
  --ntv3-model /assets/NTv3_8M_pre --output-dir /toy/out/test_pred `
  --threshold 0.5 --batch-size 8
if ($LASTEXITCODE -ne 0) { Fail "toy prediction failed" }

Write-Host "`nDONE"
Write-Host "checkpoint: $(Join-Path $here 'out\seed42_toy\best_nucleotide_v3_mamba_model.pth')"
Write-Host "predictions: $(Join-Path $here 'out\test_pred\predictions_threshold_0.5.csv')"
