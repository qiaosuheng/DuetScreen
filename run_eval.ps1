# Evaluate a checkpoint on a labelled FASTA (Windows PowerShell 5.1+).
#
# Run from the repository root:
#   $env:MODEL="runs\full_seed42\seed42\best_nucleotide_v3_mamba_model.pth"
#   $env:DATA="data\mixed\test.fasta"
#   $env:CACHE="assets\protein_seg_cache_35m.pkl"
#   .\run_eval.ps1
param(
    [string]$Model = $env:MODEL,
    [string]$Data = $env:DATA,
    [string]$Cache = $env:CACHE,
    [string]$Assets = "",
    [string]$Out = "",
    [double]$Threshold = 0.5
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$image = "duetscreen-release:1.0"
if (-not $Assets) { $Assets = Join-Path $here "assets" }
if (-not $Cache) { $Cache = Join-Path $Assets "protein_seg_cache_35m.pkl" }
if (-not $Out) { $Out = Join-Path $here "out_eval" }

function Fail($m) { Write-Host "FAIL: $m"; exit 1 }

if (-not $Model) { Fail "set MODEL or pass -Model" }
if (-not (Test-Path $Model)) { Fail "checkpoint not found: $Model" }
if (-not $Data) { Fail "set DATA or pass -Data" }
if (-not (Test-Path $Data)) { Fail "FASTA not found: $Data" }
if (-not (Test-Path (Join-Path $Assets "NTv3_8M_pre"))) { Fail "missing NTv3_8M_pre under $Assets" }
if (-not (Test-Path $Cache)) { Fail "protein cache not found: $Cache" }
docker image inspect $image *> $null
if ($LASTEXITCODE -ne 0) { Fail "image $image not found; run .\run_quickstart.ps1 first" }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

docker run --rm --gpus all `
  -v "$((Resolve-Path (Split-Path $Model)).Path):/model:ro" `
  -v "$((Resolve-Path (Split-Path $Data)).Path):/data:ro" `
  -v "$((Resolve-Path $Assets).Path):/assets:ro" `
  -v "$((Resolve-Path (Split-Path $Cache)).Path):/cache:ro" `
  -v "$((Resolve-Path $Out).Path):/out" `
  -e "DUETSCREEN_PROTEIN_SEG_CACHE=/cache/$(Split-Path $Cache -Leaf)" `
  --entrypoint python $image -m duetscreen.predict `
  --fasta "/data/$(Split-Path $Data -Leaf)" `
  --model "/model/$(Split-Path $Model -Leaf)" `
  --ntv3-model /assets/NTv3_8M_pre `
  --output-dir /out --threshold $Threshold --batch-size 64
if ($LASTEXITCODE -ne 0) { Fail "evaluation failed" }

Write-Host "`nResults: $(Join-Path $Out 'detailed_results.json')"
