# One-command verification of the DuetScreen release (Windows PowerShell 5.1+).
#
#   .\run_quickstart.ps1                 # uses .\assets or ..\assets
#   .\run_quickstart.ps1 -Assets D:\path\to\assets
#   .\run_quickstart.ps1 -NoGpu          # prediction on CPU, training skipped
#
# Exit code 0 = every executed check passed.
param(
    [string]$Assets = "",
    [switch]$NoGpu
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$image = "duetscreen-release:1.0"
$out = Join-Path $here "out_quickstart"
$work = Join-Path $here "work_quickstart"
$gpu = if ($NoGpu) { @() } else { @("--gpus", "all") }

function Say($m) { Write-Host "`n== $m" }
function Fail($m) { Write-Host "FAIL: $m"; exit 1 }

Say "1/5 environment"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail "docker command not found" }
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail "Docker daemon is not running (start Docker Desktop first)" }
Write-Host ("docker server: " + (docker version --format "{{.Server.Version}}"))

Say "2/5 assets"
if (-not $Assets) {
    foreach ($c in @((Join-Path $here "assets"), (Join-Path $here "..\assets"))) {
        if (Test-Path (Join-Path $c "checkpoints\seed42")) { $Assets = $c; break }
    }
}
if (-not $Assets -or -not (Test-Path (Join-Path $Assets "checkpoints\seed42"))) {
    $zip = Get-ChildItem -Path $here, (Join-Path $here ".."), (Join-Path $here "..\06_Docker") `
        -Filter "DuetScreen_Docker_Assets.zip" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $zip) { Fail "assets not found: pass -Assets or place DuetScreen_Docker_Assets.zip next to this folder" }
    Say ("    extracting " + $zip.FullName)
    $Assets = Join-Path $here "assets"
    New-Item -ItemType Directory -Force -Path $Assets | Out-Null
    Expand-Archive -Path $zip.FullName -DestinationPath $Assets -Force
}
$Assets = (Resolve-Path $Assets).Path
Get-ChildItem $Assets | ForEach-Object { Write-Host ("    " + $_.Name) }

Say "3/5 image"
docker image inspect $image *> $null
if ($LASTEXITCODE -ne 0) {
    Say "    building $image (about 20 minutes the first time)"
    Push-Location $here
    docker build -t $image .
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { Fail "docker build failed" }
} else {
    Write-Host "    $image already present"
}

Say "4/5 prediction + score check"
Remove-Item -Recurse -Force $out, $work -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $out, $work | Out-Null
$common = @("run", "--rm") + $gpu + @(
    "-v", "${Assets}:/assets:ro", "-v", "${out}:/out", "-v", "${work}:/work",
    "-v", "$here\tools:/tools:ro",
    "-e", "DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl",
    "--entrypoint", "python", $image)
& docker @common -m duetscreen.predict `
    --fasta /assets/smoke.fasta `
    --model /assets/checkpoints/seed42/best_nucleotide_v3_mamba_model.pth `
    --output-dir /out --threshold 0.5 --batch-size 4 `
    --ntv3-model /assets/NTv3_8M_pre
if ($LASTEXITCODE -ne 0) { Fail "prediction run failed" }
& docker @common /tools/verify_results.py `
    --predictions /out/predictions_threshold_0.5.csv `
    --expected /tools/expected_smoke.json
if ($LASTEXITCODE -ne 0) { Fail "predictions do not match the reference values" }

Say "5/5 training check"
if ($NoGpu) {
    Write-Host "    SKIPPED (training requires a GPU)"
} else {
    $trainArgs = @("run", "--rm") + $gpu + @(
        "-v", "${Assets}:/assets:ro", "-v", "${work}:/work",
        "-e", "DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl", $image,
        "--smoke", "--records", "/assets/mini_train_cache.pkl",
        "--output-dir", "/work/smoke", "--ntv3-model", "/assets/NTv3_8M_pre")
    & docker @trainArgs
    if ($LASTEXITCODE -ne 0) { Fail "training smoke failed" }
    & docker run --rm --entrypoint python `
        -v "${work}:/work" -v "$here\tools:/tools:ro" $image `
        /tools/verify_results.py --smoke /work/smoke/smoke.json
    if ($LASTEXITCODE -ne 0) { Fail "training smoke report check failed" }
}

Say "ALL CHECKS PASSED"
Write-Host "predictions: $out\predictions_threshold_0.5.csv"
Write-Host "training report: $work\smoke\smoke.json"
