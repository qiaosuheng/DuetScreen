# Environment check for the DuetScreen release (Windows PowerShell 5.1+).
#
#   .\run_quickstart.ps1
#   .\run_quickstart.ps1 -Assets D:\path\to\assets    # folder containing NTv3_8M_pre
#
# Exit code 0 = environment is ready for training.
param([string]$Assets = "")

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$image = "duetscreen-release:1.0"

function Say($m) { Write-Host "`n== $m" }
function Fail($m) { Write-Host "FAIL: $m"; exit 1 }

Say "1/4 Docker"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail "docker command not found" }
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail "Docker daemon is not running (start Docker Desktop first)" }
Write-Host ("    docker server: " + (docker version --format "{{.Server.Version}}"))

Say "2/4 encoder"
if (-not $Assets) {
    foreach ($c in @((Join-Path $here "assets"), (Join-Path $here "..\assets"))) {
        if (Test-Path (Join-Path $c "NTv3_8M_pre")) { $Assets = $c; break }
    }
}
if (-not $Assets -or -not (Test-Path (Join-Path $Assets "NTv3_8M_pre"))) {
    Fail "NT-V3 folder not found: pass -Assets with a folder containing NTv3_8M_pre"
}
$Assets = (Resolve-Path $Assets).Path
Write-Host ("    " + (Join-Path $Assets "NTv3_8M_pre"))

Say "3/4 image"
docker image inspect $image *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    building $image (about 20 minutes the first time)"
    Push-Location $here
    docker build -t $image .
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Fail "docker build failed" }
} else {
    Write-Host "    $image already present"
}

Say "4/4 package import"
docker run --rm --entrypoint python $image -c "import torch, duetscreen; print('    torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('    duetscreen import OK')"
if ($LASTEXITCODE -ne 0) { Fail "package import failed" }

Say "READY"
Write-Host @"
Next steps:
  1. download the benchmark from https://doi.org/10.5281/zenodo.23534656
  2. place train.fasta / val.fasta under data\mixed\
  3. run: `$env:DATA='data\mixed'; ./run_full_training.sh
"@
