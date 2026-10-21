# Troubleshooting

`./run_quickstart.sh` (or `.\run_quickstart.ps1` on Windows) performs every
check automatically and prints `ALL CHECKS PASSED` on success. If something
fails, the messages below cover the common cases.

## Docker

| symptom | cause / fix |
|---|---|
| `Cannot connect to the Docker daemon` | Start Docker Desktop (Windows/macOS) or `sudo systemctl start docker`. |
| `could not select device driver "" with capabilities: [[gpu]]` | Install the NVIDIA Container Toolkit and restart Docker. Quick check: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`. |
| Build stops while compiling `mamba-ssm` / `causal-conv1d` | The build stage compiles CUDA kernels; it needs ~20 GB disk and several minutes. Set `MAX_JOBS=2` (already the default in the Dockerfile) if RAM is tight. |
| Build cannot download packages | The first build needs internet (PyPI + base image). On slow links, `docker build --build-arg PIP_INDEX_URL=<mirror> .` |
| Image is ~16 GB | Expected: CUDA runtime + compiled kernels. `docker system prune` after the build is fine. |

## GPU

| symptom | fix |
|---|---|
| `CUDA out of memory` during training | Training uses ~6.4 GB at batch 16 / 2048 bp. Lower `--batch-size 8`. |
| No GPU available | Prediction runs on CPU: `./run_quickstart.sh` with `NO_GPU=1` (bash) or `-NoGpu` (PowerShell). Training needs a GPU. |

## Assets

| symptom | fix |
|---|---|
| `assets not found` | Point at the folder that contains `checkpoints/`, `NTv3_8M_pre/` and `protein_seg_cache_35m.pkl`: `ASSETS=/path/to/assets ./run_quickstart.sh`. |
| `The NT-V3 checkpoint folder stored in this checkpoint does not exist` | Pass `--ntv3-model /assets/NTv3_8M_pre` to the predict CLI (the quickstart does this for you); a saved checkpoint path may differ from the local installation path. |
| ESM2 download fails while building a cache | Pre-download `facebook/esm2_t12_35M_UR50D` and pass the local folder via `--esm2`. |

## Results

| symptom | cause |
|---|---|
| `FAIL: <record> differs by ...` | Scores differ from the reference beyond 0.002. Re-run; if it persists, confirm you used `checkpoints/seed42` and the shipped `protein_seg_cache_35m.pkl`. The 0.002 limit is the quickstart comparison tolerance; small numerical differences can depend on hardware and precision. |
| A few scores differ from the recorded values by up to ~0.03 | Expected on a different GPU: floating-point nondeterminism. Aggregate metrics on the full test split still match to the fourth decimal (see CHECKS.md). |
| All scores ≈ 0.5 | The protein cache is missing entries for your sequences (they have no stop-free ORF ≥ 30 nt) — build a cache for them with `duetscreen.protein_cache`; or `DUETSCREEN_PROTEIN_SEG_CACHE` is unset. |

## Windows notes

- Run the quickstart from **PowerShell** (`run_quickstart.ps1`) or from **Git
  Bash** (`bash run_quickstart.sh`). Git Bash is handled (`MSYS_NO_PATHCONV`),
  but if you copy individual `docker run` commands by hand into Git Bash,
  prefix the command with `MSYS_NO_PATHCONV=1` so container paths such as
  `/assets/...` are not rewritten into `C:/...`.
- Paths with spaces work; quote them (`-v "D:\my data\assets:/assets:ro"`).
