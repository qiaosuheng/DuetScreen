# Troubleshooting

`./run_quickstart.sh` (or `.\run_quickstart.ps1` on Windows) is an **environment
check** only. It verifies Docker, `assets/NTv3_8M_pre`, builds the image if needed
and imports `torch` / `duetscreen`; success prints `READY`. It does not run a
prediction, compute metrics or train a model.

For labelled evaluation use `./run_eval.sh`; for the toy workflow use
`./toy/run_toy.sh`; for training use `./run_full_training.sh`.

## Docker and platform

| symptom | cause / fix |
|---|---|
| `Cannot connect to the Docker daemon` | Start Docker Desktop (Windows) or `sudo systemctl start docker` (Linux). |
| `could not select device driver "" with capabilities: [[gpu]]` | Install the NVIDIA Container Toolkit and restart Docker. Check: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`. |
| Build stops while compiling `mamba-ssm` / `causal-conv1d` | The build compiles CUDA kernels; allow ~20 GB disk and several minutes. Keep `MAX_JOBS=2` if RAM is tight. |
| Build cannot download packages | The first build needs internet (PyPI + base image). On slow links: `docker build --build-arg PIP_INDEX_URL=<mirror> .` |
| Image is large (~16 GB) | Expected: CUDA runtime plus compiled kernels. |

**Supported platforms:** Linux with an NVIDIA GPU, or Windows 11 with WSL2 GPU
support (Docker Desktop). macOS with CUDA is **not** supported; CPU-only training
is not a supported path.

## GPU

| symptom | fix |
|---|---|
| `CUDA out of memory` during training | Training uses about 6.4 GB at batch 16 / 2048 bp. Lower `--batch-size 8`. |
| No GPU available | Training requires a GPU. Prediction/evaluation also use `--gpus all` in the provided scripts; CPU inference is not a validated path. |

## Assets

| symptom | fix |
|---|---|
| `NT-V3 folder not found` | Put `NTv3_8M_pre/` under `assets/` (from `NTv3_8M_pre.zip`) or set `ASSETS=/path/to/assets`. |
| `checkpoint not found` in `run_eval.sh` | Set `MODEL=/path/to/best_nucleotide_v3_mamba_model.pth`; the released one is in the seed-42 record. |
| `protein cache not found` in `run_eval.sh` | Set `CACHE=/path/to/protein_seg_cache_35m.pkl`, or use the cache produced by training at `runs/shared/protein_seg_cache.pkl`. |
| `The NT-V3 checkpoint folder stored in this checkpoint does not exist` | Pass `--ntv3-model /assets/NTv3_8M_pre`; the checkpoint stores the training host's absolute path. |
| ESM2 download fails while building a cache | Pre-download `facebook/esm2_t12_35M_UR50D` and pass the folder via `--esm2`. |

## Evaluation results

| symptom | cause |
|---|---|
| `FAIL: no metrics in ...detailed_results.json` | The FASTA headers do not carry labels. Use `pos_` / `neg_` prefixes or `|label:1` / `|label:0`. |
| All scores ≈ 0.5 | The protein cache has no entry for your sequences (no stop-free ORF ≥ 30 nt), or `DUETSCREEN_PROTEIN_SEG_CACHE` is unset. Build a cache for them with `duetscreen.protein_cache`. |
| A few scores differ from recorded values | Floating-point nondeterminism across GPUs; aggregate metrics are the intended comparison. |

## Toy

| symptom | fix |
|---|---|
| `toy/data` or `toy/cache` missing | Extract only those two directories from `DuetScreen_ToySubset.zip`; see `toy/README.md`. |
| `toy/README.md` or scripts look older after unpacking | The v1 archive contains wrapper files. Restore the repository copies with `git checkout -- toy/README.md toy/run_toy.sh toy/run_toy.ps1`; do not extract those files from the archive. |
| `run_toy.sh` fails in Git Bash on Windows | Use `run_toy.ps1` in a Windows terminal; the v1 archive's Bash script predates the path fix. |

## Windows notes

- Bash blocks target Linux/WSL2 or Git Bash; PowerShell blocks target the `.ps1`
  wrappers. Do not mix them in one terminal.
- If PowerShell reports that running scripts is disabled, start it with
  `powershell -ExecutionPolicy Bypass -File .\run_quickstart.ps1`.
- Git Bash is handled by the scripts (`MSYS_NO_PATHCONV`). If you copy individual
  `docker run` commands by hand, prefix them with `MSYS_NO_PATHCONV=1` so paths
  such as `/assets/...` are not rewritten to `C:/...`.
- Paths with spaces work; quote them (`-v "D:\my data\assets:/assets:ro"`).
