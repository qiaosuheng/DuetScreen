# Toy subset: one-epoch training and prediction check

A small, fixed subset for reviewers who want to see the full pipeline run in a few minutes instead
of training on the complete benchmark.

The resource archive is `DuetScreen_ToySubset.zip`
(https://doi.org/10.5281/zenodo.23538086). It is self-contained: the full training caches
(https://doi.org/10.5281/zenodo.23537662) are **not** needed for the toy workflow.

The published archive is **v1**. Besides `data/` and `cache/`, it contains the v1 copies of
`README.md`, `run_toy.sh`, `run_toy.ps1` and `expected_metrics.json`. Those wrapper files are older
than this repository, and extracting the whole archive over `toy/` would overwrite them. **Extract
only `data/` and `cache/`**; keep the repository versions of the scripts and this README.

## Install the resources

Run from the repository root.

Linux / WSL2 / Git Bash:

```bash
unzip -o DuetScreen_ToySubset.zip 'data/*' 'cache/*' -d toy/
```

Windows PowerShell:

```powershell
Expand-Archive DuetScreen_ToySubset.zip -DestinationPath toy_tmp
Copy-Item -Path toy_tmp\data,toy_tmp\cache -Destination toy\ -Recurse -Force
Remove-Item toy_tmp -Recurse -Force
```

The result must be exactly:

```
toy/
  README.md          # repository version, not from the archive
  run_toy.sh         # repository version
  run_toy.ps1        # repository version
  data/
    train_toy.fasta
    val_toy.fasta
    test_toy.fasta
  cache/
    variants_data_toy.pkl
    protein_seg_cache_toy.pkl
```

## Contents used from the archive

| Path | Contents |
|---|---|
| `data/train_toy.fasta` | 120 records (60 positive / 60 negative) |
| `data/val_toy.fasta` | 40 records (20 / 20) |
| `data/test_toy.fasta` | 40 records (20 / 20) |
| `cache/variants_data_toy.pkl` | supervised views built from the toy train/val split (183 segments, 366 views) |
| `cache/protein_seg_cache_toy.pkl` | ESM2-35M 16-segment cache for all 223 toy sequences |

The v1 wrapper files in the archive (`README.md`, `run_toy.sh`, `run_toy.ps1`, `expected_metrics.json`)
are not used. They are listed for transparency only; the repository copies are authoritative.

The FASTAs are a stratified random subset of the released mixed-pool train/val/test splits
(sampling seed 42). Labels use the `pos_` / `neg_` prefixes.

## Requirements

- Docker image built from the repository: `docker build -t duetscreen-release:1.0 .`
- NT-V3 encoder folder: `assets/NTv3_8M_pre` (default layout: inside the repository)
- One GPU with at least 8 GB of memory

## Run

From the repository root, with `toy/data/` and `toy/cache/` in place:

```bash
ASSETS=assets ./toy/run_toy.sh
```

On Windows use `.\toy\run_toy.ps1 -Assets assets`. If PowerShell reports that running scripts is
disabled, start it with `powershell -ExecutionPolicy Bypass -File .\toy\run_toy.ps1 -Assets assets`.

The script runs one training epoch on the toy split and then predicts the toy test set. After the
Docker image has been built, this takes about two minutes on one GPU.

Expected sanity values (one epoch, seed 42):

| Quantity | Value |
|---|---:|
| Validation accuracy | 0.8125 |
| Validation combined accuracy | 0.8125 |
| Validation AUROC | 0.5876 |

A one-epoch model is not a performance benchmark: the test predictions are expected to be close to
all-positive or all-negative. The point is to verify that data loading, view construction, the
protein cache, training, checkpoint writing and prediction all work end to end.

## Manual commands

```bash
# training
docker run --rm --gpus all \
  -v "$PWD/toy:/toy" -v "$PWD/assets/NTv3_8M_pre:/assets/NTv3_8M_pre:ro" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/toy/cache/protein_seg_cache_toy.pkl \
  --entrypoint python duetscreen-release:1.0 -m duetscreen.train \
  --records /toy/cache/variants_data_toy.pkl --output-dir /toy/out/seed42_toy \
  --ntv3-model /assets/NTv3_8M_pre --seed 42 --epochs 1 --patience 1 \
  --batch-size 16 --max-seq-len 2048 --external-test test=/toy/data/test_toy.fasta

# prediction
docker run --rm --gpus all \
  -v "$PWD/toy:/toy" -v "$PWD/assets/NTv3_8M_pre:/assets/NTv3_8M_pre:ro" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/toy/cache/protein_seg_cache_toy.pkl \
  --entrypoint python duetscreen-release:1.0 -m duetscreen.predict \
  --fasta /toy/data/test_toy.fasta \
  --model /toy/out/seed42_toy/best_nucleotide_v3_mamba_model.pth \
  --ntv3-model /assets/NTv3_8M_pre --output-dir /toy/out/test_pred \
  --threshold 0.5 --batch-size 8
```

Do not pair `protein_seg_cache_toy.pkl` with a different view cache: cache keys are exact DNA
sequences, and views generated with another seed will miss keys.
