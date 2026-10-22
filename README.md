# DuetScreen

Reference implementation of **DuetScreen**, a dual-space screening model for DNA
sequences. The model pairs a partially adapted Nucleotide Transformer v3 encoder
(decoder trainable; convolutional tower and transformer backbone frozen) with a
frozen protein language model: the translated open reading frame is encoded as a
16-segment positional profile by ESM2-35M, refined by a segment self-attention
block, and read out through a nucleotide-conditioned cross-attention. A
dual-stream downstream network (Mamba blocks plus a ConvNeXt-style side branch)
processes the nucleotide representation, and the two spaces are combined by late
fusion. The classification head is trained with the multi-view supervised
objective described in the paper.

The model is intended for **biosecurity screening research**: flagging material
derived from sequences of concern. It does not predict whether a sequence is
expressed, viable or functional.

## Table of contents

- [Architecture](#architecture)
- [First-time setup: download → train → metrics](#first-time-setup-download--train--metrics)
- [What is included](#what-is-included)
- [Project layout after setup](#project-layout-after-setup)
- [Data availability](#data-availability)
- [Model weights](#model-weights)
- [Full training caches](#full-training-caches)
- [Build the protein cache (ESM2-35M)](#build-the-protein-cache-esm2-35m)
- [Environment](#environment)
- [Quickstart](#quickstart)
- [Toy validation (1 epoch)](#toy-validation-1-epoch)
- [Training from the public dataset](#full-retraining-from-the-public-dataset)
- [Prediction](#predict)
- [Training](#train)
- [Output files](#output-files)
- [Configuration reference](#configuration-reference)
- [FAQ](#faq)
- [Citation and license](#citation-and-license)

## Architecture

![DuetScreen architecture](docs/fig1bc.png)

**b**, The NT-V3 encoder: the convolutional tower and transformer backbone are frozen, the
deconvolutional tower is adapted for the screening task. **c**, Dual-space model: the nucleotide
stream (adapted NT-V3, state-space and convolution branches, token fusion) and the protein stream
(translation, frozen ESM2-35M, 16 segment summaries, segment self-attention, protein readout) are
fused into a joint representation, followed by the classification head and the training-only
contrastive projection.

## First-time setup: download → train → metrics

Every command below is run from the repository root, i.e. the directory that contains
`run_full_training.sh`, `run_eval.sh` and `Dockerfile`. Scripts are **not** installed globally and
are **not** on `PATH`; call them with their relative path (`./run_full_training.sh`) from that
directory.

### 0. Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine (Linux), with GPU support.
- NVIDIA driver and the NVIDIA Container Toolkit.
- One GPU with at least 8 GB of memory; about 20 GB free disk.

Verify that Docker can see the GPU before doing anything else:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If this command fails, fix Docker GPU support first (Docker Desktop settings → Resources → WSL
integration / GPU, or install the NVIDIA Container Toolkit on Linux); the rest of the workflow will
not work without it.

The one-command scripts below call `docker`. The native Python path in **Environment** is an
alternative for users who prefer to run the `python -m duetscreen.*` commands directly.

### 1. Clone

```bash
git clone https://github.com/qiaosuheng/DuetScreen.git
cd DuetScreen
```

### 2. Download and place the files

| Download | From | Put it at |
|---|---|---|
| `NTv3_8M_pre.zip` | https://doi.org/10.5281/zenodo.23535423 | unzip into `assets/` → `assets/NTv3_8M_pre/` |
| `mixed pool.zip` | https://doi.org/10.5281/zenodo.23534656 | unzip into `data/mixed/` → `train.fasta`, `val.fasta`, `test.fasta` |
| `variants_data_seed42.pkl` | https://doi.org/10.5281/zenodo.23537662 | rename to `runs/full_seed42/variants_data.pkl` |
| `protein_seg_cache_35m.7z.001–032` | https://doi.org/10.5281/zenodo.23537662 | extract → `assets/protein_seg_cache_35m.pkl` |
| `best_nucleotide_v3_mamba_model.pth` (optional) | https://doi.org/10.5281/zenodo.23535423 | `assets/checkpoints/seed42/best_nucleotide_v3_mamba_model.pth` |
| `DuetScreen_ToySubset.zip` (optional toy) | https://doi.org/10.5281/zenodo.23538086 | unzip into `toy/` |

```bash
mkdir -p assets data/mixed runs/full_seed42 assets/checkpoints/seed42

# nucleotide encoder
wget -O /tmp/NTv3_8M_pre.zip "https://zenodo.org/records/23535423/files/NTv3_8M_pre.zip?download=1"
unzip -o /tmp/NTv3_8M_pre.zip -d assets/

# benchmark split
wget -O /tmp/mixed_pool.zip "https://zenodo.org/records/23534656/files/mixed%20pool.zip?download=1"
unzip -o /tmp/mixed_pool.zip -d data/mixed

# full training caches (optional if you let the training script rebuild them)
wget -O runs/full_seed42/variants_data.pkl \
  "https://zenodo.org/records/23537662/files/variants_data_seed42.pkl?download=1"
# protein cache: download all 32 volumes, then
7z x protein_seg_cache_35m.7z.001 -oassets/
```

The two caches in record `23537662` are a matched pair. **Full training needs both.** The toy archive
is independent and contains its own `toy/data/` and `toy/cache/`.

### 3. Build the image once

```bash
docker build -t duetscreen-release:1.0 .
```

Run this **from the repository root** (the folder that contains `Dockerfile`). The trailing `.` is the
build context, and Docker automatically reads `./Dockerfile` from that context; you do **not** pass
the file name explicitly. `-t duetscreen-release:1.0` names the image, and the training/evaluation
scripts expect exactly that name. To build from another directory:

```bash
docker build -f /path/to/DuetScreen/Dockerfile -t duetscreen-release:1.0 /path/to/DuetScreen
```

The `Dockerfile` **is** the environment configuration: it pins Python 3.11, PyTorch 2.8.0 with
CUDA 12.8, transformers 4.57.1 and compiles `mamba-ssm 2.2.6.post3` / `causal-conv1d 1.5.3.post1`.
You do not need to edit it; build it once (about 20 minutes the first time) and reuse the image for
training and evaluation. It is only used at build time; running training or prediction does not
reference the file.

`./run_quickstart.sh` is an optional environment check; it builds the same image and verifies the
import. It is not required before training.

### 4. Train

```bash
DATA=data/mixed ./run_full_training.sh
```

The script is at the repository root. It builds `runs/full_seed42/variants_data.pkl` and
`runs/full_seed42/protein_seg_cache.pkl` when they are absent (several hours), or reuses the files
placed in step 2. The checkpoint is written to
`runs/full_seed42/seed42/best_nucleotide_v3_mamba_model.pth`.

### 5. Evaluate and print metrics

```bash
MODEL=runs/full_seed42/seed42/best_nucleotide_v3_mamba_model.pth \
DATA=data/mixed/test.fasta \
CACHE=assets/protein_seg_cache_35m.pkl \
./run_eval.sh
```

`run_eval.sh` is at the repository root. It prints accuracy, F1, precision, recall, AUROC and AUPRC,
and writes `out_eval/detailed_results.json` plus `out_eval/predictions_threshold_0.5.csv`. If you
did not download the full protein cache, use the one produced by training:
`CACHE=runs/full_seed42/protein_seg_cache.pkl`.

On Windows use `.\run_eval.ps1 -Model ... -Data ... -Cache ...`; if PowerShell blocks scripts, add
`-ExecutionPolicy Bypass` by starting it as
`powershell -ExecutionPolicy Bypass -File .\run_eval.ps1 ...`.

### 6. Optional: toy check (about two minutes)

```bash
unzip -o DuetScreen_ToySubset.zip -d toy/
ASSETS=assets ./toy/run_toy.sh
```

`toy/run_toy.sh` lives in the `toy/` subdirectory and needs no full cache. The source tree contains
only `toy/run_toy.sh`; `toy/data/` and `toy/cache/` appear after unpacking the toy archive.

On Windows use `.\toy\run_toy.ps1 -Assets assets`; if PowerShell blocks scripts, run it with
`powershell -ExecutionPolicy Bypass -File .\toy\run_toy.ps1 -Assets assets`.

### Where the scripts live

| Script | Directory | Purpose |
|---|---|---|
| `run_quickstart.sh` | repository root | optional Docker / encoder / import check |
| `run_full_training.sh` | repository root | one-command training on `DATA=data/mixed` |
| `run_eval.sh` | repository root | one-command prediction + ACC/F1/AUROC |
| `run_eval.ps1` | repository root | Windows version of the evaluation wrapper |
| `toy/run_toy.sh` | `toy/` | one-epoch toy training + prediction |
| `docker/run_train.sh` | `docker/` | container wrapper for training |
| `docker/run_predict.sh` | `docker/` | container wrapper for prediction |

## What is included

- `duetscreen/`: model, encoder, data pipeline, training-view generator, protein cache, trainer and prediction CLI.
- `run_full_training.sh`, `run_eval.sh`, `run_quickstart.sh` (repository root): one-command training, evaluation and environment check.
- `configs/duetscreen_release.json`: the configuration used for the reported runs.
- `docker/`: container entry points for training, smoke training and prediction.
- `tools/`: cache-coverage and result-verification utilities.
- `toy/`: one-epoch toy workflow; `toy/data/` and `toy/cache/` are created after unpacking the toy archive.
- `Dockerfile`, `requirements.txt`: pinned environment.
- `docs/`: setup and troubleshooting notes.

## What is not tracked in git

- Benchmark sequences (Zenodo, DOI below).
- Encoder weights (Hugging Face).
- Run outputs: `runs/`, checkpoints, protein caches and logs are created locally.

## Project layout after setup

A working checkout after downloading the encoder, the dataset and running one training job:

```
DuetScreen/
├── duetscreen/                  model, encoder, data, trainer, predict CLI
├── configs/                     released configuration snapshot
├── docker/                      run_train.sh, run_predict.sh, run_train_smoke.sh
├── docs/                        setup notes
├── tools/                       cache coverage and result verification
├── Dockerfile
├── requirements.txt
├── run_full_training.sh
├── run_quickstart.sh
├── run_eval.sh
├── run_eval.ps1
├── toy/
│   ├── README.md
│   ├── run_toy.sh
│   ├── run_toy.ps1
│   ├── expected_metrics.json
│   ├── data/                    after unpacking DuetScreen_ToySubset.zip
│   │   ├── train_toy.fasta
│   │   ├── val_toy.fasta
│   │   └── test_toy.fasta
│   └── cache/                   after unpacking DuetScreen_ToySubset.zip
│       ├── variants_data_toy.pkl
│       └── protein_seg_cache_toy.pkl
├── assets/
│   └── NTv3_8M_pre/             nucleotide encoder (Hugging Face download)
├── data/
│   └── mixed/
│       ├── train.fasta
│       ├── val.fasta
│       └── test.fasta
└── runs/
    └── full_seed42/
        ├── variants_data.pkl    training views built from train/val
        ├── protein_seg_cache.pkl
        └── seed42/
            ├── best_nucleotide_v3_mamba_model.pth
            ├── nucleotide_v3_training_log.json
            ├── progress.jsonl
            └── complete.json
```

The Docker image `duetscreen-release:1.0` is built once from the repository root; no other local
state is required. `runs/` is created by the training script and is not tracked by git.

`toy/data/` and `toy/cache/` are not tracked in git: unpack
`DuetScreen_ToySubset.zip` (https://doi.org/10.5281/zenodo.23538086) directly into `toy/`. The
archive already contains the `data/` and `cache/` folders, and `toy/run_toy.sh` writes its outputs to
`toy/out/`.

## Data availability

The benchmark dataset is archived on Zenodo: **https://doi.org/10.5281/zenodo.23534656**
(mixed pool, strict pool and the species/family holdout sets, `pos_`/`neg_` labels in the FASTA IDs).

| Archive | Files to place under `data/` | Records |
|---|---|---|
| `mixed pool.zip` | `mixed/train.fasta`, `mixed/val.fasta`, `mixed/test.fasta` | 13,430 / 1,679 / 1,679 |
| `strict pool.zip` | `strict/train.fasta`, `strict/val.fasta`, `strict/test.fasta` | 6,446 / 805 / 805 |
| `holdout pool.zip` | `holdout/generalization_species.fasta`, `holdout/generalization_family.fasta` | 1,164 / 817 |

```bash
mkdir -p data && cd data
wget "https://zenodo.org/records/23534656/files/mixed%20pool.zip?download=1" -O mixed_pool.zip
wget "https://zenodo.org/records/23534656/files/strict%20pool.zip?download=1" -O strict_pool.zip
wget "https://zenodo.org/records/23534656/files/holdout%20pool.zip?download=1" -O holdout_pool.zip
unzip -o mixed_pool.zip -d mixed && unzip -o strict_pool.zip -d strict && unzip -o holdout_pool.zip -d holdout
cd ..
```

The training recipe writes the checkpoint to
`runs/full_seed<ID>/seed<ID>/best_nucleotide_v3_mamba_model.pth`; prediction accepts that path
through `MODEL=` or `--model`. Training is stochastic, so a fresh run follows the
same recipe and performance range as the paper's seeds without being bit-identical (the seed-42
reference reached 0.8497 validation combined accuracy).

Encoder weights are obtained from their upstream releases (see below).

## Model weights

The seed-42 checkpoint and the NT-V3 encoder are archived at
**https://doi.org/10.5281/zenodo.23535423** (CC BY 4.0):

| File | Contents |
|---|---|
| `best_nucleotide_v3_mamba_model.pth` | seed-42 checkpoint (90 MB) |
| `NTv3_8M_pre.zip` | NT-V3-Pre-8M encoder folder (29 MB) |

Unpack `NTv3_8M_pre.zip` to `assets/NTv3_8M_pre` and pass the checkpoint to prediction with
`--model`. Training writes its own checkpoint to `runs/full_seed<ID>/seed<ID>/`; the released
seed-42 checkpoint can be used for prediction or as a starting point.

## Full training caches

The seed-42 training-view cache and the complete ESM2 protein cache are archived at
**https://doi.org/10.5281/zenodo.23537662**:

| File | Contents |
|---|---|
| `variants_data_seed42.pkl` | full training-view cache built with `build_cache --seed 42` (16,738 records) |
| `protein_seg_cache_35m.7z.001` … `.032` | 16-segment ESM2-35M cache, split into 32 volumes (about 820 MB compressed, 979 MB extracted) |

**Full training needs both files.** The toy workflow does **not** use this record: it only needs the
toy archive (`DuetScreen_ToySubset.zip`), which already contains its own small view and protein
caches.

The cache also contains embeddings for the test/holdout sequences. This is not leakage: training only
looks up sequences present in `variants_data.pkl`, and the extra keys are never read (see
[Why a merged cache is not test-set leakage](#why-a-merged-cache-is-not-test-set-leakage)).

Extract the protein cache from the first volume:

```bash
7z x protein_seg_cache_35m.7z.001 -o./caches     # Linux / macOS (p7zip)
# Windows: 7-Zip -> Extract Here on protein_seg_cache_35m.7z.001
```

Place the extracted cache and the training views where `run_full_training.sh` looks for them:

```
assets/
  protein_seg_cache_35m.pkl        extracted from the 7z volumes
runs/
  full_seed42/
    variants_data.pkl              renamed from variants_data_seed42.pkl
```

`DATA=data/mixed ./run_full_training.sh` then reuses both caches instead of rebuilding them.

## Environment

Two paths use the same code. The Docker path pins the CUDA/torch/Mamba combination and is the
recommended route; the native path is for users who already have a matching toolchain.

### Option A: Docker (recommended)

```bash
docker build -t duetscreen-release:1.0 .
```

The image is based on `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` and builds the Mamba kernels
during the image build (about 20 minutes once). Only `Dockerfile` and `requirements.txt` are shipped;
the image itself is built locally, and no prebuilt image or weight archive is distributed.

### Option B: native Python

```bash
python -m venv .venv && . .venv/bin/activate
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install causal-conv1d==1.5.3.post1 mamba-ssm==2.2.6.post3
```

`mamba-ssm` and `causal-conv1d` compile against the CUDA toolkit; a matching `nvcc` is required if no
prebuilt wheel fits your platform.

### Tested versions

| Component | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.8.0 (CUDA 12.8) |
| CUDA runtime / driver | 12.8 / >= 570 |
| mamba-ssm | 2.2.6.post3 |
| causal-conv1d | 1.5.3.post1 |
| transformers | 4.57.1 |
| GPU memory | >= 8 GB for training |

## Quickstart

A four-step environment check: Docker, the NT-V3 encoder folder, the image build and the package
import. It does not download data or weights and does not run a prediction.

```bash
./run_quickstart.sh                          # Linux / macOS / Git Bash
ASSETS=/path/to/assets ./run_quickstart.sh   # folder containing NTv3_8M_pre/
```

A Windows wrapper is included with the same options.

The script builds `duetscreen-release:1.0` from the included `Dockerfile` when the image is absent,
then checks that `torch` and `duetscreen` import inside the container. A ready environment ends with
`READY`.

**Prerequisites**: Docker with GPU support (`--gpus all` requires an NVIDIA driver and the NVIDIA
Container Toolkit), ~20 GB free disk, internet access for the first build and for the ESM2 download.

```
release/
  DuetScreen/          <- this package (docker build here)
  assets/
    NTv3_8M_pre/       <- Hugging Face download
```

```bash
cd DuetScreen
docker build -t duetscreen-release:1.0 .   # about 20 minutes; compiles the Mamba kernels once
```

Training and prediction commands are documented below; the wrappers in `docker/` run the same steps
inside the container. Only `Dockerfile` and `requirements.txt` are shipped; the image is built
locally.

**Scoring sequences outside the released record set**: the protein channel reads
a cache keyed by exact DNA sequence. Build a cache for your own records first
(GPU recommended; ~1–2 min for thousands of short records after the first model
download):

```bash
docker run --rm --gpus all --entrypoint python \
  -v "$PWD/assets:/assets:ro" -v "$PWD/work:/work" \
  duetscreen-release:1.0 -m duetscreen.protein_cache \
  --input /work/your_records.fasta --output /work/protein_seg_cache_custom.pkl \
  --esm2 facebook/esm2_t12_35M_UR50D --device cuda --batch-size 16
# then point DUETSCREEN_PROTEIN_SEG_CACHE at /work/protein_seg_cache_custom.pkl
```

Records without a stop-free ORF of at least 30 nt have no row; the model then
falls back to a zero protein vector, which is the trained "no protein channel"
condition. Rebuilding a cache with this module reproduces rows to fp16 precision
(max |diff| 5e-4).

## Toy validation (1 epoch)

A fixed 120/40/40 subset with precomputed training views and protein cache is distributed
separately (`DuetScreen_ToySubset.zip`, https://doi.org/10.5281/zenodo.23538086). The toy archive is
self-contained: it does **not** require the full cache record. Unpack it into `toy/` so that
`toy/data/` and `toy/cache/` exist, then run:

```bash
ASSETS=assets ./toy/run_toy.sh
```

The script trains for one epoch on the toy split and predicts the toy test set. On one GPU this
takes about two minutes and exercises data loading, view construction, the protein cache, training,
checkpoint writing and prediction end to end. Sanity values for seed 42 are validation accuracy
0.8125 and validation AUROC 0.5876. This is a pipeline check, not a performance benchmark: the
one-epoch test predictions are expected to be close to all-positive or all-negative.

## Full retraining from the public dataset

The package contains the complete training path: split preparation, the
supervised view generator (synonymous/codon-conservative and fragment-confusion
views plus random-mutation contrastive negatives), the protein-cache builder,
the trainer, and the audit checks. No paper checkpoint is required.

| Stage | Command | Output |
|---|---|---|
| training views | `python -m duetscreen.build_cache --train-fasta ... --val-fasta ...` | `variants_data.pkl` |
| protein cache | `python -m duetscreen.protein_cache --input ... --output ... --esm2 ...` | `protein_seg_cache.pkl` |
| training | `python -m duetscreen.train --records ... --ntv3-model ...` | `best_nucleotide_v3_mamba_model.pth` |

`run_full_training.sh` runs the three stages in order and skips a stage whose output already exists.

```bash
# 1) dataset (mixed pool shown; strict pool and holdouts are separate archives)
wget -O mixed_pool.zip "https://zenodo.org/records/23534656/files/mixed%20pool.zip?download=1"
unzip -o mixed_pool.zip -d data/mixed          # train.fasta, val.fasta, test.fasta

# 2) nucleotide encoder (Hugging Face)
huggingface-cli download InstaDeepAI/NTv3_8M_pre --local-dir assets/NTv3_8M_pre

# 3) build the Docker image once
docker build -t duetscreen-release:1.0 .

# 4) train (builds the record cache and, if absent, the ESM2 protein cache automatically)
DATA=data/mixed ./run_full_training.sh
SEED=43 EPOCHS=20 DATA=data/mixed ./run_full_training.sh
```

The protein cache is rebuilt on the fly with `facebook/esm2_t12_35M_UR50D` when no cache is present,
so the only assets needed are the dataset and the NT-V3 encoder folder. `ASSETS=` can
point at any directory containing `NTv3_8M_pre/`.

The script (1) builds `variants_data.pkl` with the same generator and the same
configuration fields as the released runs, (2) builds and checks the protein
cache (ESM2-35M when absent), and (3) trains
20 epochs with early stopping and keeps the best checkpoint by validation
combined accuracy. Expect several hours on one GPU for the full split
(~15k records); the paper's seed-42 run reached 0.8497 validation combined
accuracy, which is the reference range for a healthy run.

The result is **the same method, trained on your machine**: same architecture,
objective, data and hyper-parameters, but a fresh initialization, so it will not
be bit-identical to the paper's checkpoints (training is stochastic; the ten
paper seeds differ from each other in the same way).

Minimum requirements: a cache with at least 8 training records per class for the
one-step `--smoke` check; the full recipe expects the complete split.

## Required assets

| asset | role | source |
|---|---|---|
| NT-V3-Pre-8M | nucleotide encoder with trainable decoder and frozen backbone | `InstaDeepAI/NTv3_8M_pre` on Hugging Face, local folder |
| ESM2-35M (`esm2_t12_35M_UR50D`) | frozen protein encoder | `facebook/esm2_t12_35M_UR50D` on Hugging Face |
| trained checkpoint | needed for prediction; produced by the training recipe | your own run |

`assets_manifest.json` records the expected file names and SHA-256 values for the
encoder folder and, when present, the protein cache and checkpoints of a local run.

## Build the protein cache (ESM2-35M)

The protein branch is frozen: each DNA record is translated, the stop-free ORF is split into 16
segments, and ESM2-35M produces one 480-dimensional vector per segment. The cache stores these
vectors keyed by the **exact DNA sequence**. Because the key is the sequence, the cache must be built
from the same sequences that will be trained or scored, including augmented training views.

`run_full_training.sh` builds the cache automatically when it is absent. Build it manually when you
bring new data, or when you want to reuse a cache across runs.

### For training (views included)

```bash
# 1) build the supervised training views from your train/val split
python -m duetscreen.build_cache \
    --train-fasta data/mixed/train.fasta --val-fasta data/mixed/val.fasta \
    --output runs/full_seed42/variants_data.pkl --seed 42

# 2) embed every sequence in that view cache (originals + semantic/confusion views)
python -m duetscreen.protein_cache \
    --input  runs/full_seed42/variants_data.pkl \
    --output runs/full_seed42/protein_seg_cache.pkl \
    --esm2   facebook/esm2_t12_35M_UR50D \
    --device cuda --batch-size 16

# 3) point the trainer at it
export DUETSCREEN_PROTEIN_SEG_CACHE=$PWD/runs/full_seed42/protein_seg_cache.pkl
```

Passing `variants_data.pkl` (not just the FASTA) is important: views generated with another seed are
different sequences and will miss the cache.

### For new FASTA (inference)

```bash
python -m duetscreen.protein_cache \
    --input  new_records.fasta \
    --output protein_seg_cache_new.pkl \
    --esm2   facebook/esm2_t12_35M_UR50D \
    --device cuda --batch-size 16

export DUETSCREEN_PROTEIN_SEG_CACHE=$PWD/protein_seg_cache_new.pkl
```

`--esm2` accepts a Hugging Face ID (downloaded on first use) or a local folder for offline runs.
ESM2-35M is public and does not require a token. On one modern GPU this takes seconds for tens of
sequences, minutes for thousands, and a few hours for the full seed-42 view set (~59k sequences).
Records without a stop-free interval of at least 30 nt receive a zero vector; the projection learns a
"no protein channel" bias from that case.

### Check coverage before training

```bash
python tools/check_cache_coverage.py \
  --records runs/full_seed42/variants_data.pkl \
  --cache   runs/full_seed42/protein_seg_cache.pkl
```

`missing=0` means the cache covers every training view. The prebuilt cache for the released split is
available at https://doi.org/10.5281/zenodo.23537662 if you do not want to rebuild it.

### One cache or several?

Each invocation writes **one** cache for its input; nothing is merged automatically.

- training only → build from `variants_data.pkl`;
- inference only → build from the FASTA you will score;
- both → either build once from a combined input, or merge the two pickles (keys are DNA strings):

```python
import pickle

merged = {}
for path in ["protein_seg_cache_train.pkl", "protein_seg_cache_test.pkl"]:
    merged.update(pickle.load(open(path, "rb")))
pickle.dump(merged, open("protein_seg_cache_35m.pkl", "wb"), protocol=4)
print(len(merged), "sequences")
```

The released 979 MB cache is such a superset: it was built once from a combined record set and covers
59,398 sequences (training views plus test/species/family cohorts). It is not the result of two runs
merging themselves.

### Why a merged cache is not test-set leakage

A merged cache contains ESM2 embeddings for test/holdout sequences, but it is a **lookup table**, not
a training set:

- training iterates only the records stored in `variants_data.pkl` (train/val); each batch calls
  `lookup(seqs)` for those sequences, and extra keys in the cache are never read;
- the cache stores no labels, and ESM2-35M is frozen, so no gradient reaches it;
- test/holdout FASTAs are used only by evaluation, and the trainer's leakage audit requires an
  external test file and checks that it does not overlap the training split.

A single merged cache is therefore safe and convenient: it does not expose test protein features to
the training loop. Separate caches (train-only and test-only) are equally valid if you prefer that
layout.

You can verify the split directly:

```python
import pickle
from Bio import SeqIO

records = pickle.load(open("runs/full_seed42/variants_data.pkl", "rb"))
train = {r["original_seq"].upper() for r in records}
test = {str(r.seq).upper() for r in SeqIO.parse("data/mixed/test.fasta", "fasta")}
print("overlap", len(train & test))      # expected: 0
```

## Predict

The checkpoint path selects the seed: point `model_path` at
`runs/full_seed<SEED>/seed<SEED>/best_nucleotide_v3_mamba_model.pth` (training output) or at
`assets/checkpoints/seed42/best_nucleotide_v3_mamba_model.pth` (released checkpoint). There is no
automatic seed lookup and no need to `cd` when the path is correct; the loader prints the seed stored
inside the checkpoint.

```python
from duetscreen.predict import SimpleNTv3Predictor

p = SimpleNTv3Predictor(
    'runs/full_seed42/seed42/best_nucleotide_v3_mamba_model.pth',
    output_dir='out', threshold=0.5,
    ntv3_model='assets/NTv3_8M_pre')
p.predict('records.fasta', batch_size=64, threshold=0.5)
```

```bash
python -m duetscreen.predict --fasta records.fasta \
  --model runs/full_seed42/seed42/best_nucleotide_v3_mamba_model.pth \
  --ntv3-model assets/NTv3_8M_pre --output-dir out --threshold 0.5
```

For a labelled test FASTA, `./run_eval.sh` at the repository root wraps this call and prints
accuracy, F1, AUROC and AUPRC; see **First-time setup** for the required `MODEL`, `DATA` and `CACHE`
variables.

Outputs per-record scores plus a metrics block when labels are present in the
FASTA headers (`|label:1` / `|label:0`).

| Argument | Required | Meaning |
|---|---|---|
| `--fasta` | yes | input FASTA |
| `--model` | yes | checkpoint path (selects the seed) |
| `--ntv3-model` | no | encoder folder when the checkpoint stores a path from another host |
| `--threshold` | no | classification threshold (default 0.5) |
| `--batch-size` | no | inference batch size (default 64) |
| `--output-dir` | no | output directory |

## Train

Training expects a pickle holding the prepared record list. Each record is a
dict with at least:

```
original_seq           str, DNA sequence
original_label         int (0/1)          # 'label' is accepted as a fallback
segment_id             str (optional)
positive_views         list[str] (optional)  # supervised alternative views
view_types             list[str]          # 'semantic' / 'confusion', same order
random_mutation_sequence  str (optional)  # random-mutation view
split                  'train' | 'val' | 'test'
```

```bash
python -m duetscreen.train \
    --records records.pkl --output-dir runs/seed42 \
    --ntv3-model /path/to/NTv3_8M_pre --seed 42 \
    --epochs 20 --patience 5 --batch-size 16 --max-seq-len 2048
python -m duetscreen.train ... --smoke     # one forward/backward check
```

The trainer writes `model_audit.json` (parameter budget: downstream ≤ 4M),
`optimizer_audit.json`, `progress.jsonl`, `best_nucleotide_v3_mamba_model.pth`
(selected by validation combined accuracy, first maximum, 1-based epoch) and
`complete.json`.

## Output files

| Stage | File | Description |
|---|---|---|
| training | `variants_data.pkl` | supervised views built from train/val |
| training | `protein_seg_cache.pkl` | 16-segment ESM2-35M cache |
| training | `best_nucleotide_v3_mamba_model.pth` | checkpoint selected by validation combined accuracy |
| training | `nucleotide_v3_training_log.json` | per-epoch metrics and selection record |
| training | `progress.jsonl`, `complete.json` | step log and completion record |
| prediction | per-record scores | CSV written to `--output-dir`; metrics added when labels are present |

## Configuration reference

| Argument | Default | Meaning |
|---|---|---|
| `--records` | required | training cache (`variants_data.pkl`) built by `duetscreen.build_cache` |
| `--output-dir` | required | run directory; receives checkpoint, audits and logs |
| `--ntv3-model` | required | local folder of `NTv3_8M_pre` |
| `--seed` | 42 | training seed |
| `--epochs` | 20 | maximum epochs |
| `--patience` | 5 | early-stopping patience on validation combined accuracy |
| `--batch-size` | 16 | per-step batch size |
| `--max-seq-len` | 2048 | maximum input length |
| `--smoke` | off | one forward/backward step, for environment checks |

The protein cache is passed through the environment variable
`DUETSCREEN_PROTEIN_SEG_CACHE`; `run_full_training.sh` builds it automatically when absent.

**Hardware and runtime.** One GPU with at least 8 GB of memory. Building the protein cache for the
full split takes a few minutes on GPU; 20 training epochs take several hours depending on the GPU.

## Released configuration

See `configs/duetscreen_release.json`: NT-V3-Pre-8M with 2,763,264 trainable
decoder parameters, 2,987,761 downstream parameters (5,751,025 trainable in
total), ESM2-35M frozen, 16 segments, width 480 -> 256, 4 segment-attention
heads, batch 16, at most 20 epochs with patience 5, maximum 2,048 bp input.

## FAQ

**CUDA out of memory.** Reduce `--batch-size` or `--max-seq-len`.

**Mamba kernel fails to build.** Use the Docker image; it pins torch 2.8.0 with CUDA 12.8 and builds
`mamba-ssm`/`causal-conv1d` once inside the image.

**Hugging Face download is slow or requires a token.** Set `HF_ENDPOINT` to a mirror and run
`huggingface-cli login` before downloading `InstaDeepAI/NTv3_8M_pre`.

**Training stops early.** Early stopping uses validation combined accuracy with patience 5; see
`nucleotide_v3_training_log.json` for the selected epoch.

**Where is the dataset?** Zenodo DOI [10.5281/zenodo.23534656](https://doi.org/10.5281/zenodo.23534656).

## Citation and license

If you use DuetScreen, please cite the manuscript (citation details to be added upon publication)
and the Zenodo dataset record.

See `LICENSE` (MIT for this code) and `THIRD_PARTY.md` for the licences of the
encoder weights and dependencies. Encoder weights retain their upstream
licences. The separate assets archive includes the NT-V3 checkpoint; encoder
weights are not embedded in this source-code archive.
