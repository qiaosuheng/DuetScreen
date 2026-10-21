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

## Contents

```
duetscreen/            model, encoder, data, trainer, protein cache, predict CLI
configs/               released configuration snapshot
assets_manifest.json   required third-party assets, versions and hashes
Dockerfile             reproducible CPU/GPU environment (see notes inside)
```

## Dataset, checkpoints and scope

The benchmark dataset is archived on Zenodo: **https://doi.org/10.5281/zenodo.23534656**
(mixed pool, strict pool and the species/family holdout sets, `pos_`/`neg_` labels in the FASTA IDs).

**Trained checkpoints from the paper are not redistributed.** This repository provides the complete
training path, so a checkpoint can be trained from scratch on the published dataset with the same
architecture, objective and hyper-parameters. Training is stochastic: a fresh run will not be
bit-identical to the paper's seeds, but it follows the same recipe and reaches the same performance
range (the seed-42 reference reached 0.8497 validation combined accuracy).

Encoder weights are obtained from their upstream releases (see below); they are not embedded in
this repository.

## Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` provide the Mamba2 kernels. Prebuilt wheels exist
for common CUDA/PyTorch combinations; otherwise they compile against the CUDA
toolkit (the Dockerfile shows one working recipe).

## Quickstart (optional: requires a trained checkpoint)

The quickstart is a self-check for the code path. It compares predictions against stored example
scores, so it needs a checkpoint. Either train one first (see **Full retraining**) and pass its path
through `MODEL=`, or skip this section and go directly to training.

**One command runs the quickstart checks** (environment check → build → predict → compare
scores against the recorded reference → one-step training check → PASS/FAIL):

```bash
./run_quickstart.sh            # Linux / macOS / Git Bash
.\run_quickstart.ps1           # Windows PowerShell
NO_GPU=1 ./run_quickstart.sh   # CPU prediction only (training skipped)
```

Arrange the working folder so the script finds the encoder and your checkpoint:

```
release/
  DuetScreen/   <- this package
  assets/       <- checkpoints/, NTv3_8M_pre/, protein_seg_cache_35m.pkl, ...
```

The quickstart compares predictions with the eight stored example scores and
runs a single training-step functionality check. It does not retrain the model; prediction uses the checkpoint you provide. A separate one-epoch execution check was also completed
during release validation; see `CHECKS.md`. Longer training is separate from
the quickstart, and the data archive is supplied separately from this code.

Success ends with `ALL CHECKS PASSED` when the executed checks pass. The score
comparison uses a tolerance of 0.002 against `tools/expected_smoke.json`; the
recorded maximum difference on the release host was 4.5e-9. For individual
steps, follow the instructions below; see
`TROUBLESHOOTING.md` for common failures. If you use an AI coding assistant, it
can walk you through this README — a ready-to-paste prompt is in
docs/ASSISTED_SETUP.md.

`CHECKS.md` distinguishes the checks actually executed from the longer
training instructions provided here.

**Prerequisites**: Docker with GPU support (`--gpus all` requires an NVIDIA
driver and the NVIDIA Container Toolkit), ~20 GB free disk, internet access for
the first build and for the ESM2 download. All commands are run from the folder
that contains `assets/` and the unpacked package:

```
release/
  DuetScreen/          <- this package (docker build here)
  assets/              <- NTv3_8M_pre/ plus your trained checkpoint
```

```bash
cd DuetScreen
# 1) build (≈20 min; compiles the Mamba kernels once)
docker build -t duetscreen-release:1.0 .

# 2) assets (once, from the release folder)
#    assets/
#      NTv3_8M_pre/                  NT-V3-Pre-8M checkpoint folder
#      protein_seg_cache_35m.pkl    shipped 16-segment cache (released record set)
#      mini_train_cache.pkl         shipped (training smoke, 32 records)
#      smoke.fasta                  shipped (8 prediction records)
#      checkpoints/my_run/best_nucleotide_v3_mamba_model.pth

cd ..   # back to release/, so $PWD/assets works below

# 3) predict (override the training entrypoint)
docker run --rm --gpus all --entrypoint python \
  -v "$PWD/assets:/assets:ro" -v "$PWD/out:/out" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl \
  duetscreen-release:1.0 -m duetscreen.predict \
  --fasta /assets/smoke.fasta \
  --model /assets/checkpoints/my_run/best_nucleotide_v3_mamba_model.pth \
  --output-dir /out --threshold 0.5 --batch-size 4 \
  --ntv3-model /assets/NTv3_8M_pre

# 4) training sanity check (one forward/backward step)
docker run --rm --gpus all \
  -v "$PWD/assets:/assets:ro" -v "$PWD/work:/work" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl \
  duetscreen-release:1.0 --smoke \
  --records /assets/mini_train_cache.pkl --output-dir /work/smoke \
  --ntv3-model /assets/NTv3_8M_pre

# 5) full training (prepared cache; 20 epochs, early stop patience 5)
docker run --rm --gpus all \
  -v "$PWD/assets:/assets:ro" -v "$PWD/work:/work" \
  -e DUETSCREEN_PROTEIN_SEG_CACHE=/assets/protein_seg_cache_35m.pkl \
  duetscreen-release:1.0 \
  --records /work/records.pkl --output-dir /work/seed42 \
  --ntv3-model /assets/NTv3_8M_pre --seed 42 \
  --epochs 20 --patience 5 --batch-size 16 --max-seq-len 2048
```

Convenience wrappers are in `docker/`. Training requires a GPU; prediction also
runs on CPU (drop `--gpus all`; slower).

The prediction CLI scores the first 2,048 bp of each record, as the released
evaluation pipeline did. On the full test split the package reproduces the
released metrics to the fourth decimal (ACC 0.8748, F1 0.7139, AUROC 0.9216 vs
0.9217 reported); a handful of records differ by up to 0.03 from cross-GPU
floating-point drift. Cohort-specific published numbers (species/family
holdouts) use their own protocols — see `CHECKS.md` for the comparison scope.

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
condition. Rebuilding a cache with this module reproduced the shipped rows to
fp16 precision (max |diff| 5e-4).

## Full retraining from the public dataset

The package contains the complete training path: split preparation, the
supervised view generator (synonymous/codon-conservative and fragment-confusion
views plus random-mutation contrastive negatives), the protein-cache builder,
the trainer, and the audit checks. No paper checkpoint is required.

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

The protein cache is rebuilt on the fly with `facebook/esm2_t12_35M_UR50D` when the shipped cache
is absent, so the only assets needed are the dataset and the NT-V3 encoder folder. `ASSETS=` can
point at any directory containing `NTv3_8M_pre/`.

The script (1) builds `variants_data.pkl` with the same generator and the same
configuration fields as the released runs, (2) checks that the shipped protein
cache covers every record (rebuilding it with ESM2-35M if not), and (3) trains
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
| trained checkpoint (optional) | required only for prediction/quickstart, produced by the training recipe | your own run |

`assets_manifest.json` records the expected file names and SHA-256 values for the
encoder folder and, when present, the protein cache and checkpoints of a local run.

## Build the protein cache

The protein channel reads a cache of 16-segment ESM2-35M embeddings keyed by DNA
sequence. A matching cache ships in the release assets; rebuild it only for new
record sets (GPU recommended):

```bash
python -m duetscreen.protein_cache \
    --input  records.fasta \
    --output protein_seg_cache_35m.pkl \
    --esm2   /assets/esm2_t12_35M_UR50D   # or facebook/esm2_t12_35M_UR50D
export DUETSCREEN_PROTEIN_SEG_CACHE=$PWD/protein_seg_cache_35m.pkl
```

Records without a stop-free interval of at least 30 nt receive a zero vector;
the projection learns a "no protein channel" bias from that case.

## Predict

```bash
python - <<'PY'
from duetscreen.predict import SimpleNTv3Predictor
p = SimpleNTv3Predictor('best_nucleotide_v3_mamba_model.pth', output_dir='out', threshold=0.5)
p.predict('records.fasta', batch_size=64, threshold=0.5)
PY
```

Outputs per-record scores plus a metrics block when labels are present in the
FASTA headers (`|label:1` / `|label:0`).

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

## Released configuration

See `configs/duetscreen_release.json`: NT-V3-Pre-8M with 2,763,264 trainable
decoder parameters, 2,987,761 downstream parameters (5,751,025 trainable in
total), ESM2-35M frozen, 16 segments, width 480 -> 256, 4 segment-attention
heads, batch 16, at most 20 epochs with patience 5, maximum 2,048 bp input.

## Data and model assets

The benchmark dataset is archived on Zenodo: **https://doi.org/10.5281/zenodo.23534656**.
Encoder weights are downloaded from Hugging Face (`InstaDeepAI/NTv3_8M_pre`,
`facebook/esm2_t12_35M_UR50D`). The paper's trained checkpoints are not redistributed; train a
checkpoint with the recipe above. See `DATA_AND_WEIGHTS.md` for the expected layout. This repository
contains source code only; no weights or sequence data are committed here.

## Citation and license

If you use DuetScreen, please cite the manuscript (citation details to be added upon publication)
and the Zenodo record for the data and model assets.

See `LICENSE` (MIT for this code) and `THIRD_PARTY.md` for the licences of the
encoder weights and dependencies. Encoder weights retain their upstream
licences. The separate assets archive includes the NT-V3 checkpoint; encoder
weights are not embedded in this source-code archive.
