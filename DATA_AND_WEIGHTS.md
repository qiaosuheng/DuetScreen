# Data and encoder assets

This repository contains source code only. The benchmark dataset is published on Zenodo; the
seed-42 checkpoint and the NT-V3 encoder are archived in a separate Zenodo record. New checkpoints
are produced locally by the training recipe.

## Where things live

| Resource | Location |
|---|---|
| Benchmark dataset (mixed, strict, holdouts) | Zenodo: https://doi.org/10.5281/zenodo.23534656 |
| Seed-42 checkpoint and NT-V3 encoder | Zenodo: https://doi.org/10.5281/zenodo.23535423 |
| Nucleotide encoder | `NTv3_8M_pre.zip` in the checkpoint record, or `InstaDeepAI/NTv3_8M_pre` on Hugging Face |
| Protein encoder | Hugging Face: `facebook/esm2_t12_35M_UR50D` (downloaded automatically) |
| Additional checkpoints | produced locally by `run_full_training.sh` |

## Dataset layout after download

```
data/
  mixed/train.fasta, val.fasta, test.fasta
  strict/train.fasta, val.fasta, test.fasta
  holdout/generalization_species.fasta, generalization_family.fasta
```

Record IDs carry `pos_` / `neg_` prefixes; the training cache builder reads labels from them.

## Encoder layout

```
assets/
  NTv3_8M_pre/          # huggingface-cli download InstaDeepAI/NTv3_8M_pre --local-dir ...
```

`run_full_training.sh` looks for `assets/NTv3_8M_pre` (or `ASSETS=<dir>` containing it). The ESM2
protein cache is rebuilt automatically from `facebook/esm2_t12_35M_UR50D` if no cache is present.

## Training from scratch

```bash
docker build -t duetscreen-release:1.0 .
DATA=data/mixed ./run_full_training.sh
```

No checkpoint is needed to start training. The script builds the training-view cache, the protein
cache and then trains for 20 epochs with early stopping.

## Expected files of a local run

```
runs/full_seed42/
  variants_data.pkl
  protein_seg_cache.pkl
  seed42/best_nucleotide_v3_mamba_model.pth
  seed42/progress.jsonl
```

`assets_manifest.json` records the expected names and hashes for the encoder folder and, when
present, the protein cache and checkpoints of a local run.

## Licence

Code: MIT (`LICENSE`). Third-party components: see `THIRD_PARTY.md`. Dataset: CC BY 4.0 for
author-created curation and splits; upstream sequence sources retain their own terms.
