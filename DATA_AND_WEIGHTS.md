# Data and model assets

This repository contains **source code**. The public assets are distributed as four
Zenodo records; files created by a local run are listed separately at the end.

## Public download records

| Resource | Record | Notes |
|---|---|---|
| Benchmark splits | https://doi.org/10.5281/zenodo.23534656 | mixed, strict, holdout FASTA |
| Seed-42 checkpoint + NT-V3 encoder | https://doi.org/10.5281/zenodo.23535423 | checkpoint 90.2 MB; encoder archive 28.6 MB |
| Shared training views + protein cache | https://doi.org/10.5281/zenodo.23537662 | view cache 60.0 MB; protein cache 32 volumes, 979 MB extracted |
| Toy subset | https://doi.org/10.5281/zenodo.23538086 | 120/40/40 demo with its own caches |

Only the **seed-42** checkpoint is public. Seeds 43–51 are not part of the public
weight record. The training-view cache is shared: it was generated once with seed 42
and reused for every training seed.

Exact sizes and SHA-256 values are in `assets_manifest.json`.

## Placement

```
data/
  mixed/train.fasta, val.fasta, test.fasta
  strict/train.fasta, val.fasta, test.fasta
  holdout/generalization_species.fasta, generalization_family.fasta

assets/
  NTv3_8M_pre/                       extracted from NTv3_8M_pre.zip
  checkpoints/seed42/best_nucleotide_v3_mamba_model.pth
  protein_seg_cache_35m.pkl          extracted from the 32 volumes

runs/
  shared/
    variants_data.pkl                renamed from variants_data_seed42.pkl
```

Record IDs carry `pos_` / `neg_` prefixes; the training-cache builder reads labels
from them. `run_full_training.sh` looks for `assets/NTv3_8M_pre` (or `ASSETS=<dir>`
containing it) and the shared cache under `runs/shared/`.

## Produced locally (not fixed release files)

| Path | Produced by |
|---|---|
| `runs/shared/variants_data.pkl` | `duetscreen.build_cache` (deterministic only for a fixed seed and code version) |
| `runs/shared/protein_seg_cache.pkl` | `duetscreen.protein_cache` from the view cache |
| `runs/full_seed<ID>/seed<ID>/best_nucleotide_v3_mamba_model.pth` | `run_full_training.sh`; a local training output, not a release hash |
| `out_eval/` | `run_eval.sh` predictions and metrics |

`assets_manifest.json` separates these local outputs from the public assets.

## Training and evaluation

```bash
docker build -t duetscreen-release:1.0 .
DATA=data/mixed ./run_full_training.sh
```

No checkpoint is needed to start training. If the view cache is absent the script
builds it once under `runs/shared/`; the protein cache is built only when no
covering cache is present.

The protein cache is a frozen ESM2 lookup table and also contains test/holdout
embeddings. Training reads only keys for sequences in `variants_data.pkl`; extra
entries are never used, so this is not test-set leakage.

## Licence

Code: MIT (`LICENSE`). Author-created data, splits, caches and checkpoints: CC BY
4.0 as stated on the Zenodo records. Upstream components (NT-V3, ESM2, sequence
databases) retain their own terms; see `THIRD_PARTY.md`.
