# Data and model assets

This repository contains source code only. The benchmark, the released checkpoints and the
precomputed protein-segment cache are distributed as separate archives.

## Archives

| Archive | Contents | Size (approx.) |
|---|---|---|
| `DuetScreen_Benchmark_v1.0.zip` | source pool, mixed and strict splits, species/family holdouts, per-record evidence tables, build pipeline, annotation caches | 382 MB |
| `DuetScreen_Model_Assets_v1.0.zip` | ten released checkpoints (seeds 42-51), NT-V3-Pre-8M weights, 16-segment protein cache, smoke inputs | 1.8 GB |
| `DuetScreen_Source_Data_v1.0.zip` | numerical results, prediction arrays and configuration records supporting the manuscript figures and tables | 262 MB |

The archives are deposited on Zenodo. Replace `DOI_PLACEHOLDER` below with the record DOI once published.

```
https://doi.org/DOI_PLACEHOLDER
```

## Layout after download

```
release/
  DuetScreen/                     <- this repository
  assets/
    checkpoints/seed42..51/best_nucleotide_v3_mamba_model.pth
    NTv3_8M_pre/
    protein_seg_cache_35m.pkl
    mini_train_cache.pkl
    smoke.fasta
```

`run_quickstart.sh` / `run_quickstart.ps1` look for `assets/` next to the repository root and
will unpack `DuetScreen_Model_Assets_v1.0.zip` automatically if it is present.

## Checksums

Per-file SHA-256 values are listed in `assets_manifest.json` (checkpoints, encoder, caches) and in
the Zenodo record (`SHA256SUMS.txt`). Verify after download before running the quickstart.

## Licence

Code: MIT (`LICENSE`). Third-party components: see `THIRD_PARTY.md`.
Data and model weights: see the Zenodo record licence; upstream sequence sources retain their own terms.
