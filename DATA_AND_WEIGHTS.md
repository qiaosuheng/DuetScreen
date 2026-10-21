# Data and model assets

This repository contains source code only. The benchmark dataset and the released model assets are
archived together on Zenodo; numerical source data for figures and tables accompany the journal
submission.

## Where things live

| Resource | Location | Size |
|---|---|---|
| Benchmark dataset | Zenodo, DOI `DOI_PLACEHOLDER`, `DuetScreen_Benchmark_v1.0.zip` | 382 MB |
| Released checkpoints, NT-V3 weights, protein cache, smoke inputs | Zenodo, DOI `DOI_PLACEHOLDER`, `DuetScreen_Model_Assets_v1.0.zip` | 1.8 GB |
| Numerical source data for figures and tables | Journal submission / Source Data archive | 29 MB |

## Layout after downloading the model assets

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

`run_quickstart.sh` / `run_quickstart.ps1` look for `assets/` next to the repository root and will
unpack `DuetScreen_Model_Assets_v1.0.zip` if it is present there.

## Benchmark

The Zenodo archive contains the source pool (18,740 records), mixed-pool and strict-pool splits,
species and family holdouts, per-record evidence tables and the dataset build pipeline. Unpack it
separately; it is not required for the quickstart, which uses the bundled smoke inputs.

## Checksums

- Model assets: per-file SHA-256 in `assets_manifest.json`; archive hash in the Zenodo `SHA256SUMS.txt`.
- Benchmark: `MANIFEST.json` inside the archive and `SHA256SUMS.txt` on the Zenodo record.

## Licence

Code: MIT (`LICENSE`). Third-party components: see `THIRD_PARTY.md`.
Dataset and trained checkpoints: CC BY 4.0 for author-created content; upstream sequence and model
sources retain their own terms.
