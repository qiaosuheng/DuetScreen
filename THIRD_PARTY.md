# Third-party components and licensing

This repository contains **code** (MIT, see `LICENSE`). It does not relicense
upstream model weights or third-party sequence data.

## What is distributed where

| Material | Where | Licence |
|---|---|---|
| DuetScreen source code | this repository | MIT |
| Benchmark curation, splits, evidence tables | Zenodo record 23534656 / submission archive | CC BY 4.0 for author-created content |
| Seed-42 checkpoint and NT-V3 encoder archive | Zenodo record 23535423 | CC BY 4.0 for the author-created checkpoint; the NT-V3 weights retain their upstream terms |
| Shared training views and protein cache | Zenodo record 23537662 | CC BY 4.0 for author-created caches |
| Toy subset | Zenodo record 23538086 | CC BY 4.0 for author-created content |

The paper's other seed checkpoints (43–51) are **not** part of the public weight
record.

## Upstream components

| Component | Version / source | Upstream terms |
|---|---|---|
| Nucleotide Transformer v3 (NT-V3-Pre-8M) | `InstaDeepAI/NTv3_8M_pre` on Hugging Face; archive in record 23535423 | follow the upstream model card; non-commercial research terms apply |
| ESM2 (`esm2_t12_35M_UR50D`) | `facebook/esm2_t12_35M_UR50D`, downloaded at runtime | MIT per the upstream model card |
| transformers | 4.57.1 | Apache-2.0 |
| mamba-ssm / causal-conv1d | 2.2.6.post3 / 1.5.3.post1 | Apache-2.0 / BSD-3-Clause |
| PyTorch | 2.8.0+cu128 | BSD-3-Clause |
| Biopython | >=1.81 | BSD-3-Clause |
| scikit-learn | >=1.4 | BSD-3-Clause |

Before redistributing the checkpoint or the NT-V3 archive, confirm that the
upstream NT-V3 terms permit the intended distribution and use. This table is a
release note, not a legal conclusion; unresolved licence questions are listed in
the internal release report.
