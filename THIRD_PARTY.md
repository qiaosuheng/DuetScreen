# Third-party components

This package does not redistribute pretrained weights. It references:

| component | version / revision | licence |
|---|---|---|
| Nucleotide Transformer v3 (NT-V3-Pre-8M) | see `assets_manifest.json` (`InstaDeepAI/NTv3_8M_pre`) | follow the upstream model card; non-commercial research terms apply |
| ESM2 (`esm2_t12_35M_UR50D`) | `facebook/esm2_t12_35M_UR50D` | MIT (per upstream model card) |
| transformers | 4.57.1 | Apache-2.0 |
| mamba-ssm / causal-conv1d | 2.2.6.post3 / 1.5.3.post1 | Apache-2.0 / BSD-3-Clause |
| PyTorch | 2.8.0+cu128 | BSD-3-Clause |
| Biopython | >=1.81 | BSD-3-Clause |
| scikit-learn | >=1.4 | BSD-3-Clause |

Before redistributing a trained checkpoint, confirm that the upstream licences
of NT-V3 and ESM2 permit the intended distribution and use.
