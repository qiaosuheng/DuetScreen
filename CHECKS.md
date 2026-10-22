# Release validation and verification scope

Documentation revision: 2026-10-22 (release tag `v1.0.24`). The execution records
quoted below were produced on 2026-10-17 with the release build of that date. They
were **not** re-executed for this documentation revision; treat them as historical
evidence, not as a fresh validation of the current commit.

## Verification levels

| Level | What it can establish | Status for this revision |
|---|---|---|
| Documentation, links, file manifest, Git file modes | Release-material consistency | Checked on 2026-10-22 |
| Environment build and package import | Basic usability of the pinned environment | Historical record, 2026-10-17 |
| Small execution / prediction checks | That the recorded commands ran in the recorded environment | Historical record, 2026-10-17 |
| Full training to convergence and paper-table reproduction | Numerical reproduction of the paper | **Not re-run** for this revision |

## Current public artifacts

| Resource | Record | Public content |
|---|---|---|
| Benchmark splits | https://doi.org/10.5281/zenodo.23534656 | mixed, strict and holdout FASTA archives |
| Seed-42 checkpoint and NT-V3 encoder | https://doi.org/10.5281/zenodo.23535423 | `best_nucleotide_v3_mamba_model.pth`, `NTv3_8M_pre.zip` |
| Shared training views and protein cache | https://doi.org/10.5281/zenodo.23537662 | `variants_data_seed42.pkl`, 32 protein-cache volumes |
| Toy subset | https://doi.org/10.5281/zenodo.23538086 | `DuetScreen_ToySubset.zip` |

**Only the seed-42 checkpoint is publicly distributed.** The paper's training
seeds 42–51 are described in Source Data; seeds 43–51 are not part of the public
weight record. The view cache is shared across training seeds: it was generated
once with seed 42 and reused for every training seed.

File names, sizes and hashes are listed in `assets_manifest.json`.

## Current quickstart scope

`./run_quickstart.sh` performs an environment check only:

1. Docker is installed and the daemon is running;
2. `assets/NTv3_8M_pre` exists;
3. the image `duetscreen-release:1.0` is built if absent;
4. `import torch, duetscreen` succeeds inside the container.

It prints `READY` on success. It does **not** run a prediction, compute metrics, or
train a model. Use `./run_eval.sh` for labelled evaluation, `./toy/run_toy.sh` for
the toy workflow, and `./run_full_training.sh` for full training.

## Historical execution records (2026-10-17)

These checks were recorded during release preparation. They refer to the
17 October 2026 build, before the documentation revisions of 22 October 2026.

| Check | Recorded result |
|---|---|
| Container build and dependency imports | Passed |
| Parameter-count audit | Matched the counts below |
| Prediction on eight example records | Passed; maximum absolute difference 4.5e-9 from the stored example scores |
| Single-step training smoke check | Passed; finite loss, backward pass and parameter audit completed |
| Separate one-epoch execution check | Training, validation and checkpoint saving completed |

The 32-record smoke cache and the eight-record example FASTA were inputs to those
historical checks. They are not required for the current quickstart, evaluation
or full-training entries.

## Parameter counts

| Quantity | Observed | Paper |
|---|---:|---:|
| NT-V3 registered parameters | 7,692,475 | 7,692,475 |
| NT-V3 decoder trainable parameters | 2,763,264 | 2,763,264 |
| Downstream parameters | 2,987,761 | 2,987,761 |
| Trainable total | 5,751,025 | 5,751,025 |

These counts confirm agreement in parameter budgets; numerical agreement is
reported separately for the historical example-prediction check.

## Prediction comparison scope (historical)

The eight-example check used the seed-42 checkpoint and the reference scores in
`tools/expected_smoke.json`. The prediction CLI evaluates each record's 2,048 bp
prefix, which is what the released evaluation pipeline did.

A full-split comparison was executed afterwards on all 1,677 test records: the
package reproduced the released metrics to the fourth decimal (ACC 0.8748,
F1 0.7139, AUROC 0.9216 vs 0.9217), with four records differing by more than 0.01
owing to cross-GPU floating-point drift. Cohort-specific numbers in the paper
(species/family holdouts) follow their own evaluation protocols.

These are historical results. The current documentation revision did not repeat
them; `./run_eval.sh` is the current entry point for labelled evaluation.

## Training execution checks (historical)

The single-step smoke check returned a finite loss of approximately 0.712 and
completed backpropagation. Recorded peak GPU memory was 6.35 GiB.

The separate seed-42 execution check completed **one epoch** in 8 min 42 s on the
release host. Its recorded training accuracy was 0.742, validation accuracy 0.760
and validation combined accuracy 0.7712. These values describe the execution
check only; they are not benchmark results or the validation scores of the
published checkpoint.

The associated cache checks recorded 16,738 prepared records and protein-cache
coverage for 16,735 of them; the remaining three used the configured
missing-protein fallback. One-epoch checkpoint outputs are not distributed.

## Post-cleanup re-verification (historical)

Legacy candidate implementations were removed before release (one shared network
library replaced several historical stacks). The full path was re-executed
afterwards:

| Check | Result |
|---|---|
| Released checkpoint, complete test split | ACC 0.8748 / F1 0.7139 / AUROC 0.9216 |
| One-epoch retrain on the released cache | 8 min 54 s; train ACC 0.7536 / validation ACC 0.7802; combined 0.7691 |
| A/B against the pre-cleanup one-epoch run | AUROC 0.7933 vs 0.7917; ACC@0.5 0.7847 vs 0.7883 |
| Inference with the freshly trained checkpoint | Runs end to end (complete split, ACC@0.5 0.788) |

One-epoch probabilities are under-confident, so F1 at a fixed 0.5 threshold is low;
the training log records the validation-selected threshold. The distributed
seed-42 20-epoch checkpoint is unaffected. One-epoch checkpoints are not
distributed.

## Not verified in this revision

- Full 20-epoch training on the complete split, and reproduction of every paper
  table, were not re-run for the current commit.
- Zenodo files were not re-downloaded and re-hashed during this documentation
  pass; hashes in `assets_manifest.json` refer to the files used to prepare the
  release.
- macOS with CUDA and CPU-only training are not supported paths and were not
  validated.
- Checkpoints for seeds 43–51 are not publicly distributed; no claim is made that
  all ten paper checkpoints are downloadable.
