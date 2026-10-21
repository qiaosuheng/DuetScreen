# Release validation

This document summarises the recorded release checks performed on 17 October
2026. The checked environment used Python 3.11, PyTorch 2.8.0+cu128, CUDA 12.8,
transformers 4.57.1, mamba-ssm 2.2.6.post3 and causal-conv1d 1.5.3.post1.
Usage instructions are in `README.md`.

## Published model weights

The assets archive contains the ten paper checkpoints for seeds 42–51. Their
SHA-256 values were checked against `assets_manifest.json` and Source Data,
`09_Configuration/final_model_run_index.json`, and match all ten recorded runs.
For example, the released seed-42 checkpoint was selected at epoch 18 of its
20-epoch paper run.

The **one-epoch execution check below is a separate validation run**. Its
generated checkpoints are not included in the assets archive and are not the
models used for the paper's reported results. The 32-record training-smoke cache
contains example inputs, not model weights.

## Checks completed

| Check | Recorded result |
|---|---|
| Container build and dependency imports | Passed |
| Parameter-count audit | Matched the counts below |
| Prediction on eight example records | Passed; maximum absolute difference 4.5e-9 from the stored example scores, within the quickstart tolerance of 0.002 |
| Single-step training smoke check | Passed; finite loss, backward pass and parameter audit completed |
| Separate one-epoch execution check | Training, validation and checkpoint saving completed |

The one-command quickstart performs the example prediction and single-step
training checks. It does not run the separate one-epoch check or reproduce all
benchmark experiments. CPU-only quickstart mode skips the training check.

## Parameter counts

| Quantity | Observed | Paper |
|---|---:|---:|
| NT-V3 registered parameters | 7,692,475 | 7,692,475 |
| NT-V3 decoder trainable parameters | 2,763,264 | 2,763,264 |
| Downstream parameters | 2,987,761 | 2,987,761 |
| Trainable total | 5,751,025 | 5,751,025 |

These counts confirm agreement in parameter budgets; numerical agreement is
reported separately for the example prediction check.

## Prediction comparison scope

The eight-example check uses the released seed-42 checkpoint and the reference
scores in `tools/expected_smoke.json`. The prediction CLI evaluates each
record's 2,048 bp prefix, which is what the released evaluation pipeline did.

A full-split comparison was executed afterwards (section 3b): on all 1,677 test
records the package reproduces the released metrics to the fourth decimal
(ACC 0.8748, F1 0.7139, AUROC 0.9216 vs 0.9217), with four records differing by
more than 0.01 owing to cross-GPU floating-point drift. Cohort-specific numbers
in the paper (e.g. species/family holdouts) follow their own evaluation
protocols; reproduce them with the same cohorts and protocol.

## Training execution checks

The single-step smoke check returned a finite loss of approximately 0.712 and
completed backpropagation. Recorded peak GPU memory was 6.35 GiB.

The separate seed-42 execution check completed **one epoch** in 8 min 42 s on
the release host. Its recorded training accuracy was 0.742, validation accuracy
0.760 and validation combined accuracy 0.7712. Validation metrics and checkpoint
files were produced successfully. These values describe the execution check
only; they are not benchmark results or the validation scores of the published
checkpoints.

The associated cache checks recorded 16,738 prepared records and protein-cache
coverage for 16,735 of them; the remaining three used the configured missing-
protein fallback. The one-epoch checkpoint outputs are not distributed.

## Validation scope and file identity

Release validation comprised the checks above. It did not repeat a full
training run to convergence or re-evaluate every benchmark table entry.

Base-image versions are pinned in `Dockerfile`, and Python dependencies are
pinned in `requirements.txt`. Asset identities are recorded in
`assets_manifest.json`; the submission-level `SHA256SUMS.txt` identifies the
delivered files.

## Post-cleanup re-verification

Legacy candidate implementations were removed from the package before release
(one shared network library is now kept instead of several historical stacks).
The full path was re-executed afterwards:

| check | result |
|---|---|
| released checkpoint, complete test split | ACC 0.8748 / F1 0.7139 / AUROC 0.9216 (unchanged) |
| one-epoch retrain on the released cache | 8 min 54 s; train ACC 0.7536 / validation ACC 0.7802; combined 0.7691; best and final checkpoints written |
| A/B against the pre-cleanup one-epoch run | AUROC 0.7933 vs 0.7917, ACC@0.5 0.7847 vs 0.7883 — no regression |
| inference with the freshly trained checkpoint | runs end to end (complete split, ACC@0.5 0.788) |

One property of one-epoch runs: probabilities are under-confident (median
positive 0.29, median background 0.16), so F1 at a fixed 0.5 threshold is low;
the training log records the validation-selected threshold. The distributed
20-epoch checkpoints are unaffected. One-epoch checkpoints are not distributed.

