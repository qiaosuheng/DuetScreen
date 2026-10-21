# Assisted setup (optional)

If you prefer not to run commands yourself, any AI coding assistant that can
read files and run shell commands can set this up for you. The instructions
below are vendor-neutral — no particular product is required or endorsed.

## What to paste into the assistant

> You are helping me verify a research code release on my machine. Read
> `README.md`, `TROUBLESHOOTING.md` and `run_quickstart.sh` in this folder.
> Then:
> 1. Check that Docker is installed and running, and that GPU containers work
>    (`docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`).
> 2. Make sure the folder layout is `release/DuetScreen/` and `release/assets/`.
>    Put the NT-V3 encoder folder in `assets/NTv3_8M_pre`; the ESM2 cache is built automatically.
> 3. Run `./run_quickstart.sh` (or `.\run_quickstart.ps1` on Windows) and show
>    me the complete output.
> 4. If any step fails, use the messages in `TROUBLESHOOTING.md` to diagnose it,
>    one cause at a time, and tell me exactly which command you are changing.
> Do not modify any file except by creating a scratch copy; do not download
> anything other than the Docker base image and Python packages the build
> itself requests.

## What "success" looks like

The script ends with:

```
PASS: 8 records match within 0.002 (max |d| ...)
ok   smoke == passed
ok   downstream parameters == 2987761
ok   encoder trainable == 2763264
ok   finite loss
ALL CHECKS PASSED
```

This reports agreement with the eight stored example scores and successful
execution of the single-step training check. It does not retrain the published
checkpoints or reproduce all benchmark results. The separate one-epoch check
in `CHECKS.md` produced execution-check outputs that are not included among
the checkpoint you trained.

## If the assistant gets stuck

- Ask it to print the last 50 lines of the failing command and match them
  against `TROUBLESHOOTING.md`.
- The checks are independent: prediction (`step 4`) can be verified even if
  training (`step 5`) is skipped on a CPU-only machine (`NO_GPU=1`).
