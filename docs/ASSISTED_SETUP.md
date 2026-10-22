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
> 2. Make sure the repository root contains `assets/NTv3_8M_pre`, with `data/`,
>    `toy/` and `runs/` as needed. The default layout keeps `assets/` inside the
>    repository; do not create a separate top-level assets directory beside it.
> 3. Run `./run_quickstart.sh` (or the `.ps1` wrapper on Windows) and show me
>    the complete output.
> 4. If any step fails, use the messages in `TROUBLESHOOTING.md` to diagnose it,
>    one cause at a time, and tell me exactly which command you are changing.
> Do not modify any file except by creating a scratch copy; do not download
> anything other than the Docker base image and Python packages the build
> itself requests.

## What "success" looks like

`run_quickstart.sh` is an environment check. Success ends with:

```
READY
```

and, before that, a line confirming the package import inside the container
(`duetscreen import OK`). It does **not** run a prediction, compute metrics or
train a model.

For the other entry points:

- labelled evaluation: `./run_eval.sh` with `MODEL`, `DATA`, `CACHE`;
- toy workflow: `./toy/run_toy.sh` after unpacking the toy archive;
- full training: `./run_full_training.sh` with `DATA=data/mixed`.

See `README.md` for the three entry points and `CHECKS.md` for the verification
scope (historical execution records versus checks performed for the current
documentation revision).

## If the assistant gets stuck

- Ask it to print the last 50 lines of the failing command and match them
  against `TROUBLESHOOTING.md`.
- The entries are independent: environment check, evaluation, toy and full
  training can be diagnosed separately.
