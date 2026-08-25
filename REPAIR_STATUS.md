# Project status

## Current state

The corrected dense-autoencoder baseline is complete and is maintained on `fix/baseline-pipeline` at commit `df99b495cba3721910e344e2efd92bfb9854fd21`.

- The official HCRL captures were verified under `data/raw/`.
- The normal capture uses a chronological 60/20/20 train/validation/test split.
- Each attack capture uses a chronological 50/50 validation/test split.
- Windows contain 64 frames with hop 32 and never cross capture or split boundaries.
- The full run completed on a GPU with 30 training epochs, both threshold methods, and synchronized latency measurements.

## Verification

- The parser and regression suite passes all 13 tests.
- The raw-data verification report records file names, sizes, row counts, SHA-256 hashes, row widths, timestamps, and T/R flag counts.
- Preprocessing fits the CAN-ID map and z-score statistics on benign training frames only.
- The run metadata records seed 42, ten input features, the preserved dense-autoencoder architecture, and the CUDA device.

## Data and outputs

Raw captures, local configuration, model checkpoints, score arrays, and generated metrics are intentionally excluded from Git. Keep the run directory as a separate evidence bundle when reproducing the experiment.

## Scope

This project is the corrected benign-only dense-autoencoder baseline. It does not include timing features, ID embeddings, sequence models, alert episodes, detection delay, cross-vehicle evaluation, or later research experiments.
