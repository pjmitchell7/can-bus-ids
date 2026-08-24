# Baseline repair status

## Current phase

Phase 6: official raw data verified, chronological split completed, and bounded preprocessing validation passed. HPC execution remains pending account access and is not claimed here.

## Starting point

- Audited commit: `9956ad12ec61d44aaefdef8e907776b7086120cd`
- Repair branch: `fix/baseline-pipeline`
- Starting working tree: clean
- Raw dataset files: official HCRL captures are present under `data/raw/` and are ignored by Git
- Dataset evidence: `data/dataset_verification.json` records names, sizes, row counts, SHA-256 hashes, widths, timestamps, and flags
- Tracked `data` entries: stale symlink targets to `/scratch/pjmitchell/CAN_Bus_Security/data` were not used

## Audit checklist

- [x] Attack rows are labeled `1` by filename instead of by `Flag`.
- [x] Normal validation and test rows overlap the training capture.
- [x] The central z-score setting is not implemented by preprocessing.
- [x] Raw captures are concatenated and timestamp-sorted before windowing.
- [x] Attack-family metadata is not preserved.
- [x] The CSV label fallback assumes non-overlapping windows.
- [x] Threshold configuration names disagree between config and evaluator.
- [x] Evaluation omits confusion counts, prevalence, FPR, AUPRC, and useful family detail.
- [x] CUDA timing is not synchronized or repeated.
- [x] Reproducibility, paths, dependencies, tests, and README structure are incomplete.

## Verification

- 13 parser and synthetic regression tests pass.
- The full raw data produces 11 independent chronological ranges using normal 60/20/20 and attack 50/50 splits.
- A bounded 2,000-row-per-range preprocessing run produced 64-frame, hop-32 capture-safe windows and train-only preprocessing artifacts.
- No corrected full-data metrics exist yet.

## Remaining work

- [x] Approve and set the normal train/validation/test ratio to 60/20/20.
- [x] Place and verify the five official raw dataset files under `data/raw/`.
- [x] Build chronological split files and manifest without shuffling or cross-boundary ranges.
- [x] Validate parser behavior locally, including DLC-dependent attack rows and T/R semantics.
- [ ] Run full corrected preprocessing, dense-autoencoder training, both threshold modes, and the repeated latency probe.
- [ ] Run the prepared HPC workflow after account access and data transfer are available.

## Pending approval

The local `config.json` uses the user-approved 60/20/20 normal split, 50/50 attack validation/test split, 64-frame windows, hop 32, and CPU. The full training/evaluation run is intentionally not reported because the local Torch package is incomplete and HPC access is not yet available.
