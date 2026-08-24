# Baseline repair status

## Current phase

Phase 5: code repair and synthetic verification complete; full-data rerun is pending dataset placement and split-ratio approval.

## Starting point

- Audited commit: `9956ad12ec61d44aaefdef8e907776b7086120cd`
- Repair branch: `fix/baseline-pipeline`
- Starting working tree: clean
- Raw dataset files: not available in the local workspace or Downloads search
- Tracked `data` entries: stale symlink targets to `/scratch/pjmitchell/CAN_Bus_Security/data`

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

- 12 synthetic regression tests pass.
- The synthetic pipeline produces independent ranges, train-only preprocessing artifacts, and capture-safe windows.
- No corrected full-data metrics exist yet.

## Remaining work

- [ ] Approve and set the normal train/validation/test ratio.
- [ ] Place the five raw dataset files under data/raw/.
- [ ] Run corrected preprocessing, dense-autoencoder training, both threshold modes, and the repeated latency probe.
- [ ] Review saved metrics and commit the remaining documentation/run evidence.

## Pending approval

The normal capture split ratio must be explicit, but the handoff does not prescribe its value. Synthetic tests and code repairs can proceed before choosing it; full corrected preprocessing and reruns cannot.
