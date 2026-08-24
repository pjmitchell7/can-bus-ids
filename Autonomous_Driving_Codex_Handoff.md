# CAN Bus IDS Repository Repair - Codex Handoff

## Read this first

This handoff is only for repairing the weaknesses found in Paul Mitchell's existing `pjmitchell7/can-bus-ids` repository and then rerunning the same dense-autoencoder project honestly.

It is **not** permission to begin the later CSCI 710 research-project implementation.

Do not add or implement:

- GRU, LSTM, TCN, or Transformer models
- CAN-ID embedding experiments
- interarrival-time features
- the four-configuration research ablation
- the proposal's fixed 1% benign-FPR protocol
- alert-episode, detection-delay, or repeated-run research experiments
- leave-one-attack-out or cross-vehicle experiments
- any other new model or research question

The stopping point is a corrected, reproducible rerun of the repository's current fully connected, benign-only autoencoder. Once that repair is complete, Paul will decide separately when to begin the new research project.

## Repository and source context

- Repository: <https://github.com/pjmitchell7/can-bus-ids>
- Commit reviewed on August 24, 2026: `9956ad12ec61d44aaefdef8e907776b7086120cd`
- Original report: `Autonomous_Driving_Final_PJMitchell.pdf`
- Dataset: OCSLab/HCRL Car Hacking Dataset
- Existing attacks: DoS, Fuzzy, Gear, and RPM

The original report explains what Paul intended to build. The repository audit determines what the code actually does. When those conflict, repair the code and update the documentation rather than assuming the paper's description is already true.

## First prompt for the new Codex conversation

Place this file in the repository root, open Codex from that folder, and send:

> Read `Autonomous_Driving_Codex_Handoff.md` completely. Then inspect Git status, the current repository tree, all Python files, and the available data paths. This task is strictly a repair and clean rerun of the existing dense-autoencoder project. Do not implement anything from my later research-project proposal. First verify each audit finding against the current code, create a safe working branch, and give me a short plan. Then make the repairs in small tested stages with natural code comments and coherent Git commits. Preserve the existing model and 64-frame/hop-32 design. Do not push to GitHub until I explicitly tell you to push.

## What the existing project is supposed to remain

The repaired project must still be the same basic course project:

1. Parse one normal CAN capture and four attack captures.
2. Represent each CAN frame with 10 values:
   - `DATA0` through `DATA7`
   - DLC
   - a stable numeric CAN-ID code
3. Build windows of 64 consecutive frames with a hop of 32.
4. Flatten each `[64, 10]` window to 640 values.
5. Train a fully connected autoencoder using benign traffic only.
6. Use mean squared reconstruction error as the anomaly score.
7. Compare the existing threshold approaches.
8. Report corrected validation and test performance.

Keep the dense autoencoder architecture:

`640 -> 128 -> 64 -> 32 -> 64 -> 128 -> 640`

Keep the existing core settings unless a verified bug requires a change:

- ReLU hidden activations
- linear output layer
- dropout `0.10`
- Adam optimizer
- learning rate `1e-3`
- weight decay `1e-5`
- batch size `256`
- 30 epochs
- MSE loss

The repair should make this baseline valid and reproducible. It should not turn the repository into a new research framework yet.

## Historical results are not final evidence

The report listed:

| Threshold | Split | Precision | Recall | F1 | Flagged windows |
| --- | --- | ---: | ---: | ---: | ---: |
| Train 99th percentile | Validation | 0.998 | 0.386 | 0.557 | 36.28% |
| Train 99th percentile | Test | 0.996 | 0.177 | 0.300 | 16.62% |
| Validation-F1 optimized | Validation | 0.991 | 0.529 | 0.690 | 50.00% |
| Validation-F1 optimized | Test | 0.983 | 0.222 | 0.362 | 21.10% |

Preserve these as historical context only. The corrected labels and splits will change the data and therefore the metrics. Never write new numbers into the README or report until they come from the repaired pipeline and a saved run.

## Verified weaknesses that need repair

### 1. Attack frames are labeled incorrectly

Current problem:

`src/make_splits.py::_read_attack` assigns `label = 1` to every row from an attack file. That is not the dataset's labeling rule. Attack captures contain both injected and normal traffic.

Correct rule:

- `Flag == T` means the frame was injected.
- `Flag == R` means it was a normal frame, even though it appears in an attack capture.

Implement one strict helper and test it:

```python
def flag_to_label(flag: pd.Series) -> pd.Series:
    clean = flag.astype("string").str.strip().str.upper()
    invalid = ~clean.isin(["T", "R"])
    if invalid.any():
        bad = sorted(clean[invalid].dropna().unique().tolist())
        raise ValueError(f"Unexpected Flag values: {bad}")
    return clean.eq("T").astype("int8")
```

Do not infer the binary label from the filename. The filename supplies the attack family; `Flag` supplies the frame label.

### 2. Benign evaluation rows overlap benign training

Current problem:

- `preprocess.py` trains on all of `train_normal.csv`.
- `make_splits.py` also takes normal validation and test rows from that same file.

The model therefore trains on frames later presented as unseen benign evaluation data.

Repair:

- Sort the normal capture once in its original chronological order.
- Divide it into three non-overlapping contiguous ranges: train, validation, and test.
- Create those frame-level ranges before making windows.
- Train only on the train range.
- Use the validation normal range only in validation.
- Use the test normal range only in final test.
- Do not shuffle frames.
- Do not build a window across either split boundary.

Important: the earlier repository-audit notes established the need for disjoint ranges but did not preserve an authoritative numeric train/validation/test ratio. Do not silently import the later proposal's 60/20/20 choice. Make the ratio explicit and configurable, and ask Paul to approve the ratio once before generating the full corrected data. Synthetic tests and the rest of the implementation can be completed first.

### 3. The report claims Z-score scaling, but the final code does not do it

Current problem:

`src/config.py` says `normalize="zscore"`, but `src/preprocess.py` defines its own unrelated `Config` and saves raw payload bytes, DLC, and ID code. Payload values can dominate MSE because they have a much larger range.

Repair:

1. Remove the duplicate preprocessing configuration.
2. Build the CAN-ID map from benign training only.
3. Build the 10-feature frame matrix for benign training.
4. Calculate one mean and standard deviation per feature from benign training only.
5. Replace a zero or near-zero standard deviation with `1.0`.
6. Save the means, standard deviations, feature order, and epsilon.
7. Apply the frozen values to normal validation, normal test, and all attack captures.
8. Copy the actual scaler and ID-map artifacts into each experiment run.

Recommended artifact names:

```text
data/processed/can_id_map.json
data/processed/scaler.npz
data/processed/preprocess_meta.json
```

`preprocess_meta.json` should record the feature order, window length, hop, source split file hashes, and scaler/ID-map filenames.

Do not add timing features or ID embeddings. The repaired feature vector stays at 10 values.

### 4. Windows can cross independent recording boundaries

Current problem:

The split builder concatenates independent normal and attack files, sorts all rows by timestamp, and then the preprocessor windows the resulting table. This can create a 64-frame sequence whose first rows came from one recording and whose later rows came from another.

Repair:

- Keep each source capture separate through frame preprocessing.
- Keep each train/validation/test range separate.
- Build windows inside each capture/range independently.
- After windowing, it is safe to concatenate window arrays for overall evaluation because each individual window is already valid.
- Store capture metadata for every window.

Do not add the later proposal's end-of-capture gap-removal or timing-artifact research here. This repair is only about preventing source capture and split boundaries from being crossed.

### 5. Attack-type metadata disappears

Current problem:

The evaluator has optional per-attack code, but `make_splits.py` never creates a useful attack-type column.

Repair:

For every source row, retain:

```text
capture_id
source_row
attack_family     # normal, dos, fuzzy, gear, rpm
Flag
label
```

For every generated window, retain:

```text
capture_id
attack_family
window_start_row
window_end_row
window_label
```

For a positive window from `gear_dataset.csv`, the attack family is `gear`. An all-`R` window from that same capture still has `window_label = 0`, while its source family remains `gear`. This lets the evaluator calculate Gear recall on positive Gear windows without pretending every Gear-file row is malicious.

### 6. The old label-repair utility is wrong for overlapping windows

Current problem:

`tools/add_window_labels.py` and the fallback in `evaluate.py` take the first `N * T` frame labels and reshape them into non-overlapping blocks. That is incompatible with window length 64 and hop 32.

Repair:

Use one window-start array for both features and labels:

```python
starts = np.arange(0, len(frames) - window_len + 1, hop)

X_windows = np.stack([
    X_frames[start:start + window_len]
    for start in starts
])

y_windows = np.array([
    labels[start:start + window_len].any()
    for start in starts
], dtype=np.int8)
```

The production evaluator should require `y` and metadata from the processed artifact. It should not silently derive labels by a different method.

After the new path works, either remove `tools/add_window_labels.py` or replace it with a short deprecation error that tells users to regenerate processed windows.

### 7. Threshold configuration names do not match

Current problem:

- `config.py` uses `thresh_method`.
- `evaluate.py` reads `thr_method`.

This causes a silent fallback to the training-percentile method.

Repair:

- Pick one name, preferably `threshold_method`.
- Validate allowed values.
- Do not silently ignore an unrecognized method.
- Expose the method through the command line or a saved config.
- Save the selected method, threshold, source split, and percentile in `metrics.json`.

Retain the existing project comparisons:

1. Training-error percentile, using the intended 99th percentile as an explicit setting.
2. Validation-F1 optimization as a labeled diagnostic comparison.
3. The existing capped-F1 option may remain if it is tested and clearly named.

Do not replace this with the later proposal's 1% benign-calibration experiment yet.

### 8. Evaluation is too narrow

Current problem:

The evaluator reports precision, recall, F1, flagged rate, and median errors, but does not expose enough information to understand class imbalance or false positives.

Add to the existing evaluation:

- exact TP, FP, TN, and FN
- number of positive and negative windows
- positive-window prevalence
- accuracy, but never by itself
- specificity
- false-positive rate
- precision, recall, and F1
- flagged-window rate
- average precision/AUPRC
- optionally AUROC, labeled as secondary
- per-attack precision, recall, and F1 where both classes are present
- per-attack positive-window recall even when a family subset contains no standalone negatives

Save the thresholded predictions or at least the raw anomaly scores and aligned metadata so metrics can be regenerated without rerunning the model.

Do not add attack-episode detection, detection delay, or false-alerts-per-hour in this repair. Those belong to later research work.

### 9. CUDA latency timing is unreliable

Current problem:

The code warms up the GPU and times one forward call with `perf_counter`, but CUDA work is asynchronous. It does not synchronize before or after the measurement.

Repair the existing batch-latency probe:

```python
if device.startswith("cuda"):
    torch.cuda.synchronize()

t0 = time.perf_counter()
_ = model(sample)

if device.startswith("cuda"):
    torch.cuda.synchronize()

elapsed_ms = (time.perf_counter() - t0) * 1000
```

Use several measured repetitions after warm-up and report median and p95 instead of one unusually fast call. Preserve the original batch size as a clearly labeled throughput-oriented measurement if desired. Do not call it end-to-end latency, embedded latency, or proof of real-time deployment. This repair does not need the later proposal's separate batch-one study.

### 10. Reproducibility and paths are incomplete

Current problems:

- no dependency file
- no fixed random seed
- hard-coded local Windows path in `tools/convert_txt_to_csv.py`
- tracked `data` and backup symlinks point to a W&M scratch directory
- stale artifact names in `train.py`
- no automated tests
- README file tree and claims do not match the repository

Repair:

- add a small `requirements.txt` or `pyproject.toml`
- add a seed to config and seed Python, NumPy, and PyTorch
- make the converter accept command-line input and output paths
- replace broken tracked symlinks with a normal ignored local data layout plus `data/README.md`
- save effective config, seed, environment versions, Git commit, and data hashes with a run
- copy `can_id_map.json` and `scaler.npz`, not obsolete `id_vocab.json` and `scaler_std.npz`
- add focused pytest tests
- update README after the repaired run

Do not create a large framework or unnecessary abstractions. This repository is small; the repair should remain understandable.

## Natural code style requirements

Paul wants the changes to read like careful, ordinary work in the existing repository, not like a giant generated rewrite.

Follow these rules:

- Make surgical changes to the existing project before proposing a large restructure.
- Prefer plain functions and small modules over elaborate class hierarchies.
- Use names a student developer would naturally choose: `attack_family`, `window_starts`, `train_mean`, `train_std`.
- Comment only when the reason is not obvious from the code.
- Explain dangerous or non-obvious choices, such as why splitting happens before windowing.
- Do not comment every assignment or restate the next line in English.
- Keep docstrings short and factual.
- Avoid canned headings such as `# Robust Enterprise Data Pipeline`.
- Do not add emojis, marketing language, or claims like "production ready."
- Preserve useful existing code instead of replacing everything just to standardize style.
- When touching an overly long existing comment, simplify it only if doing so improves clarity.
- Use consistent formatting, but do not perform unrelated mass formatting in the same commit as a behavioral fix.
- Do not add comments about Codex, AI generation, or the conversation.

Good comments:

```python
# Split the normal capture before windowing so overlapping windows cannot leak
# frames across train, validation, and test.
```

```python
# Build each capture separately. Concatenating the completed windows is safe;
# concatenating raw captures is not.
```

Unhelpful comments:

```python
# Convert the labels to an array.
labels = labels.to_numpy()
```

```python
# This highly robust and scalable function elegantly processes the dataset.
```

The goal is readable code that matches the size and history of the project. Do not deliberately introduce mistakes, uneven formatting, slang, or fake personal anecdotes to make it "look human."

## Natural Git history requirements

Create commits only after real, coherent work. Do not fabricate dates, backdate commits, change the configured author, create empty commits, or pretend manual work happened when it did not.

Before editing:

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

If there are unrelated user changes, preserve them. Create a branch such as:

```bash
git switch -c fix/baseline-pipeline
```

Use Paul's already configured Git identity. Do not edit global Git name or email.

Make a small number of coherent commits. A natural sequence would be:

1. `fix attack labels and keep source metadata`
2. `separate normal training and evaluation data`
3. `apply train-only scaling in preprocessing`
4. `make threshold selection explicit`
5. `add per-attack metrics and reliable latency timing`
6. `add pipeline tests and update documentation`

These are examples, not mandatory text. Match commits to the work actually completed. If two fixes are inseparable and tested together, one clear commit is better than artificial fragmentation.

Before each commit:

```bash
git diff --check
pytest -q
git status --short
git diff --stat
```

Inspect the staged diff before committing. Do not mix generated experiment files, raw data, virtual environments, or unrelated formatting into source commits.

Commit-message style:

- short, direct subject line
- imperative or plain lowercase style consistent across the branch
- optional body only when the reason is not clear from the diff
- no essay-sized autogenerated commit message
- no fake ticket numbers
- no mention of AI authorship in the subject

Do not push automatically. Once the local branch is complete, tested, and summarized, ask Paul before:

```bash
git push -u origin fix/baseline-pipeline
```

If Paul explicitly tells Codex in the new conversation to push as part of the task, verify the branch and remote immediately before pushing.

## Minimal target structure

Avoid an unnecessary full rewrite. A reasonable repaired tree is:

```text
can-bus-ids/
  README.md
  requirements.txt                 # or pyproject.toml
  data/
    README.md
    raw/                            # ignored
    processed/                      # ignored
  experiments/                     # run outputs ignored
  src/
    __init__.py
    config.py
    make_splits.py
    preprocess.py
    model_autoencoder.py
    train.py
    evaluate.py
    metrics.py                      # add only if evaluate.py becomes unwieldy
    utils.py
  tools/
    convert_txt_to_csv.py
    add_window_labels.py            # deprecated or removed after verification
  tests/
    test_make_splits.py
    test_preprocess.py
    test_window_labels.py
    test_thresholds.py
    test_metrics.py
```

Adding one or two focused helper modules is fine. Do not turn the repair into a packaging migration unless the current layout genuinely blocks testing.

## Low-level implementation order

### Phase 0 - Inspect and protect the current work

1. Read every tracked source file and README.
2. Record the current commit and Git status.
3. Check whether `data` and `data.bak_20251105_201334` are symbolic links.
4. Locate the actual raw files without moving or deleting them.
5. Record file names, sizes, and SHA-256 hashes.
6. Create the repair branch.
7. Create a short `REPAIR_STATUS.md` containing the audit checklist and current phase.

Do not delete or overwrite old experiment directories. New corrected runs should have distinct names such as `run_corrected_<timestamp>` or carry a `pipeline_version` field.

### Phase 1 - Add tests that reproduce the bugs

Create tiny synthetic CSV fixtures. Tests should run without the full dataset or a GPU.

Write these tests before changing behavior:

#### Flag semantics

- `T` becomes 1.
- `R` becomes 0.
- all rows in an attack file are not automatically 1.
- invalid flags fail clearly.

#### Split isolation

- normal train, validation, and test frame indices are disjoint.
- order is preserved.
- the split occurs before windows.

#### Window alignment

With `window_len=4`, `hop=2`, and eight frames, starts must be `[0, 2, 4]`.

If the only positive frame is index 3:

- window 0 covers `[0,1,2,3]` and is positive
- window 1 covers `[2,3,4,5]` and is positive
- window 2 covers `[4,5,6,7]` and is negative

#### Capture boundary

Create two three-frame captures with a four-frame window. The code must produce zero windows, not one window made by joining the captures.

#### Training-only scaling

- fit a scaler on a small training matrix
- include an extreme value only in validation
- prove that the saved training mean/std do not change
- prove constant features do not divide by zero

#### Stable ID mapping

- train IDs produce deterministic codes
- an evaluation-only ID maps to the reserved unknown value
- evaluation does not expand or reorder the map

#### Threshold configuration

- each allowed method calls the intended selector
- a misspelled method raises an error
- saved metadata names the actual method

#### Metrics

- confusion counts match a hand-computed example
- false-positive rate and specificity are correct
- per-attack recall filters the intended positive windows

These tests become the evidence that the repair addresses the audit rather than just rearranging code.

### Phase 2 - Repair split construction and metadata

Modify `src/make_splits.py` first.

Expected behavior:

1. Load normal capture with `attack_family="normal"`, `capture_id="normal"`, `label=0`, and an original `source_row`.
2. Load each attack capture with its own `capture_id` and family.
3. Derive attack labels from `Flag`.
4. Keep each capture independent.
5. Split each attack capture contiguously 50/50 into validation and test, preserving the old project design.
6. Split the normal capture contiguously into train, validation, and test using the explicit ratio Paul approves.
7. Write separate frame files or a manifest that preserves capture boundaries.

Suggested outputs:

```text
data/interim/normal_train.csv
data/interim/normal_val.csv
data/interim/normal_test.csv
data/interim/dos_val.csv
data/interim/dos_test.csv
data/interim/fuzzy_val.csv
data/interim/fuzzy_test.csv
data/interim/gear_val.csv
data/interim/gear_test.csv
data/interim/rpm_val.csv
data/interim/rpm_test.csv
data/interim/split_manifest.json
```

The exact format can change if memory or storage makes a manifest cleaner. The required property is that the preprocessor knows every capture boundary.

Do not create one `val_mix.csv` or `test_mix.csv` by interleaving raw frames from different files. The evaluator can combine completed windows later.

Manifest fields:

```text
split
capture_id
attack_family
source_path
source_sha256
first_source_row
last_source_row
frame_count
injected_frame_count
```

### Phase 3 - Repair preprocessing

Use the central config. Remove or stop using the private `Config` inside `preprocess.py`.

Processing order:

1. Read `normal_train` frames.
2. Normalize CAN-ID strings consistently.
3. Build the sorted ID map from `normal_train` only.
4. Convert bytes, DLC, and ID code into a `[frames, 10]` float matrix.
5. Fit feature-wise Z-score statistics on that matrix.
6. Save ID map and scaler.
7. Transform each normal or attack capture/range separately.
8. Build 64-frame windows with hop 32 inside that one range.
9. Build labels using the same starts.
10. Save window metadata.
11. Concatenate completed validation windows and completed test windows only after each source has been windowed.

Recommended processed arrays:

```text
train_windows.npz
  X: float32 [N, 64, 10]

val_windows.npz
  X: float32 [N, 64, 10]
  y: int8    [N]
  attack_family: short string or integer code [N]
  capture_id: short string or integer code [N]
  start_row: int64 [N]
  end_row: int64 [N]

test_windows.npz
  same fields as validation
```

If NumPy string arrays are inconvenient, save integer metadata codes and a JSON mapping. Avoid pickle-dependent object arrays when possible.

The window count for one capture/range is:

```python
0 if n_frames < window_len else 1 + (n_frames - window_len) // hop
```

Unknown CAN IDs should use the same reserved scalar code everywhere. Record that code in the ID-map metadata.

### Phase 4 - Keep training behavior stable but reproducible

Do not change the model architecture.

Modify training only as needed to:

- load the corrected scaled windows
- use a configured seed
- seed Python, NumPy, PyTorch, and CUDA
- save the actual scaler and ID map
- save a resolved config
- record parameter count
- record device and dependency versions
- use a corrected run-directory name

Recommended seed helper:

```python
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Do not add early stopping, a new optimizer, a new bottleneck, or a sequence model during this repair unless Paul separately approves a change after the baseline rerun.

### Phase 5 - Repair threshold selection and evaluation

Refactor only enough to make the current evaluator testable.

Recommended separation:

```text
load model and arrays
compute reconstruction scores
select an explicitly configured threshold
calculate overall metrics
calculate per-attack metrics
measure synchronized latency
save scores and metrics
```

Run the two report-compatible threshold modes separately:

#### Training percentile

- calculate reconstruction errors on corrected benign training windows
- use configured percentile, explicitly `99.0` for the report-compatible run
- evaluate unchanged on validation and test

#### Validation-F1 diagnostic

- search thresholds on the corrected labeled validation windows
- freeze the selected threshold
- apply it unchanged to test
- label it diagnostic because it uses attack labels

Save a separate result block for each method. Do not allow the second method to overwrite the first method's files without a distinct name.

Overall metrics should use the combined completed windows, not a synthetic raw-frame stream. Per-family metrics should use saved family metadata.

For a per-family subset containing only positive windows, precision and FPR are not defined within that subset. Report positive-window recall and count rather than silently dropping the family. If normal windows from the same attack capture are included, report the exact subset definition.

### Phase 6 - Repair latency measurement

Keep the measurement close to the original purpose but make it trustworthy:

1. Put model in eval mode.
2. Prepare the sample before timing.
3. Warm up several times.
4. Run at least 100 measured repetitions if practical.
5. Synchronize CUDA around every measured forward pass.
6. Save median, p95, batch size, input shape, device, and dtype.

Do not include data loading or preprocessing unless the metric is explicitly named end-to-end. Do not convert this phase into a new latency study.

### Phase 7 - Full corrected rerun and documentation

Before the full run:

- all tests pass
- Paul has approved the normal split ratio
- raw file hashes are recorded
- processed split counts are reviewed
- positive/negative window prevalence is printed for validation and test
- no source row overlaps train, validation, and test

Then run:

1. corrected preprocessing
2. one corrected dense-autoencoder training run with the fixed seed
3. training-percentile evaluation
4. validation-F1 diagnostic evaluation
5. synchronized latency benchmark

Save all artifacts. Review the results for obvious contradictions before updating documentation.

README update should state:

- exact data placement and command sequence
- actual feature representation
- actual Z-score behavior
- actual 64/32 windowing
- `T/R` label semantics
- disjoint normal splits
- capture-safe windowing
- threshold methods
- corrected metrics with a clear run identifier
- limitations

Do not claim that 64 frames always equal 100 ms. Do not claim embedded deployment. Do not claim the old results were valid after the corrected numbers exist.

## File-by-file repair map

### `src/config.py`

- remove stale unused preprocessing options or make them real
- add one split-ratio configuration without silently borrowing the proposal value
- add window length and hop used by preprocessing
- add `normalize="zscore"` as an implemented option
- use one `threshold_method` name
- add seed
- add evaluation batch size and latency repetitions
- validate values

### `src/make_splits.py`

- derive binary labels from `Flag`
- keep `capture_id`, family, source row, and original order
- create disjoint normal ranges
- split each attack capture separately
- stop timestamp-sorting unrelated files into one stream
- write split metadata

### `src/preprocess.py`

- use central config
- fit ID map and scaler on benign training only
- save and reload both artifacts
- process each capture/range independently
- derive labels with the exact window starts
- save family/capture/start-row metadata
- do not add timing or embedding features

### `src/model_autoencoder.py`

- keep architecture unchanged
- add only small type/shape checks or parameter-count helper if useful
- do not replace it with another model

### `src/train.py`

- seed the run
- save actual preprocessing artifacts
- save environment and run metadata
- use corrected windows
- keep original optimizer, architecture, and main training settings

### `src/evaluate.py`

- remove incorrect CSV reshape fallback
- validate threshold method
- add confusion counts, FPR, AUPRC, prevalence, and per-family output
- save raw scores/predictions or aligned score artifacts
- synchronize CUDA timing and use repeated measurements
- preserve both old threshold comparisons explicitly

### `tools/add_window_labels.py`

- deprecate or remove after corrected preprocessing is verified
- never use it on 64/32 overlapping windows

### `tools/convert_txt_to_csv.py`

- replace hard-coded paths with `argparse`
- retain existing parser behavior where correct
- report malformed/skipped rows
- add parser tests

### `.gitignore` and `data` links

- verify tracked entries are symlinks before changing them
- remove only the links from Git, not their target contents
- keep raw/processed data ignored
- allow a small `data/README.md` explaining expected files
- ignore run outputs, checkpoints, score arrays, caches, and local environments

### `ReadME.md`

- update after code and corrected run exist
- use the real file tree and commands
- remove claims not supported by artifacts
- distinguish historical and corrected results

## Required checks before calling the repair complete

### Data integrity

- only `T` rows are frame-level attacks
- normal train/validation/test source rows are disjoint
- attack captures remain separate until after windowing
- no window crosses a source or split boundary
- validation/test IDs do not alter the training-derived map
- validation/test values do not alter training-derived scaler statistics
- window labels use exact starts `[0, 32, 64, ...]`

### Model integrity

- architecture remains `640-128-64-32-64-128-640`
- training data is benign only
- input features are the corrected scaled 10-feature representation
- seed and effective config are saved

### Evaluation integrity

- threshold method in config matches actual code path
- training-percentile and validation-F1 results are stored separately
- TP/FP/TN/FN reproduce all derived metrics
- prevalence and FPR are visible
- every per-attack result has an exact window count
- test threshold is never reselected on test
- CUDA timing is synchronized and repeated

### Repository integrity

- tests pass from the repository root
- dependencies are documented
- no raw data or large generated artifacts are staged
- Git diff contains no unrelated rewrite
- commits correspond to real phases and pass tests
- nothing has been pushed without Paul's instruction

## Suggested command flow

Codex may adjust exact flags to the repaired CLI, but the workflow should remain this simple:

```bash
python -m pytest -q
python -m src.make_splits --config config.json
python -m src.preprocess --config config.json
python -m src.train --config config.json
python -m src.evaluate --run experiments/run_corrected_<timestamp> --threshold-method train_percentile
python -m src.evaluate --run experiments/run_corrected_<timestamp> --threshold-method val_f1
```

Every command should have a small debug mode such as `--max-rows` or `--max-windows`. Debug artifacts must be labeled and excluded from final metrics.

## Stop condition

Stop this body of work when all of the following are true:

1. The existing dense-autoencoder pipeline uses correct `T/R` labels.
2. Benign train, validation, and test data are disjoint.
3. Windows respect capture and split boundaries.
4. Z-score statistics and the ID map come only from benign training.
5. Attack-family metadata survives into evaluation.
6. Threshold selection is explicit and testable.
7. Evaluation includes FPR, prevalence, AUPRC, confusion counts, and per-attack results.
8. GPU timing is synchronized and repeated.
9. Tests, dependencies, paths, run artifacts, README, and Git history are clean.
10. A corrected run of the same dense baseline has been completed and clearly separated from the old numbers.

At that point, summarize what changed and wait. Do not begin any later proposal work unless Paul starts a separate task authorizing it.
