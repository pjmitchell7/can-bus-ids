# CAN Bus intrusion detection baseline

This repository contains a corrected, reproducible rerun of the existing benign-only dense autoencoder baseline for the OCSLab/HCRL Car Hacking Dataset. It is intentionally not the later sequence-model or ablation research project.

## Data placement

The dataset is not committed. Put these files in `data/raw/`:

- `train_normal.csv` with a header
- `DoS_dataset.csv`
- `Fuzzy_dataset.csv`
- `gear_dataset.csv`
- `RPM_dataset.csv`

The attack captures are headerless HCRL CSV files with Timestamp, CAN ID, DLC, up to eight payload bytes, and a Flag column. The converter in tools/convert_txt_to_csv.py accepts text-log input and explicit --input/--output paths.

## Pipeline

Each frame has ten model features: DATA0 through DATA7, DLC, and a stable numeric CAN-ID code. Flag == T means injected and Flag == R means normal, including normal frames that occur in an attack capture.

The normal capture is kept in chronological order and split into three contiguous, non-overlapping ranges before windowing. The split ratio is deliberately not chosen in code; set normal_split_ratio in a copied config.json after approving the policy. Each attack capture is split contiguously into validation and test halves and remains separate until after windowing.

Preprocessing fits the sorted CAN-ID map and feature-wise z-score statistics on benign training frames only. Constant or near-constant features use a standard deviation of 1.0. The frozen artifacts are saved as can_id_map.json, scaler.npz, and preprocess_meta.json.

Windows contain 64 consecutive frames with hop 32. A window is positive if any frame in that same window has label 1. Window metadata records capture ID, attack family, source start row, and source end row. No window crosses a capture or split boundary.

The model remains the original fully connected autoencoder:

`640 -> 128 -> 64 -> 32 -> 64 -> 128 -> 640`

It uses ReLU hidden layers, a linear output layer, dropout 0.10, Adam, learning rate 1e-3, weight decay 1e-5, batch size 256, 30 epochs, and MSE loss.

## Reproduction commands

Install dependencies:

```
python -m pip install -r requirements.txt
```

Copy config.example.json to config.json, then set the approved normal split ratio. Run:

```
python -m pytest -q
python -m src.make_splits --config config.json
python -m src.preprocess --config config.json
python -m src.train --config config.json
python -m src.evaluate --run experiments/run_corrected_<timestamp> --threshold-method train_percentile
python -m src.evaluate --run experiments/run_corrected_<timestamp> --threshold-method val_f1
```

Use --max-rows for debug runs. Debug preprocessing is written below a debug_maxrows_<n> directory and is marked in metadata; it must not be used as final evidence.

## Evaluation

The two retained comparisons are:

1. train_percentile, using the configured 99th percentile of benign training reconstruction errors.
2. val_f1, a labeled diagnostic that selects on validation and freezes the threshold for test.

The optional f1_capped method remains available as a clearly named diagnostic. Each method writes its own metrics_<method>.json and aligned scores_<method>_val.npz/scores_<method>_test.npz.

Metrics include TP, FP, TN, FN, positive and negative counts, prevalence, accuracy, specificity, false-positive rate, precision, recall, F1, flagged-window rate, AUPRC, optional AUROC, and per-family results. A family subset with no standalone negatives reports positive-window recall and does not invent precision or false-positive metrics.

Latency is a repeated forward-pass measurement with CUDA synchronization around each timed call. It reports median, p95, batch size, input shape, device, and dtype. It is not an end-to-end or embedded-deployment latency claim.

## Current evidence and limitations

Historical numbers from the prior report are context only and are not reproduced here. No corrected full-data numbers should be added until a saved run has been completed with the real raw files and the approved normal split ratio.

The baseline uses static overlapping windows and a single vehicle capture. It does not add timing features, ID embeddings, sequence models, alert episodes, detection delay, cross-vehicle evaluation, or other later research experiments.
