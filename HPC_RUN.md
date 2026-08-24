# HPC run preparation

This is a prepared workflow only. It has not been executed, and this repository records no HPC results.

## Connect and allocate

After W&M account access is available:

```bash
ssh -J pjmitchell@bastion.wm.edu pjmitchell@gulf.sciclone.wm.edu
cd ~/can-bus-ids
salloc -N1 -n8 -t 1-0 --gpus=1
hostname
nvidia-smi
```

The repository and the five verified raw files must be present on the allocated node or its shared filesystem. Transfer the repository without `data/raw/`, local `config.json`, archives, or generated outputs; transfer the verified raw files separately into `data/raw/`.

## Run sequence

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python tools/validate_raw_dataset.py --output data/dataset_verification.json
python -m src.make_splits --config config.json
python -m src.preprocess --config config.json
python -m src.train --config config.json
python -m src.evaluate --run experiments/run_corrected_<timestamp> --threshold-method train_percentile
python -m src.evaluate --run experiments/run_corrected_<timestamp> --threshold-method val_f1
```

Copy `config.hpc.example.json` to the local `config.json` on the cluster. It sets `device` to `cuda` and preserves the approved split, feature order, 64-frame window, hop 32, dense-autoencoder architecture, and training settings. Keep the generated run directory and metrics as the evidence bundle. Do not commit raw data, archives, credentials, `config.json`, or large generated artifacts.

## Required checks before reporting results

- `nvidia-smi` shows the allocated GPU and the training log records the device.
- The dataset verification report passes and its SHA-256 values match the transferred files.
- The split manifest is chronological and disjoint; no frames are shuffled.
- Preprocessing metadata shows a train-only ID map and scaler, with 64-frame/hop-32 windows.
- Both threshold methods have saved validation/test scores and metrics.
- Latency is reported only from the synchronized repeated benchmark; it is not an end-to-end deployment claim.
