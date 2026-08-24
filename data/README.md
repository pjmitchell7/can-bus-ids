# Local data layout

The dataset is not stored in Git. Place the converted files under `data/raw/`:

- `train_normal.csv`
- `DoS_dataset.csv`
- `Fuzzy_dataset.csv`
- `gear_dataset.csv`
- `RPM_dataset.csv`

Run `python -m src.make_splits --config config.json` to create frame-level ranges under `data/interim/`, then run preprocessing. The raw files must remain local and should not be committed.

`dataset_verification.json` is a non-sensitive run record for the official captures. It records file sizes, row counts, SHA-256 hashes, raw row widths, timestamps, and validated T/R flag counts.
