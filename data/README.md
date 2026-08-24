# Local data layout

The dataset is not stored in Git. Place the converted files under `data/raw/`:

- `train_normal.csv`
- `DoS_dataset.csv`
- `Fuzzy_dataset.csv`
- `gear_dataset.csv`
- `RPM_dataset.csv`

Run `python -m src.make_splits --config config.json` to create frame-level ranges under `data/interim/`, then run preprocessing. The raw files must remain local and should not be committed.
