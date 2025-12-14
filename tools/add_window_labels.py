# tools/add_window_labels.py
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

def add_labels(npz_path: Path, csv_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    if "X" not in d:
        raise ValueError(f"{npz_path} has no 'X'")
    X = d["X"]
    if X.ndim != 3:
        raise ValueError(f"{npz_path} X ndim must be 3, got {X.shape}")
    N, T, F = X.shape

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} missing 'label' column")
    labels = df["label"].astype(int).values

    need = N * T
    if len(labels) < need:
        raise ValueError(f"{csv_path} has only {len(labels)} rows, need {need} for {N}x{T} windows")

    labels = labels[:need].reshape(N, T)
    y = (labels.sum(axis=1) > 0).astype(np.int64)

    # backup then overwrite with y included
    bkp = npz_path.with_suffix(npz_path.suffix + ".bak")
    shutil.copy2(npz_path, bkp)
    np.savez_compressed(npz_path, X=X, y=y)
    print(f"Updated {npz_path} with y shape={y.shape}  (backup: {bkp.name})")

def main():
    root = Path.cwd()
    val_npz = root / "data/processed/val_windows.npz"
    test_npz = root / "data/processed/test_windows.npz"
    val_csv = root / "data/raw/val_mix.csv"
    test_csv = root / "data/raw/test_mix.csv"

    add_labels(val_npz, val_csv)
    add_labels(test_npz, test_csv)

if __name__ == "__main__":
    main()
