import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.make_splits import build_splits
from src.preprocess import process


def _normal_frame(n):
    data = {
        "Timestamp": np.arange(n, dtype=float),
        "CAN_ID": ["100"] * n,
        "DLC": [8] * n,
    }
    for index in range(8):
        data[f"DATA{index}"] = ["00"] * n
    return pd.DataFrame(data)


def _attack_frame(n, family):
    data = _normal_frame(n)
    data["CAN_ID"] = [family] * n
    data["Flag"] = ["R"] * n
    data.loc[n - 1, "Flag"] = "T"
    return data


def test_splits_and_preprocessing_keep_sources_separate(tmp_path: Path):
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    raw.mkdir()
    _normal_frame(24).to_csv(raw / "train_normal.csv", index=False)
    filenames = {
        "dos": "DoS_dataset.csv",
        "fuzzy": "Fuzzy_dataset.csv",
        "gear": "gear_dataset.csv",
        "rpm": "RPM_dataset.csv",
    }
    for family, filename in filenames.items():
        _attack_frame(8, family).to_csv(raw / filename, index=False, header=False)

    cfg = Config(
        raw_dir=str(raw),
        interim_dir=str(interim),
        processed_dir=str(processed),
        normal_split_ratio=(0.5, 0.25, 0.25),
        window_len=4,
        hop=2,
        device="cpu",
    )
    build_splits(cfg)
    process(cfg)

    with open(interim / "split_manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert len(manifest["entries"]) == 11
    train_rows = [entry for entry in manifest["entries"] if entry["split"] == "train"]
    assert train_rows[0]["attack_family"] == "normal"

    with np.load(processed / "train_windows.npz", allow_pickle=False) as train:
        assert train["X"].shape[1:] == (4, 10)
    with np.load(processed / "val_windows.npz", allow_pickle=False) as val:
        assert set(val["attack_family"].tolist()) == {"normal", "dos", "fuzzy", "gear", "rpm"}
        assert np.all(val["window_end_row"] - val["window_start_row"] == 3)
    with np.load(processed / "test_windows.npz", allow_pickle=False) as test:
        assert test["y"].sum() > 0

    with open(processed / "can_id_map.json", encoding="utf-8") as handle:
        id_map = json.load(handle)
    assert "dos" not in id_map["mapping"]
