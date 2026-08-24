"""Create disjoint frame ranges while preserving capture boundaries and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, resolve_project_path
from .utils import sha256_file


RAW_COLUMNS = [
    "Timestamp", "CAN_ID", "DLC",
    "DATA0", "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7",
    "Flag",
]
BASE_COLUMNS = [*RAW_COLUMNS, "capture_id", "source_row", "attack_family", "label"]


def flag_to_label(flag: pd.Series) -> pd.Series:
    """Map injected (T) and replayed/normal (R) frame flags to binary labels."""
    clean = flag.astype("string").str.strip().str.upper()
    invalid = ~clean.isin(["T", "R"])
    if clean.isna().any() or invalid.any():
        bad = sorted(clean[invalid].dropna().unique().tolist())
        if clean.isna().any():
            bad.append("<missing>")
        raise ValueError(f"Unexpected Flag values: {bad}")
    return clean.eq("T").astype("int8")


def _normalize_frame(df: pd.DataFrame, capture_id: str, attack_family: str) -> pd.DataFrame:
    frame = df.copy()
    for column in RAW_COLUMNS:
        if column not in frame.columns:
            if column.startswith("DATA"):
                frame[column] = "00"
            else:
                raise ValueError(f"Missing required column {column!r}")
    frame["Flag"] = frame["Flag"].astype("string").str.strip().str.upper()
    frame["label"] = flag_to_label(frame["Flag"])
    frame["Timestamp"] = pd.to_numeric(frame["Timestamp"], errors="coerce")
    frame["CAN_ID"] = (
        frame["CAN_ID"].astype("string").fillna("").str.strip().str.lower().str.replace("0x", "", regex=False)
    )
    frame["DLC"] = pd.to_numeric(frame["DLC"], errors="coerce").fillna(0).clip(0, 8).astype("int16")
    for column in [f"DATA{i}" for i in range(8)]:
        frame[column] = frame[column].fillna("00").astype("string").str.strip()
    frame["capture_id"] = capture_id
    frame["attack_family"] = attack_family
    if "source_row" not in frame.columns:
        frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    return frame[BASE_COLUMNS]


def _read_attack(path: str | Path, capture_id: str, attack_family: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, names=list(range(12)), dtype="string")
    if len(raw) and str(raw.iloc[0, 0]).strip().lower() == "timestamp":
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"Attack capture is empty: {path}")

    values = raw.to_numpy(dtype=object)
    dlc = pd.to_numeric(raw.iloc[:, 2], errors="coerce").fillna(0).clip(0, 8).astype(int).to_numpy()
    row_index = np.arange(len(raw))
    flag_position = 3 + dlc
    if int(flag_position.max()) >= values.shape[1]:
        raise ValueError(f"Attack capture has rows shorter than DLC requires: {path}")
    frame = pd.DataFrame({
        "Timestamp": raw.iloc[:, 0],
        "CAN_ID": raw.iloc[:, 1],
        "DLC": raw.iloc[:, 2],
        "Flag": values[row_index, flag_position],
    })
    for index in range(8):
        column = np.full(len(raw), pd.NA, dtype=object)
        present = dlc > index
        if present.any():
            if 3 + index >= values.shape[1]:
                raise ValueError(f"Attack capture has a row shorter than DLC requires: {path}")
            column[present] = values[present, 3 + index]
        frame[f"DATA{index}"] = column
    return _normalize_frame(frame, capture_id, attack_family)


def _read_normal(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype="string")
    if "Flag" not in raw.columns:
        raw["Flag"] = "R"
    frame = _normalize_frame(raw, "normal", "normal")
    if bool(frame["label"].any()):
        raise ValueError("The normal capture contains injected (T) frames")
    frame["label"] = np.int8(0)
    return frame


def split_contiguous(df: pd.DataFrame, ratios: tuple[float, float, float]) -> dict[str, pd.DataFrame]:
    """Split one already ordered capture into contiguous train/val/test ranges."""
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must contain three positive values summing to 1")
    n = len(df)
    train_end = int(np.floor(n * ratios[0]))
    val_end = train_end + int(np.floor(n * ratios[1]))
    if n >= 3 and (train_end == 0 or val_end == train_end or val_end == n):
        raise ValueError("normal split ratio creates an empty frame range")
    return {
        "train": df.iloc[:train_end].copy(),
        "val": df.iloc[train_end:val_end].copy(),
        "test": df.iloc[val_end:].copy(),
    }


def _source_entry(
    split: str,
    frame_path: Path,
    source_path: Path,
    frame: pd.DataFrame,
    source_row_count: int,
) -> dict:
    root = resolve_project_path(".")
    try:
        source_value = str(source_path.relative_to(root))
    except ValueError:
        source_value = str(source_path)
    try:
        frame_value = str(frame_path.relative_to(root))
    except ValueError:
        frame_value = str(frame_path)
    return {
        "split": split,
        "capture_id": str(frame["capture_id"].iloc[0]),
        "attack_family": str(frame["attack_family"].iloc[0]),
        "source_path": source_value,
        "source_sha256": sha256_file(source_path),
        "source_size_bytes": int(source_path.stat().st_size),
        "source_row_count": int(source_row_count),
        "frame_path": frame_value,
        "first_source_row": int(frame["source_row"].iloc[0]),
        "last_source_row": int(frame["source_row"].iloc[-1]),
        "frame_count": int(len(frame)),
        "injected_frame_count": int(frame["label"].sum()),
    }


def build_splits(cfg: Config, max_rows: int | None = None) -> Path:
    """Write independent frame ranges and return the manifest path."""
    cfg.validate(require_normal_split=True)
    root = resolve_project_path(".")
    raw_dir = resolve_project_path(cfg.raw_dir)
    interim_dir = resolve_project_path(cfg.interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    normal_path = raw_dir / cfg.normal_file
    normal = _read_normal(normal_path)
    normal_source_row_count = len(normal)
    attack_sources = [(family, raw_dir / filename) for family, filename in cfg.attack_files]
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        normal = normal.iloc[:max_rows].copy()

    normal = normal.sort_values("source_row", kind="mergesort").reset_index(drop=True)
    normal_ranges = split_contiguous(normal, cfg.normal_split_ratio)
    entries: list[dict] = []

    for split, frame in normal_ranges.items():
        output = interim_dir / f"normal_{split}.csv"
        frame.to_csv(output, index=False)
        entries.append(_source_entry(split, output, normal_path, frame, normal_source_row_count))

    for family, source_path in attack_sources:
        frame = _read_attack(source_path, family, family)
        source_row_count = len(frame)
        if max_rows is not None:
            frame = frame.iloc[:max_rows].copy()
        mid = int(np.floor(len(frame) * cfg.attack_val_fraction))
        ranges = {"val": frame.iloc[:mid].copy(), "test": frame.iloc[mid:].copy()}
        for split, part in ranges.items():
            if part.empty:
                raise ValueError(f"{family} capture has no frames in {split} after splitting")
            output = interim_dir / f"{family}_{split}.csv"
            part.to_csv(output, index=False)
            entries.append(_source_entry(split, output, source_path, part, source_row_count))

    manifest = {
        "pipeline_version": "corrected_dense_autoencoder_v1",
        "normal_split_ratio": list(cfg.normal_split_ratio),
        "attack_val_fraction": cfg.attack_val_fraction,
        "window_len": cfg.window_len,
        "hop": cfg.hop,
        "is_debug": max_rows is not None,
        "max_rows": max_rows,
        "entries": entries,
    }
    manifest_path = interim_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote {manifest_path} with {len(entries)} independent ranges")
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--normal-split", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config.from_json(args.config) if args.config else Config()
    if args.normal_split:
        cfg.normal_split_ratio = tuple(args.normal_split)
    build_splits(cfg, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
