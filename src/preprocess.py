"""Fit train-only preprocessing and build capture-safe window artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, resolve_project_path
from .utils import hex_byte_to_int, sha256_file


def window_starts(n_frames: int, window_len: int, hop: int) -> np.ndarray:
    """Return starts for overlapping windows that fit inside one frame range."""
    if n_frames < window_len:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_frames - window_len + 1, hop, dtype=np.int64)


def fit_scaler(X_train: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise population mean/std and protect constant features."""
    if X_train.ndim != 2 or len(X_train) == 0:
        raise ValueError("X_train must be a non-empty two-dimensional matrix")
    mean = X_train.mean(axis=0, dtype=np.float64)
    std = X_train.std(axis=0, dtype=np.float64)
    std[std < epsilon] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def build_window_arrays(
    X_frames: np.ndarray,
    labels: np.ndarray,
    frame_meta: pd.DataFrame,
    window_len: int,
    hop: int,
) -> dict[str, np.ndarray]:
    """Build features, labels, and metadata from one independent capture range."""
    if len(X_frames) != len(labels) or len(X_frames) != len(frame_meta):
        raise ValueError("frame features, labels, and metadata must have the same length")
    starts = window_starts(len(X_frames), window_len, hop)
    empty_X = np.empty((0, window_len, X_frames.shape[1]), dtype=np.float32)
    if len(starts) == 0:
        return {
            "X": empty_X,
            "y": np.empty(0, dtype=np.int8),
            "capture_id": np.empty(0, dtype="U1"),
            "attack_family": np.empty(0, dtype="U1"),
            "window_start_row": np.empty(0, dtype=np.int64),
            "window_end_row": np.empty(0, dtype=np.int64),
        }
    X = np.stack([X_frames[start:start + window_len] for start in starts]).astype(np.float32)
    y = np.array([labels[start:start + window_len].any() for start in starts], dtype=np.int8)
    return {
        "X": X,
        "y": y,
        "capture_id": np.asarray(frame_meta.iloc[starts]["capture_id"].astype(str), dtype="U"),
        "attack_family": np.asarray(frame_meta.iloc[starts]["attack_family"].astype(str), dtype="U"),
        "window_start_row": frame_meta.iloc[starts]["source_row"].astype(np.int64).to_numpy(),
        "window_end_row": frame_meta.iloc[starts + window_len - 1]["source_row"].astype(np.int64).to_numpy(),
    }


def _normalize_id(value: object) -> str:
    return str(value).strip().lower().replace("0x", "")


def _frame_matrix(frame: pd.DataFrame, cfg: Config, id_to_code: dict[str, int]) -> np.ndarray:
    payload = np.array(
        [[hex_byte_to_int(row[column]) for column in cfg.payload_cols] for _, row in frame.iterrows()],
        dtype=np.float32,
    )
    dlc = pd.to_numeric(frame[cfg.dlc_col], errors="coerce").fillna(0).clip(0, 8).to_numpy(dtype=np.float32)
    ids = frame[cfg.can_id_col].map(_normalize_id)
    can_id = ids.map(lambda value: id_to_code.get(value, cfg.unknown_id_code)).to_numpy(dtype=np.float32)
    return np.column_stack([payload, dlc, can_id]).astype(np.float32)


def build_id_map(frames: list[pd.DataFrame], cfg: Config) -> dict[str, int]:
    ids: set[str] = set()
    for frame in frames:
        ids.update(frame[cfg.can_id_col].map(_normalize_id).tolist())
    ids.discard("")
    return {can_id: index for index, can_id in enumerate(sorted(ids))}


def _save_windows(path: Path, arrays: dict[str, np.ndarray], include_labels: bool = True) -> None:
    values = {"X": arrays["X"].astype(np.float32)}
    if include_labels:
        values.update({key: value for key, value in arrays.items() if key != "X"})
    np.savez_compressed(path, **values)


def process(cfg: Config, max_rows: int | None = None, out_dir: str | Path | None = None) -> Path:
    """Process every manifest range independently and return the output directory."""
    cfg.validate()
    manifest_path = resolve_project_path(cfg.interim_dir) / "split_manifest.json"
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    root = resolve_project_path(".")
    output_dir = resolve_project_path(out_dir) if out_dir else resolve_project_path(cfg.processed_dir)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        if out_dir is None:
            output_dir = output_dir / f"debug_maxrows_{max_rows}"
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = manifest["entries"]
    train_entries = [entry for entry in entries if entry["split"] == "train"]
    if any(entry["attack_family"] != "normal" for entry in train_entries):
        raise ValueError("Training ranges must contain benign normal traffic only")
    train_frames = []
    for entry in train_entries:
        frame = pd.read_csv(root / entry["frame_path"], dtype={"CAN_ID": "string"})
        train_frames.append(frame.iloc[:max_rows] if max_rows else frame)
    if not train_frames:
        raise ValueError("The split manifest contains no training range")

    id_map = build_id_map(train_frames, cfg)
    train_matrix = np.concatenate([_frame_matrix(frame, cfg, id_map) for frame in train_frames], axis=0)
    train_mean, train_std = fit_scaler(train_matrix, cfg.scaler_epsilon)

    with open(output_dir / "can_id_map.json", "w", encoding="utf-8") as handle:
        json.dump({"mapping": id_map, "unknown_code": cfg.unknown_id_code}, handle, indent=2, sort_keys=True)
    np.savez_compressed(
        output_dir / "scaler.npz",
        mean=train_mean,
        std=train_std,
        epsilon=np.float32(cfg.scaler_epsilon),
    )

    grouped: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "val": [], "test": []}
    for entry in entries:
        frame = pd.read_csv(root / entry["frame_path"])
        if max_rows is not None:
            frame = frame.iloc[:max_rows].copy()
        X = _frame_matrix(frame, cfg, id_map)
        X = (X - train_mean) / train_std
        arrays = build_window_arrays(
            X,
            frame["label"].to_numpy(dtype=np.int8),
            frame,
            cfg.window_len,
            cfg.hop,
        )
        grouped[entry["split"]].append(arrays)

    def combine(split: str) -> dict[str, np.ndarray]:
        parts = grouped[split]
        if not parts:
            raise ValueError(f"No ranges found for split {split!r}")
        return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}

    train = combine("train")
    val = combine("val")
    test = combine("test")
    _save_windows(output_dir / "train_windows.npz", train, include_labels=False)
    _save_windows(output_dir / "val_windows.npz", val)
    _save_windows(output_dir / "test_windows.npz", test)

    source_hashes = {entry["source_path"]: entry["source_sha256"] for entry in entries}
    split_hashes = {
        entry["frame_path"]: sha256_file(root / entry["frame_path"])
        for entry in entries
    }
    meta = {
        "pipeline_version": "corrected_dense_autoencoder_v1",
        "feature_order": cfg.feature_order,
        "window_len": cfg.window_len,
        "hop": cfg.hop,
        "normalization": cfg.normalize,
        "scaler_epsilon": cfg.scaler_epsilon,
        "unknown_id_code": cfg.unknown_id_code,
        "id_map_filename": "can_id_map.json",
        "scaler_filename": "scaler.npz",
        "split_manifest_sha256": sha256_file(manifest_path),
        "source_file_hashes": source_hashes,
        "split_file_hashes": split_hashes,
        "is_debug": max_rows is not None,
        "debug_max_rows": max_rows,
        "window_counts": {
            "train": int(len(train["X"])),
            "val": int(len(val["X"])),
            "test": int(len(test["X"])),
        },
    }
    with open(output_dir / "preprocess_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Saved processed windows under {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = Config.from_json(args.config) if args.config else Config()
    process(cfg, max_rows=args.max_rows, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
