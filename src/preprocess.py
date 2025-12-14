# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Optional
import json
import numpy as np
import pandas as pd
import torch

# ---- Config -----------------------------------------------------------------

@dataclass
class Config:
    # Raw CSVs
    train_csv: str = "data/raw/train_normal.csv"
    val_csv:   str = "data/raw/val_mix.csv"
    test_csv:  str = "data/raw/test_mix.csv"

    # Expected columns
    ts_col: str = "Timestamp"
    id_col: str = "CAN_ID"
    dlc_col: str = "DLC"
    payload_cols: List[str] = field(default_factory=lambda: [
        "DATA0","DATA1","DATA2","DATA3","DATA4","DATA5","DATA6","DATA7"
    ])

    # Windowing
    window_len: int = 64   # frames per window
    hop:        int = 32   # stride

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    out_dir: str = "data/processed"

# ---- Utilities ---------------------------------------------------------------

def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _hex_lookup_table() -> np.ndarray:
    # Map b"00".."ff" to 0..255 for vectorized hex decode
    lut = np.full((256, 256), -1, dtype=np.int16)
    hexd = b"0123456789abcdefABCDEF"
    for hi in range(256):
        for lo in range(256):
            s = bytes([hi, lo])
            if s[0] in hexd and s[1] in hexd:
                try:
                    lut[hi, lo] = int(s.decode("ascii"), 16)
                except Exception:
                    pass
    return lut

_LUT = _hex_lookup_table()

def _hex_series_to_uint8(series: pd.Series) -> np.ndarray:
    # Convert series of strings like "fe","0a" to uint8
    s = series.astype("string").fillna("00").str.strip().str.slice(0, 2).astype("string")
    arr = s.to_numpy(dtype=object)
    first  = np.frombuffer("".join([x[0] if len(x) > 0 else "0" for x in arr]).encode("ascii"), dtype=np.uint8)
    second = np.frombuffer("".join([x[1] if len(x) > 1 else "0" for x in arr]).encode("ascii"), dtype=np.uint8)
    out = _LUT[first, second].astype(np.int16)
    out[out < 0] = 0
    return out.astype(np.uint8)

def _read_can_csv(path: str, cfg: Config, max_rows: Optional[int]) -> pd.DataFrame:
    # Build desired usecols, then intersect with actual header so I can optionally pull label
    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    desired = [cfg.ts_col, cfg.id_col, cfg.dlc_col] + cfg.payload_cols
    if "label" in header_cols:
        desired += ["label"]
    usecols = [c for c in desired if c in header_cols]

    dtypes = {
        cfg.ts_col:  "float64",
        cfg.id_col:  "string",
        cfg.dlc_col: "int16",
        **{c: "string" for c in cfg.payload_cols},
    }
    if "label" in usecols:
        dtypes["label"] = "int8"

    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtypes,
        nrows=max_rows,
        engine="c",
        low_memory=False,
    )

    # Normalize CAN_ID to lowercase hex without 0x
    if cfg.id_col in df.columns:
        df[cfg.id_col] = df[cfg.id_col].astype("string").str.lower().str.replace("0x", "", regex=False)

    # Clip DLC to [0,8]
    if cfg.dlc_col in df.columns:
        df[cfg.dlc_col] = pd.to_numeric(df[cfg.dlc_col], errors="coerce").fillna(0).clip(0, 8).astype(np.int16)

    return df

def _extract_payload_uint8(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    cols = []
    for c in cfg.payload_cols:
        cols.append(_hex_series_to_uint8(df[c]) if c in df.columns else np.zeros(len(df), dtype=np.uint8))
    return np.stack(cols, axis=1)  # (N, 8)

def _encode_can_id(df: pd.DataFrame, cfg: Config, id_to_code: dict[str, int] | None) -> np.ndarray:
    if cfg.id_col not in df.columns:
        return np.zeros(len(df), dtype=np.int32)

    s = df[cfg.id_col].astype("string").fillna("").str.lower().str.replace("0x", "", regex=False)

    if id_to_code is None:
        # Fallback, but we should not use this in the final pipeline anymore
        codes, _ = pd.factorize(s)
        return codes.astype(np.int32)

    # Map using the shared vocabulary; unknown IDs become -1
    mapped = s.map(lambda x: id_to_code.get(str(x), -1)).to_numpy(dtype=np.int32, copy=False)
    return mapped

def _build_feature_matrix(df: pd.DataFrame, cfg: Config, id_to_code: dict[str, int] | None) -> np.ndarray:
    p = _extract_payload_uint8(df, cfg).astype(np.float32)
    dlc = (df[cfg.dlc_col].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)) if cfg.dlc_col in df.columns else np.zeros((len(df), 1), dtype=np.float32)
    cid = _encode_can_id(df, cfg, id_to_code).astype(np.float32).reshape(-1, 1)
    X = np.concatenate([p, dlc, cid], axis=1)
    return X

def _windows_torch(X_np: np.ndarray, win: int, hop: int, device: str) -> np.ndarray:
    """
    Slide along time axis producing (M, win, F)
    X_np: (N, F)
    """
    N, F = X_np.shape
    if N < win:
        raise ValueError(f"not enough rows ({N}) for window_len={win}")

    t = torch.from_numpy(X_np)  # (N, F)
    if device == "cuda":
        t = t.to("cuda", non_blocking=True)

    # Unfold along N after transpose to (F, N)
    u = t.transpose(0, 1).unfold(dimension=1, size=win, step=hop).contiguous()  # (F, M, win)
    out = u.permute(1, 2, 0).contiguous()  # (M, win, F)
    return out.detach().cpu().numpy()

def _derive_window_labels(df: pd.DataFrame, win: int, hop: int) -> Optional[np.ndarray]:
    if "label" not in df.columns:
        return None
    lab = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    N = len(lab)
    if N < win:
        return None
    M = 1 + max(0, (N - win) // hop)
    y = np.zeros(M, dtype=np.int32)
    start = 0
    for i in range(M):
        y[i] = int(np.any(lab[start:start + win] == 1))
        start += hop
    return y

def _process_one(path: str, cfg: Config, max_rows: Optional[int], out_name: str, id_to_code: dict[str, int] | None) -> tuple[str, Optional[np.ndarray]]:
    df = _read_can_csv(path, cfg, max_rows)
    X = _build_feature_matrix(df, cfg, id_to_code)                          # (N, F)
    W = _windows_torch(X, cfg.window_len, cfg.hop, cfg.device)              # (M, win, F)
    y_win = _derive_window_labels(df, cfg.window_len, cfg.hop)

    _ensure_outdir(cfg.out_dir)
    out_path = os.path.join(cfg.out_dir, out_name)
    if y_win is not None:
        np.savez_compressed(out_path, X=W.astype(np.float32), y=y_win.astype(np.int32))
    else:
        np.savez_compressed(out_path, X=W.astype(np.float32))

    print(f"Saved: {out_path}  windows={W.shape[0]}  win_len={W.shape[1]}  feat_dim={W.shape[2]}")
    return out_path, y_win

def _save_can_id_map(path: str, id_to_code: dict[str, int]) -> None:
    with open(path, "w") as f:
        json.dump(id_to_code, f, indent=2, sort_keys=True)

def _load_can_id_map(path: str) -> dict[str, int]:
    with open(path, "r") as f:
        return json.load(f)

def _build_can_id_map_from_train(df: pd.DataFrame, cfg: Config) -> dict[str, int]:
    # Stable ordering: sort unique IDs so mapping is deterministic
    ids = (
        df[cfg.id_col]
        .astype("string")
        .fillna("")
        .str.lower()
        .str.replace("0x", "", regex=False)
        .unique()
    )
    ids = sorted([x for x in ids if x is not None])
    return {cid: i for i, cid in enumerate(ids)}


# ---- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=None, help="limit rows for quick debugging")
    args = parser.parse_args()

    cfg = Config()
    print(f"Device: {cfg.device}")

    # Train: read once, build ID map, save it
    df_train = _read_can_csv(cfg.train_csv, cfg, args.max_rows)
    id_map = _build_can_id_map_from_train(df_train, cfg)
    _ensure_outdir(cfg.out_dir)
    id_map_path = os.path.join(cfg.out_dir, "can_id_map.json")
    _save_can_id_map(id_map_path, id_map)
    print(f"Saved: {id_map_path}  num_ids={len(id_map)}")

    # Process train using the map
    X_train = _build_feature_matrix(df_train, cfg, id_map)
    W_train = _windows_torch(X_train, cfg.window_len, cfg.hop, cfg.device)
    out_train = os.path.join(cfg.out_dir, "train_windows.npz")
    np.savez_compressed(out_train, X=W_train.astype(np.float32))
    print(f"Saved: {out_train}  windows={W_train.shape[0]}  win_len={W_train.shape[1]}  feat_dim={W_train.shape[2]}")

    # Val/test: load the map and apply consistently
    id_map = _load_can_id_map(id_map_path)
    _process_one(cfg.val_csv, cfg, args.max_rows, "val_windows.npz", id_map)
    _process_one(cfg.test_csv, cfg, args.max_rows, "test_windows.npz", id_map)


if __name__ == "__main__":
    main()
