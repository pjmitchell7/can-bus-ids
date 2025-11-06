# I load raw CSVs, normalize features (fit on train only), pool into fixed-time windows,
# and write compact .npz arrays to data/processed for training/eval.

import os, sys, json, argparse, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# I ensure local imports resolve when running as a script.
sys.path.append(os.path.dirname(__file__))

from config import Config
from utils import parse_payload

# I map common HCRL-style headers to my canonical names.
_CAN_ALIASES = {
    "timestamp": ["Timestamp", "TimeStamp", "timeStamp", "timestamp", "Time", "time", "t", "ts", "Timestamp[ms]"],
    "can_id": ["CAN_ID", "CAN ID", "Arbitration_ID", "Arbitration ID", "can_id", "ID", "id"],
    "dlc": ["DLC", "dlc", "len", "Length"],
    "payload_str": ["payload", "Payload", "DATA", "data", "bytes", "BYTES", "frame", "Frame", "DataBytes"],
}

def _norm_map(cols):
    # I build a case-insensitive lookup map for existing columns.
    return {c.lower(): c for c in cols}

def _pick_col(df: pd.DataFrame, preferred: str, alts: list[str]):
    # I locate a column by preferred name or any alias (case-insensitive).
    lower = _norm_map(df.columns)
    if preferred.lower() in lower:
        return lower[preferred.lower()]
    for a in alts:
        if a.lower() in lower:
            return lower[a.lower()]
    raise KeyError

def _guess_time_col(df: pd.DataFrame):
    # I first try known aliases; if that fails, I guess by name and numeric monotonicity.
    try:
        return _pick_col(df, "Timestamp", _CAN_ALIASES["timestamp"])
    except KeyError:
        pass
    candidates = []
    for c in df.columns:
        if re.search(r"time|stamp|epoch", c, re.IGNORECASE):
            s = pd.to_numeric(df[c], errors="coerce")
            ok = s.notna().mean()
            if ok >= 0.8:
                rng = float(s.max() - s.min())
                candidates.append((ok, rng, c))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]
    scores = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        ok = s.notna().mean()
        rng = float((s.max() - s.min()) if ok > 0 else 0.0)
        scores.append((ok, rng, c))
    scores.sort(reverse=True)
    if scores and scores[0][0] >= 0.8:
        return scores[0][2]
    raise KeyError("I couldn't infer a timestamp column automatically.")

def _guess_can_id_col(df: pd.DataFrame):
    # I try aliases first, then guess by hex-like coverage.
    try:
        return _pick_col(df, "CAN_ID", _CAN_ALIASES["can_id"])
    except KeyError:
        pass
    best = None
    best_cov = -1.0
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        cov = s.str.match(r"^[0-9A-Fa-f]{2,8}$").mean()
        if cov > best_cov:
            best_cov = cov
            best = c
    if best is not None and best_cov >= 0.5:
        return best
    raise KeyError("I couldn't infer the CAN ID column automatically.")

def _find_payload_by_regex(df: pd.DataFrame):
    # I try to detect 8 per-byte columns via flexible regex:
    # Accept DATA0, data_0, DATA[0], Byte0, B0, etc., case-insensitive.
    name_map = {}
    for c in df.columns:
        m = re.match(r"^\s*(?:data|byte|b)\s*[\[\(_\-\s]*([0-7])[\]\)_\-\s]*\s*$", str(c), re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            name_map[idx] = c
    if len(name_map) == 8:
        return [name_map[i] for i in range(8)]
    # Fallback: pick any 8 columns whose values look like hex bytes with high coverage.
    scores = []
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        cov_hex = s.str.match(r"^[0-9A-Fa-f]{1,2}$").mean()
        cov_dec = pd.to_numeric(s, errors="coerce").between(0, 255).mean()
        score = max(cov_hex, cov_dec)
        scores.append((score, c))
    scores.sort(reverse=True)
    top = [c for sc, c in scores[:8] if sc >= 0.7]
    if len(top) == 8:
        # I preserve original order if they look like a contiguous block near the end.
        return top
    return None

def _infer_payload_cols(df: pd.DataFrame, cfg: Config):
    # I choose payload columns in this priority:
    # 1) Config payload_cols (case-insensitive) if present.
    # 2) DATA0..DATA7 (any case).
    # 3) DATA[0]..DATA[7] (any case).
    # 4) Regex-based detection for Byte0..Byte7 variants.
    # 5) Single payload string column.
    if cfg.payload_cols:
        cols = []
        lower = _norm_map(df.columns)
        ok = True
        for name in cfg.payload_cols:
            if name.lower() in lower:
                cols.append(lower[name.lower()])
            else:
                ok = False
                break
        if ok and cols:
            return cols, None
    names = [f"DATA{i}" for i in range(8)]
    lower = _norm_map(df.columns)
    if all(n.lower() in lower for n in names):
        return [lower[n.lower()] for n in names], None
    bracket = [f"DATA[{i}]" for i in range(8)]
    if all(b.lower() in lower for b in bracket):
        return [lower[b.lower()] for b in bracket], None
    by_regex = _find_payload_by_regex(df)
    if by_regex:
        return by_regex, None
    for cand in _CAN_ALIASES["payload_str"]:
        if cand in df.columns:
            return None, cand
        if cand.lower() in lower:
            return None, lower[cand.lower()]
    raise KeyError("I couldn't find payload bytes (DATA0..7 / DATA[0]..[7] / Byte0..7 / payload string).")

def _hex_to_int_series(s: pd.Series) -> pd.Series:
    # I convert a hex-like string column (e.g., 'fe', '0A', '00') to integers 0..255.
    def conv(x):
        if pd.isna(x):
            return 0
        v = str(x).strip().replace("0x","").replace("0X","")
        if v == "":
            return 0
        try:
            return int(v, 16)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return 0
    return s.apply(conv).clip(lower=0, upper=255).astype("int16")

def _coerce_payload_block(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    # I coerce each payload column to numeric bytes even if stored as hex, then stack as (N, 8).
    arrays = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            arr = s.to_numpy(dtype=np.float32, copy=False)
        else:
            arr = _hex_to_int_series(s).to_numpy(dtype=np.float32, copy=False)
        arrays.append(arr)
    return np.column_stack(arrays).astype(np.float32)

def _load_table(path: str, cfg: Config, max_rows: int | None):
    # I read with low_memory=False to avoid mixed-type chunk inference issues on huge files.
    df = pd.read_csv(path, nrows=max_rows, low_memory=False)

    # I find the timestamp column robustly.
    ts_col = _guess_time_col(df)

    # I find the CAN ID column robustly.
    id_col = _guess_can_id_col(df)

    # I find payload bytes.
    payload_cols, payload_str_col = _infer_payload_cols(df, cfg)

    if payload_cols is not None:
        # I remap bracketed names to flat names if needed and coerce hex to ints.
        if payload_cols and "[" in str(payload_cols[0]):
            flat = [f"DATA{i}" for i in range(8)]
            for src, dst in zip(payload_cols, flat):
                if dst not in df.columns:
                    df[dst] = df[src]
            payload_cols = flat
        pbytes = _coerce_payload_block(df, payload_cols)
    else:
        # I parse the single payload string column into 8 bytes.
        pbytes = np.stack(df[payload_str_col].apply(parse_payload).values).astype(np.float32)

    # I build time and inter-arrival features.
    t = pd.to_numeric(df[ts_col], errors="coerce").ffill().bfill().astype(float).to_numpy()
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0.0, 0.5).astype("float32").reshape(-1, 1)

    # I normalize CAN IDs to strings (keep hex as-is).
    ids_str = df[id_col].astype(str).to_numpy()

    return dict(pbytes=pbytes, ids_str=ids_str, dt=dt, t=t)

def _encode_ids(ids_str: np.ndarray, vocab: dict[str, int] | None):
    # I construct or apply a CAN ID vocabulary and return one-hot vectors.
    if vocab is None:
        uniq = sorted(set(ids_str.tolist()))
        vocab = {s: i for i, s in enumerate(uniq)}
    idx = np.array([vocab.get(s, -1) for s in ids_str], dtype=np.int32)
    idx[idx < 0] = 0
    onehot = np.eye(len(vocab), dtype=np.float32)[idx]
    return onehot, vocab

def _build_features(tbl: dict, cfg: Config, vocab: dict[str, int] | None, scaler: StandardScaler | None, fit: bool):
    # I concatenate payload bytes, ID one-hot, and delta_t.
    onehot, vocab = _encode_ids(tbl["ids_str"], vocab)
    parts = [tbl["pbytes"], onehot]
    if cfg.keep_delta_t:
        parts.append(tbl["dt"])
    X = np.concatenate(parts, axis=1).astype("float32")

    if fit:
        scaler = StandardScaler().fit(X)
    Xn = scaler.transform(X)
    return Xn, vocab, scaler

def _window_pool_mean(X: np.ndarray, t: np.ndarray, window_ms: int, hop_ms: int):
    # I implement fixed-duration windows by time and mean-pool rows within each window.
    w = window_ms / 1000.0
    h = hop_ms / 1000.0
    start = float(t[0]); end = float(t[-1])
    cur = start
    pooled = []
    anchors = []
    while cur + w <= end:
        mask = (t >= cur) & (t < cur + w)
        if mask.any():
            pooled.append(X[mask].mean(axis=0, keepdims=False))
            anchors.append(cur)
        cur += h
    if not pooled:
        return np.zeros((0, X.shape[1]), dtype="float32"), np.array([], dtype="float32")
    return np.stack(pooled).astype(np.float32), np.array(anchors, dtype=np.float32)

def _save_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print("Saved:", path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_rows", type=int, default=None,
                    help="Optional row cap per split for a quick dry run.")
    args = ap.parse_args()

    cfg = Config()

    train_csv = r"data\raw\train_normal.csv"
    val_csv   = r"data\raw\val_mix.csv"
    test_csv  = r"data\raw\test_mix.csv"

    # Train
    tr_tbl = _load_table(train_csv, cfg, max_rows=args.max_rows)
    X_tr, vocab, scaler = _build_features(tr_tbl, cfg, vocab=None, scaler=None, fit=True)
    Xw_tr, tw_tr = _window_pool_mean(X_tr, tr_tbl["t"], cfg.window_ms, cfg.hop_ms)
    _save_npz(r"data\processed\train_windows.npz", X=Xw_tr, t=tw_tr)

    # Val
    va_tbl = _load_table(val_csv, cfg, max_rows=args.max_rows)
    X_va, _, _ = _build_features(va_tbl, cfg, vocab=vocab, scaler=scaler, fit=False)
    Xw_va, tw_va = _window_pool_mean(X_va, va_tbl["t"], cfg.window_ms, cfg.hop_ms)
    _save_npz(r"data\processed\val_windows.npz", X=Xw_va, t=tw_va)

    # Test
    te_tbl = _load_table(test_csv, cfg, max_rows=args.max_rows)
    X_te, _, _ = _build_features(te_tbl, cfg, vocab=vocab, scaler=scaler, fit=False)
    Xw_te, tw_te = _window_pool_mean(X_te, te_tbl["t"], cfg.window_ms, cfg.hop_ms)
    _save_npz(r"data\processed\test_windows.npz", X=Xw_te, t=tw_te)

    # Artifacts
    with open(r"data\processed\id_vocab.json", "w") as f:
        json.dump(vocab, f)
    np.savez_compressed(r"data\processed\scaler_std.npz",
                        mean=scaler.mean_.astype("float32"),
                        scale=scaler.scale_.astype("float32"))
    print("Artifacts saved in data\\processed")

if __name__ == "__main__":
    main()
