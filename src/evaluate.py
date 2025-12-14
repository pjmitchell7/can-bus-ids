# Evaluate an already trained autoencoder on val and test windows.
# I prefer window labels saved in the npz. If absent, I derive them from CSV.

from __future__ import annotations

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from sklearn.metrics import precision_recall_fscore_support

from .config import Config
from .model_autoencoder import AE

P = Path


def _p(*xs) -> str:
    return str(P(*xs))


def _load_npz_dict(path: str):
    return np.load(_p(path), allow_pickle=True)


def _device_banner(device: str) -> dict:
    info = {"device": device}
    if device.startswith("cuda") and th.cuda.is_available():
        idx = th.cuda.current_device()
        info["cuda_index"] = int(idx)
        info["cuda_name"] = th.cuda.get_device_name(idx)
        info["cuda_mem_total_gb"] = float(th.cuda.get_device_properties(idx).total_memory / (1024**3))
    return info


def _recon_errs_batched(model, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    """
    Compute reconstruction MSE per window in batches to avoid GPU OOM.
    X: (N, T, F)
    Returns: (N,) float32
    """
    N = int(X.shape[0])
    in_dim = int(np.prod(X.shape[1:]))

    errs = np.empty(N, dtype=np.float32)
    model.eval()

    with th.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            Z = X[i:j].reshape(j - i, in_dim)

            t = th.as_tensor(Z, dtype=th.float32, device=device)
            r, _ = model(t)

            e = th.mean((r - t) ** 2, dim=1)
            errs[i:j] = e.detach().cpu().to(th.float32).numpy()

    return errs


def _choose_threshold_percentile(errs: np.ndarray, percentile: float) -> tuple[float, float, float]:
    thr = float(np.percentile(errs, percentile))
    return thr, float(percentile), float("nan")


def _choose_threshold_f1(
    errs: np.ndarray,
    y: np.ndarray,
    q_min: float = 50.0,
    q_max: float = 99.9,
    steps: int = 400,
) -> tuple[float, float, float]:
    """
    Choose threshold that maximizes F1 on labeled data (val).
    Searches percentile thresholds from q_min..q_max.
    Returns: (thr, q, best_f1)
    """
    best = {"f1": -1.0, "thr": float(np.median(errs)), "q": float("nan")}

    for q in np.linspace(float(q_min), float(q_max), int(steps)):
        thr = float(np.percentile(errs, q))
        yhat = (errs >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "thr": float(thr), "q": float(q)}

    return best["thr"], best["q"], best["f1"]


def _choose_threshold_f1_capped(
    errs: np.ndarray,
    y: np.ndarray,
    max_flagged_pct: float = 20.0,
    q_min: float = 50.0,
    q_max: float = 99.9,
    steps: int = 400,
) -> tuple[float, float, float]:
    """
    Choose threshold to maximize F1 on labeled data, but only among thresholds
    that keep flagged rate <= max_flagged_pct.

    Returns: (thr, q, best_f1)
    """
    best = {"f1": -1.0, "thr": float(np.median(errs)), "q": float("nan")}

    for q in np.linspace(float(q_min), float(q_max), int(steps)):
        thr = float(np.percentile(errs, q))
        yhat = (errs >= thr).astype(int)
        flagged_pct = float(yhat.mean() * 100.0)

        if flagged_pct > float(max_flagged_pct):
            continue

        p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "thr": float(thr), "q": float(q)}

    return best["thr"], best["q"], best["f1"]


def _window_labels_from_csv(csv_path: str, N: int, T: int):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, usecols=["label"])
    lab = df["label"].astype(int).values
    need = int(N) * int(T)
    if len(lab) < need:
        return None
    lab = lab[:need].reshape(int(N), int(T))
    return lab.any(axis=1).astype(int)


def _try_attack_types_from_csv(csv_path: str, N: int, T: int) -> np.ndarray | None:
    """
    If csv has a string column describing attack type, return a per-window type label.
    The rule is majority type inside the window.
    """
    if not os.path.exists(csv_path):
        return None

    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    cand = None
    for c in ["attack", "attack_type", "type", "category", "attack_class"]:
        if c in cols:
            cand = c
            break
    if cand is None:
        return None

    df = pd.read_csv(csv_path, usecols=[cand])
    a = df[cand].astype("string").fillna("unknown").values

    need = int(N) * int(T)
    if len(a) < need:
        return None
    a = a[:need].reshape(int(N), int(T))

    out = np.empty(int(N), dtype=object)
    for i in range(int(N)):
        vals, counts = np.unique(a[i], return_counts=True)
        out[i] = str(vals[int(np.argmax(counts))])
    return out


def _scores_binary(errs: np.ndarray, y: np.ndarray | None, thr: float) -> dict:
    if y is None or len(np.unique(y)) <= 1:
        return dict(precision=float("nan"), recall=float("nan"), f1=float("nan"))
    yhat = (errs >= float(thr)).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return dict(precision=float(p), recall=float(r), f1=float(f1))


def _scores_by_attack(errs: np.ndarray, y: np.ndarray | None, attack_types: np.ndarray | None, thr: float) -> dict | None:
    if y is None or attack_types is None:
        return None

    yhat = (errs >= float(thr)).astype(int)
    out = {}
    for t in sorted(set(attack_types.tolist())):
        mask = (attack_types == t)
        if int(mask.sum()) < 10:
            continue
        yt = y[mask]
        yh = yhat[mask]
        if len(np.unique(yt)) <= 1:
            continue
        p, r, f1, _ = precision_recall_fscore_support(yt, yh, average="binary", zero_division=0)
        out[str(t)] = {"n": int(mask.sum()), "precision": float(p), "recall": float(r), "f1": float(f1)}
    return out if out else None


def run_eval(run_dir: str, cfg_override: dict | None = None):
    # Config
    with open(_p(run_dir, "cfg.json")) as f:
        cfg_dict = json.load(f)
    cfg = Config(**cfg_dict)
    if cfg_override:
        for k, v in cfg_override.items():
            setattr(cfg, k, v)

    # Defaults for eval safety on shared GPUs
    eval_batch_size = int(getattr(cfg, "eval_batch_size", 4096))
    thresh_percentile = float(getattr(cfg, "thresh_percentile", 99.5))

    # Threshold tuning knobs
    thr_method = str(getattr(cfg, "thr_method", "train_percentile"))
    thr_q_min = float(getattr(cfg, "thr_q_min", 50.0))
    thr_q_max = float(getattr(cfg, "thr_q_max", 99.9))
    thr_steps = int(getattr(cfg, "thr_steps", 400))
    max_flagged_pct = float(getattr(cfg, "max_flagged_pct", 20.0))

    # Load data dicts
    d_tr = _load_npz_dict(_p("data", "processed", "train_windows.npz"))
    d_va = _load_npz_dict(_p("data", "processed", "val_windows.npz"))
    d_te = _load_npz_dict(_p("data", "processed", "test_windows.npz"))
    X_tr, X_va, X_te = d_tr["X"], d_va["X"], d_te["X"]

    # Sanity on shapes
    if not (X_tr.shape[1:] == X_va.shape[1:] == X_te.shape[1:]):
        raise SystemExit(f"Split shapes disagree: train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    _, T, F = X_tr.shape
    N_va = int(X_va.shape[0])
    N_te = int(X_te.shape[0])
    in_dim = int(T * F)

    # Model
    model = AE(in_dim=in_dim, hidden=cfg.hidden_sizes, dropout=cfg.dropout).to(cfg.device).eval()
    sd = th.load(_p(run_dir, "model.pt"), map_location=cfg.device, weights_only=True)
    model.load_state_dict(sd)

    # Errors (batched)
    errs_tr = _recon_errs_batched(model, X_tr, cfg.device, batch_size=eval_batch_size)
    errs_va = _recon_errs_batched(model, X_va, cfg.device, batch_size=eval_batch_size)
    errs_te = _recon_errs_batched(model, X_te, cfg.device, batch_size=eval_batch_size)

    # Labels: prefer npz y, else from CSV
    y_val = d_va["y"] if "y" in d_va.files else _window_labels_from_csv(_p("data", "raw", "val_mix.csv"), N_va, int(T))
    y_test = d_te["y"] if "y" in d_te.files else _window_labels_from_csv(_p("data", "raw", "test_mix.csv"), N_te, int(T))

    # Optional attack type labels for breakdown
    atk_val = _try_attack_types_from_csv(_p("data", "raw", "val_mix.csv"), N_va, int(T))
    atk_test = _try_attack_types_from_csv(_p("data", "raw", "test_mix.csv"), N_te, int(T))

    # Threshold selection
    selected_on = "train_percentile"
    thr_q = float("nan")
    thr_f1 = float("nan")

    if thr_method == "f1_capped" and y_val is not None and len(np.unique(y_val)) > 1:
        thr, thr_q, thr_f1 = _choose_threshold_f1_capped(
            errs_va,
            y_val,
            max_flagged_pct=max_flagged_pct,
            q_min=thr_q_min,
            q_max=thr_q_max,
            steps=thr_steps,
        )
        selected_on = "val_f1_capped"
        print(f"[thr] selected_on={selected_on}  max_flagged_pct={max_flagged_pct:.3f}  q={thr_q:.3f}  thr={thr:.3f}  val_f1={thr_f1:.4f}")

    elif thr_method == "val_f1" and y_val is not None and len(np.unique(y_val)) > 1:
        thr, thr_q, thr_f1 = _choose_threshold_f1(errs_va, y_val, q_min=thr_q_min, q_max=thr_q_max, steps=thr_steps)
        selected_on = "val_f1"
        print(f"[thr] selected_on={selected_on}  q={thr_q:.3f}  thr={thr:.3f}  val_f1={thr_f1:.4f}")

    else:
        thr, thr_q, thr_f1 = _choose_threshold_percentile(errs_tr, thresh_percentile)
        selected_on = "train_percentile"
        print(f"[thr] selected_on={selected_on}  p={thresh_percentile:.3f}  thr={thr:.3f}")

    out = {
        "in_dim": int(in_dim),
        "eval_batch_size": int(eval_batch_size),
        "thresh_percentile": float(thresh_percentile),
        "thr_method": thr_method,
        "thr_q_min": float(thr_q_min),
        "thr_q_max": float(thr_q_max),
        "thr_steps": int(thr_steps),
        "max_flagged_pct": float(max_flagged_pct),
        "device_info": _device_banner(str(cfg.device)),
        "train_err_median": float(np.median(errs_tr)),
        "val_err_median": float(np.median(errs_va)),
        "test_err_median": float(np.median(errs_te)),
        "threshold": float(thr),
        "threshold_selected_on": selected_on,
        "threshold_percentile_on_val": float(thr_q),
        "val_f1_at_selected_threshold": float(thr_f1),
        "val_scores": _scores_binary(errs_va, y_val, thr),
        "test_scores": _scores_binary(errs_te, y_test, thr),
        "val_flagged_pct": float((errs_va >= thr).mean() * 100.0),
        "test_flagged_pct": float((errs_te >= thr).mean() * 100.0),
        "val_has_labels": bool(y_val is not None),
        "test_has_labels": bool(y_test is not None),
    }

    by_attack_val = _scores_by_attack(errs_va, y_val, atk_val, thr)
    by_attack_test = _scores_by_attack(errs_te, y_test, atk_test, thr)
    if by_attack_val is not None:
        out["val_by_attack"] = by_attack_val
    if by_attack_test is not None:
        out["test_by_attack"] = by_attack_test

    # Single batch latency probe
    with th.no_grad():
        sample_n = min(1024, len(X_va))
        sample = th.tensor(
            X_va[:sample_n].reshape(sample_n, -1),
            dtype=th.float32,
            device=cfg.device
        )
        for _ in range(5):
            _ = model(sample)
        t0 = time.perf_counter()
        _ = model(sample)
        out["latency_ms_sample"] = (time.perf_counter() - t0) * 1000.0

    with open(_p(run_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    exp = P("experiments")
    runs = sorted([p for p in exp.iterdir() if p.name.startswith("run_")]) if exp.is_dir() else []
    if not runs:
        raise SystemExit("No run_* folder under experiments/. Train first: python -m src.train")
    latest = str(runs[-1])
    run_eval(latest)
