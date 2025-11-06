# Evaluate the trained autoencoder on validation and test windows.
# I select a threshold on validation (F1-max or percentile) and report metrics on test.

import os, json, time
import numpy as np
import pandas as pd
import torch as th
from sklearn.metrics import precision_recall_fscore_support

from config import Config
from model_autoencoder import AE

def _load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype("float32")
    return X

def _choose_threshold(errs_val, y_val, method="f1_max", percentile=99.0):
    # I support two strategies: F1-max (supervised) and percentile (unsupervised fallback).
    if y_val is None or method == "percentile":
        return float(np.percentile(errs_val, percentile))
    best_f1, best_t = -1.0, float(np.median(errs_val))
    # I sweep a modest set of candidate thresholds to avoid O(N^2) over unique values on huge arrays.
    qs = np.linspace(50, 99.9, 100)
    for q in qs:
        t = float(np.percentile(errs_val, q))
        yhat = (errs_val >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_val, yhat, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def run_eval(run_dir: str, cfg_override: dict | None = None):
    # I load the saved config for the run, then allow selective overrides.
    with open(os.path.join(run_dir, "cfg.json")) as f:
        cfg_dict = json.load(f)
    cfg = Config(**cfg_dict)
    if cfg_override:
        for k, v in cfg_override.items():
            setattr(cfg, k, v)

    # I load processed windows produced by preprocess.py.
    tr_npz = r"data\processed\train_windows.npz"
    va_npz = r"data\processed\val_windows.npz"
    te_npz = r"data\processed\test_windows.npz"

    X_val = _load_npz(va_npz)
    X_test = _load_npz(te_npz)

    # I build the model with the correct input dimension and load weights.
    in_dim = X_val.shape[1]
    model = AE(in_dim=in_dim, hidden=cfg.hidden_sizes, dropout=cfg.dropout)
    model.load_state_dict(th.load(os.path.join(run_dir, "model.pt"), map_location=cfg.device))
    model.to(cfg.device).eval()

    # I compute reconstruction errors on validation to choose a threshold.
    with th.no_grad():
        Xv = th.tensor(X_val, dtype=th.float32, device=cfg.device)
        Xvh, _ = model(Xv)
        errs_val = th.mean((Xvh - Xv) ** 2, dim=1).cpu().numpy()

    # I derive optional validation labels from val_mix.csv if present (label column).
    # If labels are absent, I fall back to percentile thresholding.
    val_csv = r"data\raw\val_mix.csv"
    if os.path.exists(val_csv):
        dfv = pd.read_csv(val_csv)
        y_val = dfv["label"].astype(int).values[:len(errs_val)] if "label" in dfv.columns else None
    else:
        y_val = None

    thr = _choose_threshold(errs_val, y_val, method=cfg.thresh_method, percentile=cfg.thresh_percentile)

    # I evaluate on test windows using the selected threshold.
    with th.no_grad():
        Xt = th.tensor(X_test, dtype=th.float32, device=cfg.device)
        Xth, _ = model(Xt)
        errs_test = th.mean((Xth - Xt) ** 2, dim=1).cpu().numpy()

    test_csv = r"data\raw\test_mix.csv"
    if os.path.exists(test_csv):
        dft = pd.read_csv(test_csv)
        y_test = dft["label"].astype(int).values[:len(errs_test)] if "label" in dft.columns else None
    else:
        y_test = None

    if y_test is not None:
        yhat = (errs_test >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, yhat, average="binary", zero_division=0)
        metrics = dict(precision=float(p), recall=float(r), f1=float(f1), threshold=float(thr))
    else:
        # I still report the chosen threshold if I don't have test labels.
        metrics = dict(precision=float("nan"), recall=float("nan"), f1=float("nan"), threshold=float(thr))

    # I also report a quick single-batch latency sample to track feasibility.
    with th.no_grad():
        sample = th.tensor(X_val[:1024], dtype=th.float32, device=cfg.device)
        t0 = time.perf_counter()
        _ = model(sample)
        latency_ms = (time.perf_counter() - t0) * 1000.0
    metrics["latency_ms_sample"] = float(latency_ms)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)
    return metrics

if __name__ == "__main__":
    # I allow running this file directly by pointing to the most recent run folder.
    # If I want to evaluate a specific run, I pass its path here or call run_eval from a one-liner.
    latest = None
    exp = "experiments"
    if os.path.isdir(exp):
        runs = sorted([os.path.join(exp, d) for d in os.listdir(exp) if d.startswith("run_")])
        latest = runs[-1] if runs else None
    if latest is None:
        raise SystemExit("No run_* folder found under experiments/. I train first with: python -m src.train")
    run_eval(latest)
