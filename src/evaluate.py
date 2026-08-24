"""Evaluate corrected windows with explicit, reproducible thresholds and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

from .config import Config, resolve_project_path
from .metrics import classification_metrics, per_attack_metrics


ALLOWED_METHODS = {"train_percentile", "val_f1", "f1_capped"}


def _recon_errs_batched(model, X: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    flat = X.reshape(len(X), -1)
    errors = np.empty(len(flat), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            stop = min(start + batch_size, len(flat))
            tensor = torch.as_tensor(flat[start:stop], dtype=torch.float32, device=device)
            reconstruction, _ = model(tensor)
            errors[start:stop] = torch.mean((reconstruction - tensor) ** 2, dim=1).cpu().numpy()
    return errors


def choose_threshold_percentile(errors: np.ndarray, percentile: float) -> tuple[float, float | None]:
    return float(np.percentile(errors, percentile)), float(percentile)


def choose_threshold_f1(errors: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
    if len(np.unique(y)) < 2:
        raise ValueError("Validation-F1 threshold selection requires both classes")
    candidates = np.unique(errors)
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        prediction = (errors >= threshold).astype(np.int8)
        metrics = classification_metrics(y, prediction)
        f1 = metrics["f1"] or 0.0
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(f1)
    return best_threshold, None


def choose_threshold_f1_capped(
    errors: np.ndarray,
    y: np.ndarray,
    max_flagged_pct: float = 20.0,
) -> tuple[float, float | None]:
    if not 0 < max_flagged_pct <= 100:
        raise ValueError("max_flagged_pct must be in (0, 100]")
    if len(np.unique(y)) < 2:
        raise ValueError("Capped-F1 threshold selection requires both classes")
    best: tuple[float, float] | None = None
    for threshold in np.unique(errors):
        prediction = (errors >= threshold).astype(np.int8)
        if float(prediction.mean() * 100) > max_flagged_pct:
            continue
        f1 = classification_metrics(y, prediction)["f1"] or 0.0
        if best is None or f1 > best[1]:
            best = (float(threshold), float(f1))
    if best is None:
        raise ValueError("No threshold satisfies max_flagged_pct")
    return best[0], None


def _load_windows(path: Path, require_labels: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    required = {"X"}
    if require_labels:
        required |= {"y", "capture_id", "attack_family", "window_start_row", "window_end_row"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"{path} is missing required processed fields: {sorted(missing)}")
    return arrays


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch is not None:
        torch.cuda.synchronize()


def _measure_latency(model: AE, X: np.ndarray, cfg: Config) -> dict:
    batch_size = min(cfg.latency_batch_size, len(X))
    if batch_size == 0:
        raise ValueError("Cannot measure latency without validation windows")
    sample = torch.as_tensor(
        X[:batch_size].reshape(batch_size, -1),
        dtype=torch.float32,
        device=cfg.device,
    )
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(sample)
        values = []
        for _ in range(cfg.latency_repetitions):
            _sync_if_cuda(cfg.device)
            start = time.perf_counter()
            model(sample)
            _sync_if_cuda(cfg.device)
            values.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "repetitions": cfg.latency_repetitions,
        "batch_size": batch_size,
        "input_shape": list(X[:batch_size].shape),
        "device": cfg.device,
        "dtype": "float32",
        "measurement": "forward_pass_only",
    }


def _save_scores(
    run_dir: Path,
    method: str,
    split: str,
    scores: np.ndarray,
    threshold: float,
    arrays: dict[str, np.ndarray],
) -> None:
    predictions = (scores >= threshold).astype(np.int8)
    np.savez_compressed(
        run_dir / f"scores_{method}_{split}.npz",
        scores=scores.astype(np.float32),
        predictions=predictions,
        y=arrays["y"].astype(np.int8),
        capture_id=arrays["capture_id"],
        attack_family=arrays["attack_family"],
        window_start_row=arrays["window_start_row"],
        window_end_row=arrays["window_end_row"],
    )


def run_eval(
    run_dir: str | Path,
    threshold_method: str | None = None,
    threshold_percentile: float | None = None,
) -> Path:
    if torch is None:
        raise RuntimeError("Torch is required for model evaluation; install requirements.txt first")
    from .model_autoencoder import AE

    run_dir = Path(run_dir)
    with open(run_dir / "cfg.json", encoding="utf-8") as handle:
        cfg = Config(**json.load(handle))
    if threshold_method is not None:
        cfg.threshold_method = threshold_method
    if threshold_percentile is not None:
        cfg.threshold_percentile = threshold_percentile
    cfg.validate()
    if cfg.threshold_method not in ALLOWED_METHODS:
        raise ValueError(f"Unknown threshold method: {cfg.threshold_method!r}")

    processed = resolve_project_path(cfg.processed_dir)
    train = _load_windows(processed / "train_windows.npz", require_labels=False)
    val = _load_windows(processed / "val_windows.npz", require_labels=True)
    test = _load_windows(processed / "test_windows.npz", require_labels=True)
    shapes = [train["X"].shape[1:], val["X"].shape[1:], test["X"].shape[1:]]
    if not (shapes[0] == shapes[1] == shapes[2]):
        raise ValueError(f"Window shapes disagree: {shapes}")

    in_dim = int(np.prod(train["X"].shape[1:]))
    model = AE(in_dim=in_dim, hidden=cfg.hidden_sizes, dropout=cfg.dropout).to(cfg.device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=cfg.device, weights_only=True))
    train_scores = _recon_errs_batched(model, train["X"], cfg.device, cfg.eval_batch_size)
    val_scores = _recon_errs_batched(model, val["X"], cfg.device, cfg.eval_batch_size)
    test_scores = _recon_errs_batched(model, test["X"], cfg.device, cfg.eval_batch_size)

    if cfg.threshold_method == "train_percentile":
        threshold, selected_percentile = choose_threshold_percentile(train_scores, cfg.threshold_percentile)
        threshold_source = "train"
    elif cfg.threshold_method == "val_f1":
        threshold, selected_percentile = choose_threshold_f1(val_scores, val["y"])
        threshold_source = "validation"
    else:
        threshold, selected_percentile = choose_threshold_f1_capped(val_scores, val["y"])
        threshold_source = "validation"

    split_results = {}
    for name, scores, arrays in [("val", val_scores, val), ("test", test_scores, test)]:
        predictions = (scores >= threshold).astype(np.int8)
        metrics = classification_metrics(arrays["y"], predictions, scores)
        metrics["per_attack"] = per_attack_metrics(
            arrays["y"], predictions, scores, arrays["attack_family"]
        )
        split_results[name] = metrics
        _save_scores(run_dir, cfg.threshold_method, name, scores, threshold, arrays)

    output = {
        "pipeline_version": "corrected_dense_autoencoder_v1",
        "threshold_method": cfg.threshold_method,
        "threshold": threshold,
        "threshold_source_split": threshold_source,
        "threshold_percentile": selected_percentile,
        "threshold_percentile_config": cfg.threshold_percentile,
        "val": split_results["val"],
        "test": split_results["test"],
        "latency": _measure_latency(model, val["X"], cfg),
    }
    output_path = run_dir / f"metrics_{cfg.threshold_method}.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    aggregate_path = run_dir / "metrics.json"
    aggregate = {}
    if aggregate_path.exists():
        with open(aggregate_path, encoding="utf-8") as handle:
            aggregate = json.load(handle)
    aggregate[cfg.threshold_method] = output
    with open(aggregate_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(json.dumps(output, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--threshold-method", choices=sorted(ALLOWED_METHODS), default=None)
    parser.add_argument("--threshold-percentile", type=float, default=None)
    args = parser.parse_args()
    run_eval(args.run, args.threshold_method, args.threshold_percentile)


if __name__ == "__main__":
    main()
