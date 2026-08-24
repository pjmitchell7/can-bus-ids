"""Train the unchanged dense autoencoder on corrected benign windows."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config, resolve_project_path
from .model_autoencoder import AE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=resolve_project_path("."),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "torch", "pytest"]
    versions = {name: importlib.metadata.version(name) for name in names if _has_package(name)}
    versions["python"] = platform.python_version()
    return versions


def _has_package(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _load_train_windows(cfg: Config) -> np.ndarray:
    path = resolve_project_path(cfg.processed_dir) / "train_windows.npz"
    with np.load(path, allow_pickle=False) as data:
        X = data["X"]
    if X.ndim != 3 or X.shape[1:] != (cfg.window_len, len(cfg.feature_order)):
        raise ValueError(f"Expected train windows [N,{cfg.window_len},10], got {X.shape}")
    return X.reshape(len(X), -1).astype(np.float32, copy=False)


def run_train(cfg: Config | None = None) -> Path:
    cfg = cfg or Config()
    cfg.validate()
    seed_everything(cfg.seed)
    X = _load_train_windows(cfg)
    if len(X) == 0:
        raise ValueError("No training windows were produced")

    experiment_dir = resolve_project_path(cfg.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = experiment_dir / f"run_corrected_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = experiment_dir / f"run_corrected_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir()

    tensor = torch.from_numpy(X)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    model = AE(in_dim=X.shape[1], hidden=cfg.hidden_sizes, dropout=cfg.dropout).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.MSELoss()

    losses: list[float] = []
    model.train()
    for epoch in range(cfg.epochs):
        total = 0.0
        seen = 0
        for (batch,) in loader:
            batch = batch.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        epoch_loss = total / max(1, seen)
        losses.append(epoch_loss)
        print(f"[{epoch + 1:02d}/{cfg.epochs:02d}] loss={epoch_loss:.6f}")

    torch.save(model.state_dict(), run_dir / "model.pt")
    np.save(run_dir / "losses.npy", np.asarray(losses, dtype=np.float32))
    cfg.save_json(run_dir / "cfg.json")

    processed_dir = resolve_project_path(cfg.processed_dir)
    for filename in ["can_id_map.json", "scaler.npz", "preprocess_meta.json"]:
        source = processed_dir / filename
        if source.exists():
            (run_dir / filename).write_bytes(source.read_bytes())

    metadata = {
        "pipeline_version": "corrected_dense_autoencoder_v1",
        "seed": cfg.seed,
        "git_commit": _git_commit(),
        "device": cfg.device,
        "input_shape": [cfg.window_len, len(cfg.feature_order)],
        "flattened_input_dim": int(X.shape[1]),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "dependency_versions": _versions(),
        "processed_dir": str(processed_dir),
        "is_debug": "debug_maxrows_" in str(processed_dir),
    }
    preprocess_meta = processed_dir / "preprocess_meta.json"
    if preprocess_meta.exists():
        with open(preprocess_meta, encoding="utf-8") as handle:
            metadata["data_hashes"] = json.load(handle).get("source_file_hashes", {})
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved run to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = Config.from_json(args.config) if args.config else Config()
    run_train(cfg)


if __name__ == "__main__":
    main()
