# Train the autoencoder on benign-only windows produced by preprocess.py.
# I write all artifacts into a timestamped run directory under experiments/.

from __future__ import annotations
import os, time, json
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch as th
from torch.utils.data import TensorDataset, DataLoader

from .config import Config
from .model_autoencoder import AE


def _load_npz_windows(npz_path: Path) -> np.ndarray:
    """Load windows from npz and normalize to shape [N, F_flat]."""
    d = np.load(str(npz_path))
    X = d["X"]

    # Expected common cases:
    #  - [N, L, D]  -> flatten to [N, L*D]
    #  - [N, F]     -> already flat
    #  - [F, N]     -> transpose to [N, F] (rare, but handle)
    #  - anything else -> collapse all but batch dim
    if X.ndim == 3:
        N, L, D = X.shape
        X = X.reshape(N, L * D)
    elif X.ndim == 2:
        # Heuristic: if rows are tiny (e.g., 10 features) and columns huge (looks like N),
        # it’s probably transposed; flip to [N, F].
        if X.shape[0] < 64 and X.shape[1] > X.shape[0] * 1000:
            X = X.T
        # else: already [N, F]
    elif X.ndim > 3:
        N = X.shape[0]
        X = X.reshape(N, -1)
    else:
        raise ValueError(f"Unexpected X.ndim={X.ndim} for {npz_path}")

    # Safety check: no zero columns
    if X.shape[1] == 0:
        raise ValueError(f"No features found after reshape for {npz_path}")

    return X.astype("float32", copy=False)


def _join(*parts: str | os.PathLike) -> str:
    return os.path.join(*map(str, parts))


def run_train(cfg: Config = Config()):
    # Make run dir
    os.makedirs(cfg.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = _join(cfg.out_dir, f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Load train windows robustly (OS-neutral path)
    train_npz = _join("data", "processed", "train_windows.npz")
    if not os.path.exists(train_npz):
        # Try alternative separators if someone wrote backslashes literally
        alt_path = Path("data") / "processed" / "train_windows.npz"
        if alt_path.exists():
            train_npz = str(alt_path)
        else:
            raise FileNotFoundError(f"Could not find train windows at {train_npz}")

    X = _load_npz_windows(Path(train_npz))

    # Build dataset/loader
    device = cfg.device
    X_tensor = th.tensor(X, dtype=th.float32)
    ds = TensorDataset(X_tensor)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # Model
    in_dim = X_tensor.shape[1]  # flattened feature count
    model = AE(in_dim=in_dim, hidden=cfg.hidden_sizes, dropout=cfg.dropout).to(device)

    # Loss/opt
    opt = th.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = th.nn.MSELoss()

    if device == "cpu":
        th.set_num_threads(max(1, th.get_num_threads()))

    # Train
    losses = []
    model.train()
    for ep in range(cfg.epochs):
        ep_loss = 0.0
        seen = 0
        for (xb,) in dl:
            # Ensure 2D [B, F] before the first Linear
            if xb.ndim > 2:
                xb = xb.view(xb.size(0), -1)
            elif xb.ndim == 1:
                xb = xb.view(1, -1)

            # Guard against accidental mismatches
            if xb.shape[1] != in_dim:
                raise RuntimeError(
                    f"Input feature mismatch: xb has {xb.shape[1]} features, "
                    f"but model was built with in_dim={in_dim}."
                )

            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            xhat, _ = model(xb)
            loss = crit(xhat, xb)
            loss.backward()
            opt.step()

            ep_loss += loss.item() * xb.size(0)
            seen += xb.size(0)

        ep_loss /= max(1, seen)
        losses.append(ep_loss)
        print(f"[{ep+1:02d}/{cfg.epochs:02d}] loss={ep_loss:.6f}")

    # Save artifacts
    th.save(model.state_dict(), _join(run_dir, "model.pt"))
    np.save(_join(run_dir, "losses.npy"), np.array(losses, dtype=np.float32))

    # Copy preprocessing artifacts if present
    for fname in ["id_vocab.json", "scaler_std.npz"]:
        src = _join("data", "processed", fname)
        dst = _join(run_dir, fname)
        if os.path.exists(src):
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read())

    # Save the effective config
    with open(_join(run_dir, "cfg.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("Saved run to", run_dir)
    return run_dir


if __name__ == "__main__":
    run_train()
