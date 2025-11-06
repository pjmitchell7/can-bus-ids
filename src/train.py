# Train the autoencoder on benign-only windows produced by preprocess.py.
# I write all artifacts into a timestamped run directory under experiments/.

import os, time, json
import numpy as np
import torch as th
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import asdict

from config import Config
from model_autoencoder import AE

def _load_npz(path):
    # I load a compressed npz with keys X (windows) and t (anchors).
    d = np.load(path)
    X = d["X"].astype("float32")
    return X

def run_train(cfg: Config = Config()):
    # I make a unique output folder for this training run.
    os.makedirs(cfg.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_dir, f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    # I load train windows (benign only).
    train_npz = r"data\processed\train_windows.npz"
    X = _load_npz(train_npz)

    # I construct the dataset and data loader.
    X_tensor = th.tensor(X, dtype=th.float32)
    ds = TensorDataset(X_tensor)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # I build the model using the input feature dimension from the data.
    in_dim = X_tensor.shape[1]
    model = AE(in_dim=in_dim, hidden=cfg.hidden_sizes, dropout=cfg.dropout).to(cfg.device)

    # I choose MSE reconstruction loss and Adam optimizer for this baseline.
    opt = th.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = th.nn.MSELoss()

    # I optionally control threads on CPU to stabilize latency and results.
    if cfg.device == "cpu":
        th.set_num_threads(max(1, th.get_num_threads()))

    # I train for the configured number of epochs and record mean epoch loss.
    losses = []
    model.train()
    for ep in range(cfg.epochs):
        ep_loss = 0.0
        seen = 0
        for (xb,) in dl:
            xb = xb.to(cfg.device)
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

    # I save model weights and run metadata for reproducibility.
    th.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    np.save(os.path.join(run_dir, "losses.npy"), np.array(losses, dtype=np.float32))

    # I copy preprocessing artifacts so evaluate.py can reconstruct transforms consistently.
    # These files are produced by preprocess.py.
    for fname in ["id_vocab.json", "scaler_std.npz"]:
        src = os.path.join("data", "processed", fname)
        dst = os.path.join(run_dir, fname)
        if os.path.exists(src):
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read())

    # I persist the config I used for this run.
    with open(os.path.join(run_dir, "cfg.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("Saved run to", run_dir)
    return run_dir

if __name__ == "__main__":
    run_train()
