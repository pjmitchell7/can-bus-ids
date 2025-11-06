# Central configuration for paths, preprocessing, model, and runtime.
# I keep mutable defaults out of dataclasses by using tuples instead of lists.

from dataclasses import dataclass

@dataclass
class Config:
    # Raw split paths
    train_csv: str = r"data/raw/train_normal.csv"
    val_csv:   str = r"data/raw/val_mix.csv"
    test_csv:  str = r"data/raw/test_mix.csv"

    # Column names
    timestamp_col: str = "Timestamp"
    can_id_col:    str = "CAN_ID"
    # I use a tuple to avoid the mutable-default error
    payload_cols: tuple[str, ...] = ("DATA0","DATA1","DATA2","DATA3","DATA4","DATA5","DATA6","DATA7")

    # Preprocess options
    id_as_onehot: bool = True
    keep_delta_t: bool = True
    normalize:    str = "zscore"
    window_ms:    int = 100
    hop_ms:       int = 50
    sample_rate_hz: int = 10000

    # Model settings
    model_type: str = "autoencoder"
    hidden_sizes: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.10
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    weight_decay: float = 1e-5

    # Thresholding
    thresh_method: str = "f1_max"
    thresh_percentile: float = 99.0

    # Runtime
    device: str = "cpu"
    out_dir: str = r"experiments"
