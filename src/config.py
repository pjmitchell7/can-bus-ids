"""Central configuration for the corrected baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


_DEFAULT_DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    experiment_dir: str = "experiments"

    normal_file: str = "train_normal.csv"
    attack_files: tuple[tuple[str, str], ...] = (
        ("dos", "DoS_dataset.csv"),
        ("fuzzy", "Fuzzy_dataset.csv"),
        ("gear", "gear_dataset.csv"),
        ("rpm", "RPM_dataset.csv"),
    )

    timestamp_col: str = "Timestamp"
    can_id_col: str = "CAN_ID"
    dlc_col: str = "DLC"
    payload_cols: tuple[str, ...] = (
        "DATA0", "DATA1", "DATA2", "DATA3",
        "DATA4", "DATA5", "DATA6", "DATA7",
    )

    # Leave this unset by default so full-data runs must specify a split policy.
    normal_split_ratio: tuple[float, float, float] | None = None
    attack_val_fraction: float = 0.5

    window_len: int = 64
    hop: int = 32
    normalize: str = "zscore"
    scaler_epsilon: float = 1e-8
    unknown_id_code: int = -1

    hidden_sizes: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 30

    threshold_method: str = "train_percentile"
    threshold_percentile: float = 99.0
    eval_batch_size: int = 4096
    latency_batch_size: int = 1024
    latency_repetitions: int = 100

    seed: int = 42
    device: str = _DEFAULT_DEVICE

    def __post_init__(self) -> None:
        if isinstance(self.payload_cols, list):
            self.payload_cols = tuple(self.payload_cols)
        if isinstance(self.hidden_sizes, list):
            self.hidden_sizes = tuple(self.hidden_sizes)
        if isinstance(self.attack_files, list):
            self.attack_files = tuple(tuple(item) for item in self.attack_files)
        if self.normal_split_ratio is not None and isinstance(self.normal_split_ratio, list):
            self.normal_split_ratio = tuple(self.normal_split_ratio)

    @property
    def feature_order(self) -> list[str]:
        return [*self.payload_cols, self.dlc_col, self.can_id_col]

    def validate(self, require_normal_split: bool = False) -> None:
        if self.normalize != "zscore":
            raise ValueError(f"Unsupported normalize value: {self.normalize!r}")
        if len(self.payload_cols) != 8:
            raise ValueError("The baseline requires exactly eight payload columns")
        if self.window_len <= 0 or self.hop <= 0:
            raise ValueError("window_len and hop must be positive")
        if self.hop > self.window_len:
            raise ValueError("hop cannot be greater than window_len")
        if self.scaler_epsilon <= 0:
            raise ValueError("scaler_epsilon must be positive")
        if self.threshold_method not in {"train_percentile", "val_f1", "f1_capped"}:
            raise ValueError(f"Unknown threshold_method: {self.threshold_method!r}")
        if not 0 < self.threshold_percentile < 100:
            raise ValueError("threshold_percentile must be between 0 and 100")
        if self.latency_repetitions <= 0 or self.latency_batch_size <= 0:
            raise ValueError("latency settings must be positive")
        if self.normal_split_ratio is None:
            if require_normal_split:
                raise ValueError(
                    "normal_split_ratio must be explicitly configured before creating full splits"
                )
        else:
            if len(self.normal_split_ratio) != 3 or any(x <= 0 for x in self.normal_split_ratio):
                raise ValueError("normal_split_ratio must contain three positive values")
            if abs(sum(self.normal_split_ratio) - 1.0) > 1e-8:
                raise ValueError("normal_split_ratio must sum to 1")
        if not 0 < self.attack_val_fraction < 1:
            raise ValueError("attack_val_fraction must be between 0 and 1")
        if self.device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
            raise ValueError("CUDA was requested but is not available")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as handle:
            cfg = cls(**json.load(handle))
        cfg.validate()
        return cfg

    def save_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_root() / value
