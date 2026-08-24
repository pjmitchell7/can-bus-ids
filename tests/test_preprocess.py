import numpy as np
import pandas as pd

from src.config import Config
from src.preprocess import build_id_map, build_window_arrays, fit_scaler, window_starts


def _meta(n):
    return pd.DataFrame({
        "capture_id": ["capture"] * n,
        "attack_family": ["gear"] * n,
        "source_row": np.arange(n),
    })


def test_overlapping_window_labels_use_the_same_starts_as_features():
    assert window_starts(8, 4, 2).tolist() == [0, 2, 4]
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    arrays = build_window_arrays(np.arange(8, dtype=np.float32)[:, None], labels, _meta(8), 4, 2)
    assert arrays["y"].tolist() == [1, 1, 0]
    assert arrays["window_start_row"].tolist() == [0, 2, 4]
    assert arrays["window_end_row"].tolist() == [3, 5, 7]


def test_capture_shorter_than_window_cannot_be_joined():
    labels = np.zeros(3, dtype=np.int8)
    arrays = build_window_arrays(np.zeros((3, 10), dtype=np.float32), labels, _meta(3), 4, 2)
    assert arrays["X"].shape == (0, 4, 10)
    assert len(arrays["y"]) == 0


def test_scaler_is_fitted_without_validation_extreme_and_handles_constants():
    train = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float32)
    mean, std = fit_scaler(train)
    assert np.allclose(mean, [2.0, 5.0])
    assert np.allclose(std, [1.0, 1.0])
    validation = np.array([[1000.0, 5.0]], dtype=np.float32)
    assert np.allclose(mean, [2.0, 5.0])
    assert np.allclose(std, [1.0, 1.0])
    assert np.allclose((validation - mean) / std, [[998.0, 0.0]])


def test_id_map_is_stable_and_unknown_ids_do_not_expand_it():
    cfg = Config(device="cpu")
    frames = [
        pd.DataFrame({"CAN_ID": ["0x200", "100"]}),
        pd.DataFrame({"CAN_ID": ["300"]}),
    ]
    mapping = build_id_map(frames, cfg)
    assert mapping == {"100": 0, "200": 1, "300": 2}
    assert "400" not in mapping
