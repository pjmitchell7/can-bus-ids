import numpy as np
import pytest

from src.evaluate import choose_threshold_f1, choose_threshold_percentile


def test_threshold_methods_are_explicit_and_use_intended_data():
    errors = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    threshold, percentile = choose_threshold_percentile(errors, 99.0)
    assert percentile == 99.0
    assert threshold > 0.8
    selected, selected_percentile = choose_threshold_f1(errors, y)
    assert selected in errors
    assert selected_percentile is None


def test_misspelled_threshold_method_is_rejected():
    from src.config import Config

    with pytest.raises(ValueError, match="Unknown threshold_method"):
        Config(device="cpu", threshold_method="f1_max").validate()
