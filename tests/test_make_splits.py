import pandas as pd
import pytest

from src.make_splits import flag_to_label, split_contiguous


def test_flag_semantics_are_strict():
    values = flag_to_label(pd.Series(["T", "R", " t ", "r"]))
    assert values.tolist() == [1, 0, 1, 0]
    with pytest.raises(ValueError, match="Unexpected Flag values"):
        flag_to_label(pd.Series(["T", "X"]))


def test_contiguous_split_preserves_order_and_is_disjoint():
    frame = pd.DataFrame({"source_row": range(8)})
    parts = split_contiguous(frame, (0.5, 0.25, 0.25))
    assert parts["train"]["source_row"].tolist() == [0, 1, 2, 3]
    assert parts["val"]["source_row"].tolist() == [4, 5]
    assert parts["test"]["source_row"].tolist() == [6, 7]
    assert set(parts["train"]["source_row"]).isdisjoint(parts["val"]["source_row"])
    assert set(parts["train"]["source_row"]).isdisjoint(parts["test"]["source_row"])
