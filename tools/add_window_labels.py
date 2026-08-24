"""Deprecated because labels must be created with the same starts as windows."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "add_window_labels.py is deprecated; regenerate data/processed windows with "
        "python -m src.preprocess"
    )


if __name__ == "__main__":
    main()
