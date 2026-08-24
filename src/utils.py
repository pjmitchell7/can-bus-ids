"""Small parsing and reproducibility helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def parse_payload(s: str) -> list[int]:
    """Convert a payload string or byte token to eight integer byte values."""
    value = str(s).strip().replace(" ", "").replace("0x", "").replace("0X", "")
    if not value:
        return [0] * 8
    chunks = [value[i:i + 2] for i in range(0, len(value), 2)][:8]
    result: list[int] = []
    for chunk in chunks:
        try:
            number = int(chunk, 16)
        except ValueError:
            number = 0
        result.append(max(0, min(255, number)))
    return (result + [0] * 8)[:8]


def hex_byte_to_int(value: object) -> int:
    """Convert one hexadecimal byte token to an integer, defaulting malformed data to zero."""
    try:
        number = int(str(value).strip().replace("0x", "").replace("0X", ""), 16)
    except (TypeError, ValueError):
        return 0
    return max(0, min(255, number))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
