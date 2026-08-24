"""Stream-validate the five official HCRL Car-Hacking Dataset captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path


NORMAL_COLUMNS = [
    "Timestamp", "CAN_ID", "DLC",
    "DATA0", "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7",
]
CANONICAL_COLUMNS = [*NORMAL_COLUMNS, "Flag"]
EXPECTED_FILES = {
    "normal": "train_normal.csv",
    "dos": "DoS_dataset.csv",
    "fuzzy": "Fuzzy_dataset.csv",
    "gear": "gear_dataset.csv",
    "rpm": "RPM_dataset.csv",
}
HEX_BYTE = re.compile(r"^[0-9a-fA-F]{2}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_record(path: Path, kind: str) -> dict:
    return {
        "kind": kind,
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": 0,
        "raw_width_counts": {},
        "flag_counts": {"R": 0, "T": 0},
        "first_timestamp": None,
        "last_timestamp": None,
        "timestamps_non_decreasing": True,
        "parser_columns": CANONICAL_COLUMNS,
    }


def _check_frame_values(row: list[str], row_number: int, record: dict, *, attack: bool) -> None:
    width = len(row)
    record["raw_width_counts"][str(width)] = record["raw_width_counts"].get(str(width), 0) + 1
    if len(row) < 3:
        raise ValueError(f"{record['file']} row {row_number} has fewer than three fields")
    try:
        timestamp = float(row[0])
        dlc = int(row[2])
    except ValueError as exc:
        raise ValueError(f"{record['file']} row {row_number} has invalid Timestamp or DLC") from exc
    if not 0 <= dlc <= 8:
        raise ValueError(f"{record['file']} row {row_number} has DLC {dlc}, expected 0..8")
    previous = record["last_timestamp"]
    if previous is not None and timestamp < previous:
        record["timestamps_non_decreasing"] = False
        raise ValueError(f"{record['file']} timestamp decreases at row {row_number}")
    if record["first_timestamp"] is None:
        record["first_timestamp"] = timestamp
    record["last_timestamp"] = timestamp
    if not row[1].strip():
        raise ValueError(f"{record['file']} row {row_number} has an empty CAN_ID")

    expected_width = dlc + 4 if attack else 11
    if width != expected_width:
        raise ValueError(
            f"{record['file']} row {row_number} has {width} fields; expected {expected_width}"
        )
    payload_end = 3 + dlc if attack else 11
    for value in row[3:payload_end]:
        if not HEX_BYTE.fullmatch(value.strip()):
            raise ValueError(f"{record['file']} row {row_number} has invalid payload byte {value!r}")
    if attack:
        flag = row[3 + dlc].strip().upper()
        if flag not in {"R", "T"}:
            raise ValueError(f"{record['file']} row {row_number} has invalid Flag {flag!r}")
        record["flag_counts"][flag] += 1
    else:
        record["flag_counts"]["R"] += 1


def _validate_normal(path: Path) -> dict:
    record = _base_record(path, "normal")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != NORMAL_COLUMNS:
            raise ValueError(f"{path.name} header does not match the project parser")
        for row_number, row in enumerate(reader, start=2):
            if not row:
                raise ValueError(f"{path.name} contains a blank row at {row_number}")
            _check_frame_values(row, row_number, record, attack=False)
            record["row_count"] += 1
    record["flag_source"] = "implicit R supplied by src.make_splits._read_normal"
    return record


def _validate_attack(path: Path, kind: str) -> dict:
    record = _base_record(path, kind)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                raise ValueError(f"{path.name} contains a blank row at {row_number}")
            _check_frame_values(row, row_number, record, attack=True)
            record["row_count"] += 1
    record["flag_source"] = "raw HCRL Flag field after the DLC-dependent payload"
    return record


def validate(raw_dir: Path) -> dict:
    files = sorted(path.relative_to(raw_dir).as_posix() for path in raw_dir.rglob("*") if path.is_file())
    expected = sorted(EXPECTED_FILES.values())
    records = []
    for kind, filename in EXPECTED_FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(_validate_normal(path) if kind == "normal" else _validate_attack(path, kind))
    return {
        "dataset": "HCRL Car-Hacking Dataset",
        "source": "https://ocslab.hksecurity.net/Datasets/car-hacking-dataset",
        "verification_date": date.today().isoformat(),
        "raw_directory": str(raw_dir),
        "expected_files": expected,
        "present_files": files,
        "missing_files": sorted(set(expected) - set(files)),
        "unexpected_files": sorted(set(files) - set(expected)),
        "nested_files": [name for name in files if "/" in name],
        "flag_semantics": {"T": 1, "R": 0},
        "parser_columns": CANONICAL_COLUMNS,
        "records": records,
        "passed": files == expected and not any(not record["timestamps_non_decreasing"] for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset_verification.json"))
    args = parser.parse_args()
    report = validate(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Validated {len(report['records'])} captures; passed={report['passed']}; report={args.output}")


if __name__ == "__main__":
    main()
