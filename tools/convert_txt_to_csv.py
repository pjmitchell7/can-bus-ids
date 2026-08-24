"""Convert HCRL text logs to the CSV format used by the split builder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re


HEADER = [
    "Timestamp", "CAN_ID", "DLC",
    "DATA0", "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7",
]


def parse_line(line: str) -> list[str] | None:
    tokens = line.strip().split()
    if not tokens:
        return None
    lower = {token.lower(): index for index, token in enumerate(tokens)}

    def after(label: str) -> str | None:
        index = lower.get(label)
        return tokens[index + 1] if index is not None and index + 1 < len(tokens) else None

    timestamp = after("timestamp:")
    can_id = after("id:")
    dlc = after("dlc:")
    if timestamp is None:
        match = re.search(r"Timestamp:s*([0-9]+(?:.[0-9]+)?)", line, re.I)
        timestamp = match.group(1) if match else None
    if can_id is None:
        match = re.search(r"ID:s*([0-9A-Fa-f]+)", line)
        can_id = match.group(1) if match else None
    if dlc is None:
        match = re.search(r"DLC:s*([0-9]+)", line, re.I)
        dlc = match.group(1) if match else None
    if timestamp is None or can_id is None or dlc is None:
        return None

    dlc_index = lower.get("dlc:")
    data_tokens = tokens[dlc_index + 2:] if dlc_index is not None else []
    data = [token.lower().zfill(2) for token in data_tokens if re.fullmatch(r"[0-9A-Fa-f]{1,2}", token)][:8]
    data.extend(["00"] * (8 - len(data)))
    return [timestamp, can_id.lower().replace("0x", ""), dlc, *data]


def convert(in_path: str | Path, out_path: str | Path) -> dict[str, int]:
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = written = skipped = 0
    with open(in_path, encoding="utf-8", errors="ignore") as source, open(
        out_path, "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(HEADER)
        for line in source:
            total += 1
            row = parse_line(line)
            if row is None:
                skipped += 1
            else:
                writer.writerow(row)
                written += 1
    stats = {"total_lines": total, "written_rows": written, "skipped_lines": skipped}
    print(f"Input: {in_path}")
    print(f"Output: {out_path}")
    print(f"Total lines: {total}")
    print(f"Written rows: {written}")
    print(f"Skipped lines: {skipped}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
