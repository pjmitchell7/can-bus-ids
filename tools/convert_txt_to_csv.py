# tools/convert_txt_to_csv.py
# Converts HCRL Car-Hacking text logs (e.g., "normal_run_data.txt") to CSV.
# Expected line example:
#   Timestamp: 1479121434.850202        ID: 0350    000    DLC: 8    05 28 84 66 6d 00 00 a2
# Output CSV columns:
#   Timestamp,CAN_ID,DLC,DATA0,...,DATA7

import csv
import os
import re

# Adjust these paths if needed
ROOT = r"C:\Users\mitch\Downloads\school\CAN_Bus_Security"
IN_PATH  = os.path.join(ROOT, r"data\raw\normal_run_data.txt")
OUT_PATH = os.path.join(ROOT, r"data\raw\train_normal.csv")

HEADER = ["Timestamp","CAN_ID","DLC","DATA0","DATA1","DATA2","DATA3","DATA4","DATA5","DATA6","DATA7"]

def parse_line(line: str):
    # Tokenize by whitespace but keep tokens intact
    tokens = line.strip().split()
    if not tokens:
        return None

    # Helper to get value that follows a label token like "Timestamp:" or "ID:" or "DLC:"
    def value_after(label):
        try:
            i = tokens.index(label)
            return tokens[i+1]
        except ValueError:
            return None
        except IndexError:
            return None

    # Some files may use lowercase labels; normalize by mapping
    # Build a map label -> index, case-insensitive, but keep original tokens for values
    lower_map = {tok.lower(): idx for idx, tok in enumerate(tokens)}

    def value_after_ci(label_lower_with_colon):
        # e.g., 'timestamp:' or 'id:' or 'dlc:'
        idx = lower_map.get(label_lower_with_colon, None)
        if idx is None:
            return None
        if idx + 1 >= len(tokens):
            return None
        return tokens[idx+1]

    # Try case-insensitive label extraction first
    ts = value_after_ci("timestamp:")
    can_id = value_after_ci("id:")
    dlc = value_after_ci("dlc:")

    # Fallback to exact-case variants if needed
    if ts is None: ts = value_after("Timestamp:")
    if can_id is None: can_id = value_after("ID:")
    if dlc is None: dlc = value_after("DLC:")

    # If still missing critical fields, try regex as a last resort
    if ts is None or can_id is None or dlc is None:
        m = re.search(r"Timestamp:\s*([0-9]+(?:\.[0-9]+)?)", line, re.I)
        if m and ts is None: ts = m.group(1)
        m = re.search(r"ID:\s*([0-9A-Fa-f]+)", line)
        if m and can_id is None: can_id = m.group(1)
        m = re.search(r"DLC:\s*([0-9]+)", line, re.I)
        if m and dlc is None: dlc = m.group(1)

    if ts is None or can_id is None or dlc is None:
        return None  # skip lines that don't match the expected structure

    # Find where the data bytes start: locate the token after "DLC: <n>"
    # Locate DLC token index in a case-insensitive way
    dlc_tok_idx = lower_map.get("dlc:", None)
    if dlc_tok_idx is None:
        try:
            dlc_tok_idx = tokens.index("DLC:")
        except ValueError:
            dlc_tok_idx = None

    data_start_idx = None
    if dlc_tok_idx is not None:
        data_start_idx = dlc_tok_idx + 2  # skip "DLC:" and the number
    else:
        # Fallback: try to find the dlc value and use the following tokens as data
        try:
            dlc_idx = tokens.index(dlc)
            data_start_idx = dlc_idx + 1
        except ValueError:
            data_start_idx = None

    # Some logs include an extra field like "000" between ID and DLC; it doesn't matter as long
    # as data_start_idx points to the first data byte.
    data_bytes = []
    if data_start_idx is not None and data_start_idx < len(tokens):
        # Collect remaining tokens as hex byte candidates
        for tok in tokens[data_start_idx:]:
            # Accept 1–2 hex chars (normalize to two), sometimes tokens may be like "0a" or "FF"
            if re.fullmatch(r"[0-9A-Fa-f]{1,2}", tok):
                data_bytes.append(tok)
            else:
                # In case of stray tokens, ignore
                pass

    # Ensure exactly 8 bytes, pad or truncate as needed
    data_bytes = [b.zfill(2).lower() for b in data_bytes[:8]]
    if len(data_bytes) < 8:
        data_bytes += ["00"] * (8 - len(data_bytes))

    # Normalize outputs
    ts_str = ts  # keep as string, CSV consumers can parse float
    can_id_str = can_id.lower()  # keep hex without 0x
    dlc_int = dlc

    row = [ts_str, can_id_str, dlc_int] + data_bytes
    return row

def convert(in_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = 0
    written = 0
    skipped = 0

    with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(HEADER)

        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            row = parse_line(line)
            if row is None:
                skipped += 1
                continue
            writer.writerow(row)
            written += 1

    print(f"Input:   {in_path}")
    print(f"Output:  {out_path}")
    print(f"Total lines:   {total}")
    print(f"Written rows:  {written}")
    print(f"Skipped lines: {skipped}")

if __name__ == "__main__":
    if not os.path.exists(IN_PATH):
        raise SystemExit(f"Input file not found: {IN_PATH}")
    convert(IN_PATH, OUT_PATH)
