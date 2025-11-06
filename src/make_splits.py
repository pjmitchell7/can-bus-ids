# I rebuild val_mix.csv and test_mix.csv cleanly from the raw HCRL files.
# I accept headerless attack CSVs (timestamp, CAN_ID, DLC, up to 8 bytes, Flag).
# I pad missing DATA bytes when DLC < 8 and write consistent headers.

import os
import math
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# I define the canonical column order I want in the mixes.
COLS = ["Timestamp","CAN_ID","DLC",
        "DATA0","DATA1","DATA2","DATA3","DATA4","DATA5","DATA6","DATA7",
        "Flag","label"]

def _read_attack(path: Path) -> pd.DataFrame:
    # I read the headerless attack file and coerce to the 11 fields + Flag.
    # Some rows have fewer than 8 DATA bytes (DLC < 8); I pad them with "00".
    # I do not touch hex strings; bytes remain as strings like 'fe', '0A', etc.
    # Names include placeholders for all 11 fields; Pandas fills missing with NaN.
    names = ["Timestamp","CAN_ID","DLC",
             "DATA0","DATA1","DATA2","DATA3","DATA4","DATA5","DATA6","DATA7",
             "Flag"]
    df = pd.read_csv(path, header=None, names=names, dtype=str)
    # I pad missing DATA* with "00" and fill missing DLC with '8' if absent.
    for b in [f"DATA{i}" for i in range(8)]:
        if b in df.columns:
            df[b] = df[b].fillna("00")
        else:
            df[b] = "00"
    if "DLC" in df.columns:
        df["DLC"] = df["DLC"].fillna("8")
    else:
        df["DLC"] = "8"
    # I ensure core columns exist.
    if "Flag" not in df.columns:
        df["Flag"] = "R"
    # I attach label=1 (attack).
    df["label"] = 1
    # I coerce Timestamp to float so ordering by time works.
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    # I keep only the canonical columns.
    return df[COLS]

def _read_normal(path: Path) -> pd.DataFrame:
    # I read the normal file that already has headers.
    df = pd.read_csv(path, dtype=str)
    # I add Flag if missing (normal rows are 'R' in these sets).
    if "Flag" not in df.columns:
        df["Flag"] = "R"
    # I add label=0 (benign).
    df["label"] = 0
    # I reorder to canonical columns.
    return df[COLS]

def _concat_and_shuffle(parts: list[pd.DataFrame]) -> pd.DataFrame:
    # I concatenate and sort by timestamp (stable) to approximate a natural stream.
    df = pd.concat(parts, ignore_index=True)
    # Timestamps might have NaN; I fill with forward/backward to keep order deterministic.
    ts = pd.to_numeric(df["Timestamp"], errors="coerce")
    ts = ts.ffill().bfill()
    df["Timestamp"] = ts
    # I sort by time to mix sources deterministically.
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    return df

def main():
    # I read sources.
    normal_csv = RAW / "train_normal.csv"
    dos_csv    = RAW / "DoS_dataset.csv"
    fuzzy_csv  = RAW / "Fuzzy_dataset.csv"
    gear_csv   = RAW / "gear_dataset.csv"
    rpm_csv    = RAW / "RPM_dataset.csv"

    df_norm  = _read_normal(normal_csv)
    df_dos   = _read_attack(dos_csv)
    df_fuzzy = _read_attack(fuzzy_csv)
    df_gear  = _read_attack(gear_csv)
    df_rpm   = _read_attack(rpm_csv)

    # I build a pooled attack set and cap each attack type to the min count for balance.
    attacks = [df_dos, df_fuzzy, df_gear, df_rpm]
    n_min = min(len(x) for x in attacks)
    attacks_bal = [x.iloc[:n_min].copy() for x in attacks]
    df_attack_all = _concat_and_shuffle(attacks_bal)

    # I split attacks 50/50 into val and test.
    nA = len(df_attack_all)
    n_val = nA // 2
    df_attack_val = df_attack_all.iloc[:n_val].copy()
    df_attack_test = df_attack_all.iloc[n_val:].copy()

    # I take an equal number of benign rows for val and test (or as many as available).
    df_norm_sorted = df_norm.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    nB_val = min(len(df_norm_sorted) // 2, len(df_attack_val))
    nB_test = min(len(df_norm_sorted) - nB_val, len(df_attack_test))

    df_norm_val = df_norm_sorted.iloc[:nB_val].copy()
    df_norm_test = df_norm_sorted.iloc[nB_val:nB_val+nB_test].copy()

    # I assemble val/test mixes and sort by time.
    df_val = _concat_and_shuffle([df_norm_val, df_attack_val])
    df_test = _concat_and_shuffle([df_norm_test, df_attack_test])

    # I write with header and without index.
    out_val = RAW / "val_mix.csv"
    out_test = RAW / "test_mix.csv"
    df_val.to_csv(out_val, index=False)
    df_test.to_csv(out_test, index=False)

    print(f"Wrote: {out_val} ({len(df_val):,} rows)")
    print(f"Wrote: {out_test} ({len(df_test):,} rows)")

if __name__ == "__main__":
    main()
