"""
Batch feature extraction over the full PTB-XL dataset.

Turns 21,799 raw ECG records into a single feature matrix (Parquet), joined
with the multi-hot diagnostic superclass labels and the official fold column.

Built for a long run:
    - Checkpoints to disk every N records; safe to interrupt and resume.
    - Skips records already in the checkpoint.
    - Records per-record failures without aborting the whole job.
    - Prints throughput and ETA.

Output:
    data/features/ptbxl_features.parquet   feature matrix + labels + strat_fold

Run:  python -m extract_all           (or: python extract_all.py)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from src.ecg.loader import PTBXLLoader, SUPERCLASSES
from src.ecg.features import extract_features

OUT_DIR = os.path.join("data", "features")
OUT_PATH = os.path.join(OUT_DIR, "ptbxl_features.parquet")
CKPT_PATH = os.path.join(OUT_DIR, "_checkpoint.parquet")
CHECKPOINT_EVERY = 500


def _fmt_eta(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def load_checkpoint() -> pd.DataFrame:
    if os.path.exists(CKPT_PATH):
        df = pd.read_parquet(CKPT_PATH)
        print(f"Resuming: {len(df):,} records already extracted.")
        return df
    return pd.DataFrame()


def save_checkpoint(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(CKPT_PATH):
        prev = pd.read_parquet(CKPT_PATH)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(CKPT_PATH, index=False)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    loader = PTBXLLoader()

    done = load_checkpoint()
    done_ids = set(done["ecg_id"]) if len(done) else set()

    all_ids = list(loader.metadata.index)
    todo = [i for i in all_ids if i not in done_ids]
    print(f"Total records: {len(all_ids):,}  |  to extract: {len(todo):,}\n")

    if not todo:
        print("All records already extracted. Building final matrix.")
    else:
        buffer: list[dict] = []
        failures = 0
        start = time.time()

        for n, ecg_id in enumerate(todo, start=1):
            try:
                record = loader.load_record(ecg_id)
                feats = extract_features(record)
            except Exception as exc:
                failures += 1
                print(f"  FAIL ecg_id={ecg_id}: {exc}")
                continue

            buffer.append(feats)

            if n % CHECKPOINT_EVERY == 0:
                save_checkpoint(buffer)
                buffer = []
                elapsed = time.time() - start
                rate = n / elapsed
                eta = (len(todo) - n) / rate
                print(f"  {n:,}/{len(todo):,}  "
                      f"({rate:.1f} rec/s, ETA {_fmt_eta(eta)}, "
                      f"{failures} failures)")

        save_checkpoint(buffer)
        print(f"\nExtraction complete. Failures: {failures}")

    # ---- build final matrix: features + labels + fold ----
    feats = pd.read_parquet(CKPT_PATH).set_index("ecg_id")

    labels = loader.metadata.loc[feats.index]
    label_matrix = loader.label_matrix(labels)
    for i, name in enumerate(SUPERCLASSES):
        feats[f"label_{name}"] = label_matrix[:, i]
    feats["strat_fold"] = labels["strat_fold"].values

    feats.to_parquet(OUT_PATH)
    print(f"\nFinal matrix written: {OUT_PATH}")
    print(f"  shape: {feats.shape}  "
          f"({feats.shape[1] - len(SUPERCLASSES) - 1} features + "
          f"{len(SUPERCLASSES)} labels + fold)")

    # NaN summary for the morphology columns
    morpho = [c for c in ("qrs_duration", "qt_interval", "pr_interval",
                          "p_duration") if c in feats.columns]
    print("\n  Morphology-feature availability:")
    for c in morpho:
        pct = 100 * feats[c].notna().mean()
        print(f"    {c:<14} {pct:5.1f}% present")


if __name__ == "__main__":
    main()