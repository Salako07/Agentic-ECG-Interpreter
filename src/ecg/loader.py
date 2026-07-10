"""
PTB-XL data loader.

Responsibilities:
    - Load and parse ptbxl_database.csv (metadata + SCP diagnostic codes)
    - Map raw SCP-ECG codes to the 5 diagnostic superclasses via
      scp_statements.csv
    - Load 12-lead waveforms from disk via wfdb
    - Provide the official stratified fold splits used by PTB-XL benchmarks

The 5 diagnostic superclasses:
    NORM  Normal ECG
    MI    Myocardial Infarction
    STTC  ST/T Change
    CD    Conduction Disturbance
    HYP   Hypertrophy

A single ECG may carry multiple superclasses, so this is a multi-label problem.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import wfdb

DATA_DIR = os.path.join("data", "ptbxl")

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Official PTB-XL protocol: folds 1-8 train, 9 validation, 10 test.
# Folds 9 and 10 were human-validated, so they make the cleanest eval sets.
TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10


@dataclass
class ECGRecord:
    """A single 12-lead ECG with its metadata and labels."""

    ecg_id: int
    signal: np.ndarray            # shape (n_samples, 12), millivolts
    sampling_rate: int            # 100 Hz for the _lr records
    lead_names: list[str]
    labels: list[str]             # diagnostic superclasses, may be empty
    scp_codes: dict[str, float]   # raw SCP code -> likelihood
    age: float
    sex: int                      # 0 = male, 1 = female
    report: str                   # free-text cardiologist report

    @property
    def duration_sec(self) -> float:
        return self.signal.shape[0] / self.sampling_rate


class PTBXLLoader:
    """Loads PTB-XL metadata and waveforms from a local directory."""

    def __init__(self, data_dir: str = DATA_DIR, sampling_rate: int = 100):
        if sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500")

        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self._filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

        self.metadata = self._load_metadata()
        self.scp_map = self._load_scp_map()
        self.metadata["labels"] = self.metadata["scp_codes"].apply(self._to_superclasses)

    # ---------- loading ----------

    def _path(self, name: str) -> str:
        path = os.path.join(self.data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run the download script first."
            )
        return path

    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self._path("ptbxl_database.csv"), index_col="ecg_id")
        # scp_codes is stored as a string repr of a dict, e.g. "{'NORM': 100.0}"
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
        return df

    def _load_scp_map(self) -> dict[str, str]:
        """Map SCP code -> diagnostic superclass, for diagnostic codes only."""
        scp = pd.read_csv(self._path("scp_statements.csv"), index_col=0)
        scp = scp[scp["diagnostic"] == 1]
        return scp["diagnostic_class"].dropna().to_dict()

    def _to_superclasses(self, scp_codes: dict) -> list[str]:
        """Collapse raw SCP codes into unique diagnostic superclasses."""
        found = {self.scp_map[code] for code in scp_codes if code in self.scp_map}
        return sorted(found)

    # ---------- access ----------

    def load_record(self, ecg_id: int) -> ECGRecord:
        """Load one ECG by its ecg_id."""
        if ecg_id not in self.metadata.index:
            raise KeyError(f"ecg_id {ecg_id} not in metadata")

        row = self.metadata.loc[ecg_id]
        record_path = os.path.join(self.data_dir, row[self._filename_col])

        signal, meta = wfdb.rdsamp(record_path)

        return ECGRecord(
            ecg_id=int(ecg_id),
            signal=signal,
            sampling_rate=int(meta["fs"]),
            lead_names=list(meta["sig_name"]),
            labels=row["labels"],
            scp_codes=row["scp_codes"],
            age=float(row["age"]),
            sex=int(row["sex"]),
            report=str(row.get("report", "")),
        )

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train, val, test) metadata frames using official folds."""
        fold = self.metadata["strat_fold"]
        return (
            self.metadata[fold.isin(TRAIN_FOLDS)],
            self.metadata[fold == VAL_FOLD],
            self.metadata[fold == TEST_FOLD],
        )

    def label_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Multi-hot label matrix of shape (len(df), 5) ordered by SUPERCLASSES."""
        index = {name: i for i, name in enumerate(SUPERCLASSES)}
        matrix = np.zeros((len(df), len(SUPERCLASSES)), dtype=np.int8)

        for row_i, labels in enumerate(df["labels"]):
            for label in labels:
                matrix[row_i, index[label]] = 1

        return matrix


def _smoke_test() -> None:
    """Load the dataset and print a summary. Run: python -m src.ecg.loader"""
    loader = PTBXLLoader()

    print(f"Records loaded: {len(loader.metadata):,}")
    print(f"Diagnostic SCP codes mapped: {len(loader.scp_map)}\n")

    print("Superclass distribution (an ECG may have several):")
    for name in SUPERCLASSES:
        count = loader.metadata["labels"].apply(lambda ls: name in ls).sum()
        pct = 100 * count / len(loader.metadata)
        print(f"  {name:<5} {count:>6,}  ({pct:4.1f}%)")

    unlabeled = (loader.metadata["labels"].str.len() == 0).sum()
    print(f"  {'none':<5} {unlabeled:>6,}  ({100 * unlabeled / len(loader.metadata):4.1f}%)")

    train, val, test = loader.split()
    print(f"\nSplit: train={len(train):,}  val={len(val):,}  test={len(test):,}")

    record = loader.load_record(1)
    print(f"\nSample record ecg_id=1")
    print(f"  signal shape : {record.signal.shape}")
    print(f"  sampling rate: {record.sampling_rate} Hz")
    print(f"  duration     : {record.duration_sec:.1f} s")
    print(f"  leads        : {', '.join(record.lead_names)}")
    print(f"  age / sex    : {record.age:.0f} / {'F' if record.sex else 'M'}")
    print(f"  scp_codes    : {record.scp_codes}")
    print(f"  labels       : {record.labels or ['(none)']}")


if __name__ == "__main__":
    _smoke_test()