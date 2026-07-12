"""
Agent tools.

Two distinct capabilities with different trust levels:

  run_classification (deterministic core) — loads a PTB-XL record, extracts
  features, runs the one-vs-rest XGBoost classifier, applies tuned thresholds.
  This ALWAYS runs, in fixed order, and is fully reproducible. It is the ground
  truth the agent reasons over; the LLM cannot skip or alter it. Not an
  LLM-callable tool — it is a graph node.

  lookup_guidelines (LLM-callable) — semantic retrieval over the cardiology
  reference vector store. The LLM decides when and what to retrieve; every call
  is recorded in the decision trace.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ecg.loader import PTBXLLoader
from src.ecg.features import extract_features
from src.classifier.model import load as load_models, SUPERCLASSES


class ECGClassifier:
    """Deterministic core: record -> features -> per-class probabilities."""

    def __init__(self):
        self.loader = PTBXLLoader()
        self.models, self.thresholds, self.feat_cols = load_models()

    def classify(self, ecg_id: int) -> dict:
        record = self.loader.load_record(ecg_id)
        feats = extract_features(record)

        # Align to the exact training feature order; missing -> NaN (XGBoost ok).
        row = pd.DataFrame([feats]).reindex(columns=self.feat_cols)
        X = row.to_numpy(dtype=np.float32)

        predictions = {}
        for name in SUPERCLASSES:
            prob = float(self.models[name].predict_proba(X)[:, 1][0])
            thr = float(self.thresholds[name])
            predictions[name] = {
                "probability": round(prob, 4),
                "positive": prob >= thr,
                "threshold": thr,
            }

        positive = [n for n in SUPERCLASSES if predictions[n]["positive"]]

        return {
            "ecg_id": int(ecg_id),
            "age": float(record.age),
            "sex": "F" if record.sex else "M",
            "predictions": predictions,
            "positive_findings": positive,
            "leading_finding": max(
                SUPERCLASSES, key=lambda n: predictions[n]["probability"]
            ),
            # Included for demo/eval transparency only; not shown to the model
            # as ground truth during report generation.
            "reference_labels": list(record.labels),
        }