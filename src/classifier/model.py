"""
Multi-label ECG diagnostic classifier.

Structure (Decision D-010): one-vs-rest — five independent binary XGBoost
models, one per diagnostic superclass (NORM, MI, STTC, CD, HYP). Each model
has its own class-imbalance weighting and its own decision threshold tuned on
the validation fold. Rationale over a single MultiOutputClassifier:
    - per-class scale_pos_weight (HYP 12% vs NORM 44% need different weighting)
    - per-class threshold tuning -> better macro F1
    - native NaN handling for the morphology features (no imputation)

Unlabeled records (Decision D-011): the ~1.9% of records with no diagnostic
superclass are excluded from train / val / test, matching published PTB-XL
protocol. They are preserved by the loader for a separate "no finding"
sanity check by the agent, but do not enter the benchmark evaluation here.

Evaluation: macro AUROC and per-class AUROC on fold 10, comparable to the
PTB-XL literature.

Run:  python -m src.classifier.model
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

FEATURES_PATH = os.path.join("data", "features", "ptbxl_features.parquet")
MODEL_DIR = os.path.join("models", "ecg_classifier")

TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10

# Non-feature columns to exclude from X.
LABEL_COLS = [f"label_{c}" for c in SUPERCLASSES]
META_COLS = ["strat_fold"] + LABEL_COLS


@dataclass
class ClassMetrics:
    auroc: float
    f1: float
    precision: float
    recall: float
    threshold: float


@dataclass
class EvalReport:
    per_class: dict = field(default_factory=dict)
    macro_auroc: float = 0.0
    macro_f1: float = 0.0

    def show(self) -> None:
        print(f"\n{'class':<6} {'AUROC':>7} {'F1':>7} {'prec':>7} "
              f"{'recall':>7} {'thresh':>7}")
        print("-" * 46)
        for name, m in self.per_class.items():
            print(f"{name:<6} {m.auroc:>7.3f} {m.f1:>7.3f} {m.precision:>7.3f} "
                  f"{m.recall:>7.3f} {m.threshold:>7.2f}")
        print("-" * 46)
        print(f"{'MACRO':<6} {self.macro_auroc:>7.3f} {self.macro_f1:>7.3f}")


def load_matrix() -> pd.DataFrame:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run extract_all.py first."
        )
    return pd.read_parquet(FEATURES_PATH)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def split_xy(df: pd.DataFrame, feat_cols: list[str]):
    # D-011: exclude records with no diagnostic superclass (all-zero labels)
    # from the benchmark. They remain in the parquet for other uses.
    has_label = df[LABEL_COLS].sum(axis=1) > 0
    df = df[has_label]

    fold = df["strat_fold"]
    splits = {}
    for name, sel in (
        ("train", fold.isin(TRAIN_FOLDS)),
        ("val", fold == VAL_FOLD),
        ("test", fold == TEST_FOLD),
    ):
        sub = df[sel]
        X = sub[feat_cols].to_numpy(dtype=np.float32)
        Y = sub[LABEL_COLS].to_numpy(dtype=np.int8)
        splits[name] = (X, Y)
    return splits


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising F1 on the validation set."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train(df: pd.DataFrame | None = None, verbose: bool = True):
    """Train one XGBoost per superclass. Returns (models, thresholds, feat_cols)."""
    if df is None:
        df = load_matrix()

    feat_cols = feature_columns(df)
    splits = split_xy(df, feat_cols)
    Xtr, Ytr = splits["train"]
    Xval, Yval = splits["val"]

    if verbose:
        print(f"Features: {len(feat_cols)}  |  "
              f"train={len(Xtr):,} val={len(Xval):,} test={len(splits['test'][0]):,}")

    models, thresholds = {}, {}

    for i, name in enumerate(SUPERCLASSES):
        ytr, yval = Ytr[:, i], Yval[:, i]

        pos = ytr.sum()
        neg = len(ytr) - pos
        spw = neg / pos if pos > 0 else 1.0

        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)

        val_prob = clf.predict_proba(Xval)[:, 1]
        thr = _best_threshold(yval, val_prob)

        models[name] = clf
        thresholds[name] = thr

        if verbose:
            print(f"  trained {name:<5} (pos_weight={spw:5.2f}, thresh={thr:.2f})")

    return models, thresholds, feat_cols


def evaluate(models, thresholds, feat_cols, df=None, split="test") -> EvalReport:
    if df is None:
        df = load_matrix()
    splits = split_xy(df, feat_cols)
    X, Y = splits[split]

    report = EvalReport()
    aurocs, f1s = [], []

    for i, name in enumerate(SUPERCLASSES):
        y_true = Y[:, i]
        y_prob = models[name].predict_proba(X)[:, 1]
        thr = thresholds[name]
        y_pred = (y_prob >= thr).astype(int)

        auroc = roc_auc_score(y_true, y_prob)
        report.per_class[name] = ClassMetrics(
            auroc=auroc,
            f1=f1_score(y_true, y_pred, zero_division=0),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            threshold=thr,
        )
        aurocs.append(auroc)
        f1s.append(report.per_class[name].f1)

    report.macro_auroc = float(np.mean(aurocs))
    report.macro_f1 = float(np.mean(f1s))
    return report


def save(models, thresholds, feat_cols, out_dir: str = MODEL_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, clf in models.items():
        clf.save_model(os.path.join(out_dir, f"{name}.json"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"thresholds": thresholds, "features": feat_cols,
                   "superclasses": SUPERCLASSES}, f, indent=2)
    print(f"\nSaved models + meta to {out_dir}/")


def load(out_dir: str = MODEL_DIR):
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    models = {}
    for name in meta["superclasses"]:
        clf = XGBClassifier()
        clf.load_model(os.path.join(out_dir, f"{name}.json"))
        models[name] = clf
    return models, meta["thresholds"], meta["features"]


def main() -> None:
    df = load_matrix()
    models, thresholds, feat_cols = train(df)

    print("\n=== Validation (fold 9) ===")
    evaluate(models, thresholds, feat_cols, df, split="val").show()

    print("\n=== Test (fold 10) ===")
    evaluate(models, thresholds, feat_cols, df, split="test").show()

    save(models, thresholds, feat_cols)


if __name__ == "__main__":
    main()