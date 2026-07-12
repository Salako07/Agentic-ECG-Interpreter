"""
ECG feature extraction.

Strategy:
    - Rich delineation on Lead II (clinical rhythm lead): heart rate, HRV,
      wave morphology, and interval durations via neurokit2.
    - Lightweight statistical features on all 12 leads: amplitude, energy,
      and shape descriptors. This gives the classifier spatial information
      (e.g. a Q-wave in inferior leads) without full 12-lead delineation.

The output is a single flat dict per record, so a DataFrame built from many
records has one column per feature. Feature extraction is wrapped in defensive
handling: a noisy or pathological signal that breaks neurokit2's delineation
yields NaNs for the affected features rather than crashing the whole run.

Run standalone:  python -m src.ecg.features
"""

from __future__ import annotations

import warnings

import neurokit2 as nk
import numpy as np

from src.ecg.loader import PTBXLLoader, ECGRecord

# 12-lead order as it appears in PTB-XL.
LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Index of Lead II, used for delineation.
LEAD_II = 1

# Physiologically plausible interval ranges, in seconds. Values outside these
# bounds are treated as delineation errors and recorded as NaN rather than as
# real measurements. Ranges are deliberately generous - they reject clear
# artifacts (e.g. a 0.30s QRS) while keeping genuine pathology.
INTERVAL_BOUNDS = {
    "qrs_duration": (0.04, 0.20),   # normal 0.08-0.12; wide for BBB
    "qt_interval": (0.24, 0.60),    # normal ~0.35-0.45; wide for long-QT
    "pr_interval": (0.08, 0.32),    # normal 0.12-0.20; wide for AV block
    "p_duration": (0.04, 0.16),     # normal ~0.08-0.11
}


def _safe(fn, default=np.nan):
    """Call fn(), returning default on any exception or empty result."""
    try:
        value = fn()
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return value
    except Exception:
        return default


def _per_lead_stats(signal: np.ndarray) -> dict:
    """Statistical descriptors for each of the 12 leads."""
    features = {}

    for i, name in enumerate(LEAD_NAMES):
        lead = signal[:, i]

        features[f"{name}_mean"] = float(np.mean(lead))
        features[f"{name}_std"] = float(np.std(lead))
        features[f"{name}_min"] = float(np.min(lead))
        features[f"{name}_max"] = float(np.max(lead))
        features[f"{name}_ptp"] = float(np.ptp(lead))            # peak-to-peak
        features[f"{name}_energy"] = float(np.sum(lead**2))
        features[f"{name}_rms"] = float(np.sqrt(np.mean(lead**2)))
        # Fraction of signal energy above the mean - a crude morphology cue.
        features[f"{name}_abs_mean"] = float(np.mean(np.abs(lead)))

    return features


def _rhythm_features(signal: np.ndarray, sampling_rate: int) -> dict:
    """Heart rate, HRV, and interval features from Lead II delineation."""
    lead_ii = signal[:, LEAD_II]
    features: dict[str, float] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Clean + find R-peaks.
        cleaned = _safe(lambda: nk.ecg_clean(lead_ii, sampling_rate=sampling_rate),
                        default=None)
        if cleaned is None:
            return _empty_rhythm_features()

        signals, info = _safe(
            lambda: nk.ecg_peaks(cleaned, sampling_rate=sampling_rate),
            default=(None, None),
        ) or (None, None)

        rpeaks = None
        if info is not None:
            rpeaks = info.get("ECG_R_Peaks")

        if rpeaks is None or len(rpeaks) < 3:
            return _empty_rhythm_features()

        # Heart rate from R-R intervals.
        rr = np.diff(rpeaks) / sampling_rate               # seconds
        features["heart_rate_mean"] = float(60.0 / np.mean(rr))
        features["heart_rate_std"] = float(np.std(60.0 / rr))
        features["rr_mean"] = float(np.mean(rr))
        features["rr_std"] = float(np.std(rr))
        features["rr_min"] = float(np.min(rr))
        features["rr_max"] = float(np.max(rr))

        # HRV time-domain metrics.
        hrv = _safe(lambda: nk.hrv_time(rpeaks, sampling_rate=sampling_rate),
                    default=None)
        if hrv is not None and len(hrv) > 0:
            for col in ("HRV_RMSSD", "HRV_SDNN", "HRV_pNN50"):
                if col in hrv.columns:
                    features[col.lower()] = float(hrv[col].iloc[0])

        # Wave delineation for interval durations. The "peak" method places
        # wave boundaries more conservatively than "dwt", which systematically
        # over-widens the QRS complex.
        try:
            _, waves = nk.ecg_delineate(
                cleaned, rpeaks, sampling_rate=sampling_rate, method="peak"
            )
            features.update(_interval_features(waves, rpeaks, sampling_rate))
        except Exception:
            features.update(_empty_interval_features())

    return features


def _interval_features(waves: dict, rpeaks, sampling_rate: int) -> dict:
    """QRS, QT, PR, P-wave interval estimates from delineated landmarks.

    Uses the landmarks the "peak" delineation method reliably provides:
    Q-peak, S-peak, T-offset, P-onset, P-peak. The "peak" method does not
    populate R_Onsets/R_Offsets, so intervals are measured peak-to-peak /
    peak-to-boundary, validated against simulated ECG with known intervals:

        QRS  = Q-peak    -> S-peak      (~0.09-0.11s for normal)
        QT   = Q-peak    -> T-offset    (~0.38-0.44s for normal)
        PR   = P-onset   -> Q-peak      (~0.12-0.20s for normal)
        Pdur = P-onset   -> P-offset    (~0.08-0.11s for normal)
    """
    features = {}

    def _median_duration(starts, ends):
        starts = np.asarray(starts, dtype=float)
        ends = np.asarray(ends, dtype=float)
        n = min(len(starts), len(ends))
        if n == 0:
            return np.nan
        durations = (ends[:n] - starts[:n]) / sampling_rate
        durations = durations[np.isfinite(durations) & (durations > 0)]
        return float(np.median(durations)) if len(durations) else np.nan

    raw = {
        "qrs_duration": _median_duration(
            waves.get("ECG_Q_Peaks", []), waves.get("ECG_S_Peaks", [])
        ),
        "qt_interval": _median_duration(
            waves.get("ECG_Q_Peaks", []), waves.get("ECG_T_Offsets", [])
        ),
        "pr_interval": _median_duration(
            waves.get("ECG_P_Onsets", []), waves.get("ECG_Q_Peaks", [])
        ),
        "p_duration": _median_duration(
            waves.get("ECG_P_Onsets", []), waves.get("ECG_P_Offsets", [])
        ),
    }

    # Reject physiologically implausible values as delineation errors.
    for name, value in raw.items():
        low, high = INTERVAL_BOUNDS[name]
        if np.isfinite(value) and low <= value <= high:
            features[name] = value
        else:
            features[name] = np.nan

    return features


def _empty_interval_features() -> dict:
    return {k: np.nan for k in
            ("qrs_duration", "qt_interval", "pr_interval", "p_duration")}


def _empty_rhythm_features() -> dict:
    base = {k: np.nan for k in (
        "heart_rate_mean", "heart_rate_std", "rr_mean", "rr_std",
        "rr_min", "rr_max", "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
    )}
    base.update(_empty_interval_features())
    return base


def extract_features(record: ECGRecord) -> dict:
    """Full feature dict for one ECG record."""
    features = {"ecg_id": record.ecg_id, "age": record.age, "sex": record.sex}
    features.update(_rhythm_features(record.signal, record.sampling_rate))
    features.update(_per_lead_stats(record.signal))
    return features


def _smoke_test() -> None:
    loader = PTBXLLoader()

    print("Extracting features for 3 sample records...\n")
    for ecg_id in (1, 2, 3):
        record = loader.load_record(ecg_id)
        feats = extract_features(record)

        n_nan = sum(1 for v in feats.values()
                    if isinstance(v, float) and np.isnan(v))
        print(f"ecg_id={ecg_id}: {len(feats)} features, {n_nan} NaN, "
              f"labels={record.labels or ['(none)']}")
        print(f"  heart_rate_mean : {feats.get('heart_rate_mean')}")
        print(f"  qrs_duration    : {feats.get('qrs_duration')}")
        print(f"  qt_interval     : {feats.get('qt_interval')}")
        print(f"  II_energy       : {feats.get('II_energy'):.2f}")
        print()

    print(f"Total feature count per record: {len(feats)}")


if __name__ == "__main__":
    _smoke_test()