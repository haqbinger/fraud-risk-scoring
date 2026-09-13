"""
Fits the Platt calibrator and finds the operational decision threshold,
then saves both plus feature metadata needed to serve live predictions
(category maps, medians, feature column order) to data/processed/.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from calibration.calibrate import load_winning_model
from fraud.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    MAX_REVIEW_RATE,
)
from fraud.data import get_engine, load_data
from fraud.evaluations.splits import temporal_split
from fraud.features import build_feature_matrix, fit_median_impute

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CALIBRATOR_PATH = DATA_DIR / "platt_calibrator.joblib"
METADATA_PATH = DATA_DIR / "model_metadata.json"


def find_operational_threshold(y_true, y_prob):
    """Sweep thresholds 0.01-0.99, minimize business cost subject to the
    review-rate cap. Falls back to the min-cost threshold overall if no
    candidate satisfies the cap."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    best = None
    best_within_cap = None
    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        review_rate = y_pred.sum() / n
        cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE

        candidate = (cost, float(threshold))
        if best is None or candidate < best:
            best = candidate
        if review_rate <= MAX_REVIEW_RATE:
            if best_within_cap is None or candidate < best_within_cap:
                best_within_cap = candidate

    chosen = best_within_cap if best_within_cap is not None else best
    return chosen[1]


def run():
    engine = get_engine()
    df = load_data(engine)
    train_df, val_df, test_df = temporal_split(df, dt_col="txn_ts")

    X_train, y_train, category_maps = build_feature_matrix(train_df)
    X_val, y_val, _ = build_feature_matrix(val_df, category_maps=category_maps)
    X_test, y_test, _ = build_feature_matrix(test_df, category_maps=category_maps)

    medians = X_train.median(numeric_only=True)
    X_train = fit_median_impute(X_train, X_train, medians)
    X_val = fit_median_impute(X_train, X_val, medians)
    X_test = fit_median_impute(X_train, X_test, medians)

    model = load_winning_model()

    val_raw = model.predict_proba(X_val)[:, 1]
    platt = LogisticRegression(max_iter=1000)
    platt.fit(val_raw.reshape(-1, 1), y_val)

    test_raw = model.predict_proba(X_test)[:, 1]
    test_calibrated = platt.predict_proba(test_raw.reshape(-1, 1))[:, 1]

    threshold = find_operational_threshold(y_test, test_calibrated)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(platt, CALIBRATOR_PATH)

    metadata = {
        "model_version": "xgboost-v1",
        "threshold": threshold,
        "calibration_method": "platt",
        "feature_columns": list(X_train.columns),
        "category_maps": category_maps,
        "medians": {k: float(v) for k, v in medians.items()},
        "cost_false_negative": COST_FALSE_NEGATIVE,
        "cost_false_positive": COST_FALSE_POSITIVE,
        "max_review_rate": MAX_REVIEW_RATE,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved calibrator to {CALIBRATOR_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Operational threshold: {threshold:.4f}")


if __name__ == "__main__":
    run()
