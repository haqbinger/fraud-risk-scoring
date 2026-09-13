import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from pathlib import Path
import json
import pandas as pd
import numpy as np
from fraud.data import get_engine, load_data
from fraud.features import build_feature_matrix, fit_median_impute
from fraud.evaluations.splits import temporal_split
from fraud.evaluations.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "processed" / "winning_model.json"
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_winning_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Winning model not found at: {MODEL_PATH}\n"
            "Train the model first or check the save path in the training pipeline."
        )
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


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

    # Raw probabilities
    uncalibrated_prob = model.predict_proba(X_test)[:, 1]
    uncalibrated_metrics = compute_metrics(y_test, uncalibrated_prob)

    # ----- Manual Platt calibration on validation set -----
    val_raw = model.predict_proba(X_val)[:, 1]
    platt_model = LogisticRegression(max_iter=1000)
    platt_model.fit(val_raw.reshape(-1, 1), y_val)

    test_raw = model.predict_proba(X_test)[:, 1]
    platt_prob = platt_model.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    platt_metrics = compute_metrics(y_test, platt_prob)

    # ----- Manual isotonic calibration on validation set -----
    iso_model = IsotonicRegression(out_of_bounds="clip")
    iso_model.fit(val_raw, y_val)
    isotonic_prob = iso_model.predict(test_raw)
    isotonic_metrics = compute_metrics(y_test, isotonic_prob)

    print("=" * 60)
    print("CALIBRATION COMPARISON (test, lower Brier = better calibrated)")
    print("=" * 60)
    print(f"{'Method':<15}{'Brier':>10}{'PR-AUC':>10}{'ROC-AUC':>10}")
    for name, m in [("Uncalibrated", uncalibrated_metrics),
                    ("Platt", platt_metrics),
                    ("Isotonic", isotonic_metrics)]:
        print(f"{name:<15}{m['brier_score']:>10.4f}{m['pr_auc']:>10.4f}{m['roc_auc']:>10.4f}")

    print(
        "\nPR-AUC/ROC-AUC should barely move across methods -- calibration "
        "reshapes probabilities, it doesn't change ranking."
    )

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    for name, prob in [("Uncalibrated", uncalibrated_prob),
                        ("Platt", platt_prob),
                        ("Isotonic", isotonic_prob)]:
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives (actual)")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "calibration_curves.png")
    print(f"\nSaved reliability diagram to {REPORTS_DIR / 'calibration_curves.png'}")

    return {
        "uncalibrated": (uncalibrated_prob, uncalibrated_metrics),
        "platt": (platt_prob, platt_metrics),
        "isotonic": (isotonic_prob, isotonic_metrics),
        "y_test": y_test,
    }


if __name__ == "__main__":
    run()