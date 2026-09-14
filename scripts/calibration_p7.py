"""
P7: Calibration + cost-sensitive threshold optimization on the final
LightGBM model (P5 winner), then ONE honest evaluation on the test set.

All calibration and threshold work happens on val only. The test set is
loaded up front by temporal_split() but is not touched until the final
evaluation block at the end of main() -- clearly marked below.
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, "scripts")
from feature_selection_p2 import CAT_COLS  # noqa: E402
from model_comparison_p5 import train_lgbm_with_params  # noqa: E402
from temporal_cv_p3 import load_selected_features  # noqa: E402
from tuning_p4 import build_xy  # noqa: E402

from fraud.calibrated_pipeline import CalibratedLGBMPipeline  # noqa: E402
from fraud.config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, MAX_REVIEW_RATE, RANDOM_STATE  # noqa: E402
from fraud.data import get_engine, load_data  # noqa: E402
from fraud.evaluations.metrics import compute_metrics  # noqa: E402
from fraud.evaluations.splits import temporal_split  # noqa: E402
from calibration.cost_model import expected_cost, sweep_thresholds  # noqa: E402

REPORT_DIR = "reports/calibration"
MODELS_DIR = "models"
LGBM_BEST_PARAMS_PATH = "reports/model_comparison/lgbm_best_params.json"
BRIER_TIE_THRESHOLD = 0.002
N_CV_FOLDS = 5
THRESH_MIN, THRESH_MAX, THRESH_STEP = 0.01, 0.99, 0.001

V1_TEST_PR_AUC = 0.2189
V1_TEST_BRIER = 0.0306
V1_THRESHOLD = 0.141
V1_COST_SAVINGS = 1_532_200


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    selected_features = load_selected_features()
    cat_cols = [c for c in CAT_COLS if c in selected_features]
    print(f"Loaded {len(selected_features)} selected features ({len(cat_cols)} categorical).")

    with open(LGBM_BEST_PARAMS_PATH, encoding="utf-8") as fh:
        lgbm_best_params = json.load(fh)

    engine = get_engine()
    df = load_data(engine)
    train_full, val_full, test_full = temporal_split(df, dt_col="txn_ts")
    # test_full is not referenced again until the "FINAL TEST SET EVALUATION"
    # block near the end of this function.

    categories_map = {c: sorted(train_full[c].dropna().unique().tolist()) for c in cat_cols}
    X_train, y_train = build_xy(train_full, selected_features, cat_cols, categories_map)
    X_val, y_val = build_xy(val_full, selected_features, cat_cols, categories_map)

    print("Training LightGBM (P5 best params) on full train split ...")
    model = train_lgbm_with_params(X_train, y_train, lgbm_best_params)
    val_raw_prob = model.predict_proba(X_val)[:, 1]
    y_val = np.asarray(y_val)

    # ------------------------------------------------------------------
    # 1. Calibration
    # ------------------------------------------------------------------
    print("\nCalibrating (5-fold cross_val_predict on val, to avoid overfitting the calibrator to val) ...")
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    platt_cv_prob = cross_val_predict(
        LogisticRegression(max_iter=1000), val_raw_prob.reshape(-1, 1), y_val,
        cv=cv, method="predict_proba",
    )[:, 1]

    iso_cv_prob = cross_val_predict(
        IsotonicRegression(out_of_bounds="clip"), val_raw_prob, y_val,
        cv=cv, method="predict",
    )

    uncal_brier = brier_score_loss(y_val, val_raw_prob)
    platt_brier = brier_score_loss(y_val, platt_cv_prob)
    iso_brier = brier_score_loss(y_val, iso_cv_prob)

    print(f"Brier (out-of-fold on val): uncalibrated={uncal_brier:.4f}  platt={platt_brier:.4f}  "
          f"isotonic={iso_brier:.4f}")

    if abs(platt_brier - iso_brier) <= BRIER_TIE_THRESHOLD:
        selected_calibrator_type = "platt"
        tie_note = f"tied within {BRIER_TIE_THRESHOLD} Brier -- Platt preferred (V1 ADR 0005 reasoning)"
    elif platt_brier < iso_brier:
        selected_calibrator_type = "platt"
        tie_note = "Platt has strictly lower Brier"
    else:
        selected_calibrator_type = "isotonic"
        tie_note = "Isotonic has strictly lower Brier"
    print(f"Selected calibrator: {selected_calibrator_type} ({tie_note})")

    # Final calibrators fit on all of val -- these are what get deployed and
    # applied to test. The cross_val_predict scores above are only for the
    # honest Brier comparison.
    platt_final = LogisticRegression(max_iter=1000).fit(val_raw_prob.reshape(-1, 1), y_val)
    iso_final = IsotonicRegression(out_of_bounds="clip").fit(val_raw_prob, y_val)
    selected_calibrator = platt_final if selected_calibrator_type == "platt" else iso_final

    if selected_calibrator_type == "platt":
        calibrated_val_prob = platt_final.predict_proba(val_raw_prob.reshape(-1, 1))[:, 1]
    else:
        calibrated_val_prob = iso_final.predict(val_raw_prob)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for name, probs in [("Uncalibrated", val_raw_prob), ("Platt (CV)", platt_cv_prob), ("Isotonic (CV)", iso_cv_prob)]:
        prob_true, prob_pred = calibration_curve(y_val, probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram (val, 10 bins)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{REPORT_DIR}/reliability_diagram.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Threshold optimization (on calibrated val probabilities)
    # ------------------------------------------------------------------
    n_steps = round((THRESH_MAX - THRESH_MIN) / THRESH_STEP) + 1
    sweep_df = sweep_thresholds(
        y_val, calibrated_val_prob, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE,
        n_steps=n_steps, min_threshold=THRESH_MIN, max_threshold=THRESH_MAX,
    )
    sweep_df["review_rate"] = (sweep_df["tp"] + sweep_df["fp"]) / len(y_val)
    sweep_df["precision"] = sweep_df["tp"] / (sweep_df["tp"] + sweep_df["fp"]).replace(0, np.nan)
    sweep_df["recall"] = sweep_df["tp"] / (sweep_df["tp"] + sweep_df["fn"]).replace(0, np.nan)
    sweep_df[["precision", "recall"]] = sweep_df[["precision", "recall"]].fillna(0.0)

    feasible = sweep_df[sweep_df["review_rate"] <= MAX_REVIEW_RATE]
    if feasible.empty:
        raise ValueError("No feasible threshold under the review-rate cap.")
    best_feasible = feasible.loc[feasible["total_cost"].idxmin()]
    selected_threshold = float(best_feasible["threshold"])

    unconstrained_best = sweep_df.loc[sweep_df["total_cost"].idxmin()]

    naive = expected_cost(y_val, calibrated_val_prob, 0.5, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE)

    print(f"\nUnconstrained cost-optimal threshold: {unconstrained_best['threshold']:.3f} "
          f"(review_rate={unconstrained_best['review_rate']:.1%}, cost=${unconstrained_best['total_cost']:,.0f})")
    print(f"Operational threshold (review_rate <= {MAX_REVIEW_RATE:.0%}): {selected_threshold:.3f}  "
          f"review_rate={best_feasible['review_rate']:.1%}  precision={best_feasible['precision']:.4f}  "
          f"recall={best_feasible['recall']:.4f}  cost=${best_feasible['total_cost']:,.0f}")
    print(f"Naive threshold=0.5: cost=${naive['total_cost']:,.0f}  "
          f"(val savings vs naive: ${naive['total_cost'] - best_feasible['total_cost']:,.0f})")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sweep_df["threshold"], sweep_df["total_cost"], label="Expected cost")
    ax.axvline(unconstrained_best["threshold"], color="red", linestyle="--", label="Unconstrained optimum")
    ax.axvline(selected_threshold, color="blue", linestyle="-.", label=f"Operational optimum ({selected_threshold:.3f})")
    ax.axvline(0.5, color="gray", linestyle=":", label="Naive 0.5")
    ax.set_xlabel("Threshold (on calibrated probability)")
    ax.set_ylabel("Total expected cost ($)")
    ax.set_title(f"Cost-sensitive threshold sweep (val, {selected_calibrator_type}-calibrated)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{REPORT_DIR}/threshold_cost_curve.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. FINAL TEST SET EVALUATION -- one time, never again.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OPENING THE TEST SET -- final, one-time, honest evaluation")
    print("=" * 60)
    X_test, y_test = build_xy(test_full, selected_features, cat_cols, categories_map)
    y_test = np.asarray(y_test)
    test_raw_prob = model.predict_proba(X_test)[:, 1]
    if selected_calibrator_type == "platt":
        test_calibrated_prob = platt_final.predict_proba(test_raw_prob.reshape(-1, 1))[:, 1]
    else:
        test_calibrated_prob = iso_final.predict(test_raw_prob)

    test_metrics = compute_metrics(y_test, test_calibrated_prob, threshold=selected_threshold)
    test_naive = expected_cost(y_test, test_calibrated_prob, 0.5, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE)
    test_optimal = expected_cost(y_test, test_calibrated_prob, selected_threshold, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE)
    test_cost_savings = test_naive["total_cost"] - test_optimal["total_cost"]

    print(f"TEST PR-AUC: {test_metrics['pr_auc']:.4f}  ROC-AUC: {test_metrics['roc_auc']:.4f}  "
          f"Brier: {test_metrics['brier_score']:.4f}")
    print(f"TEST precision: {test_metrics['precision']:.4f}  recall: {test_metrics['recall']:.4f}  "
          f"F1: {test_metrics['f1']:.4f}")
    print(f"TEST cost @ threshold {selected_threshold:.3f}: ${test_optimal['total_cost']:,.0f}  "
          f"(naive @0.5: ${test_naive['total_cost']:,.0f}, savings: ${test_cost_savings:,.0f})")

    pipeline = CalibratedLGBMPipeline(
        model=model, calibrator_type=selected_calibrator_type, calibrator=selected_calibrator,
        selected_features=selected_features, cat_cols=cat_cols, categories_map=categories_map,
        threshold=selected_threshold,
    )
    joblib.dump(pipeline, f"{MODELS_DIR}/lgbm_calibrated.pkl")

    metadata = {
        "model_type": "LightGBM",
        "lgbm_params": lgbm_best_params,
        "calibrator_type": selected_calibrator_type,
        "calibrator_selection_reason": tie_note,
        "threshold": selected_threshold,
        "selected_features": selected_features,
        "categorical_features": cat_cols,
        "val_brier_uncalibrated": uncal_brier,
        "val_brier_platt": platt_brier,
        "val_brier_isotonic": iso_brier,
        "val_pr_auc": float(compute_metrics(y_val, calibrated_val_prob)["pr_auc"]),
        "test_pr_auc": test_metrics["pr_auc"],
        "test_brier": test_metrics["brier_score"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "cost_model": {
            "cost_false_negative": COST_FALSE_NEGATIVE,
            "cost_false_positive": COST_FALSE_POSITIVE,
            "max_review_rate": MAX_REVIEW_RATE,
        },
        "test_cost_savings_vs_naive_0_5": test_cost_savings,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(f"{MODELS_DIR}/model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    comparison_rows = [
        ("PR-AUC (test)", V1_TEST_PR_AUC, test_metrics["pr_auc"]),
        ("Brier (test)", V1_TEST_BRIER, test_metrics["brier_score"]),
        ("Threshold", V1_THRESHOLD, selected_threshold),
        ("Cost savings vs naive 0.5", V1_COST_SAVINGS, test_cost_savings),
    ]

    summary_lines = [
        "# P7 Calibration + Threshold Optimization Summary",
        "",
        "## Calibration",
        "",
        f"- Val Brier: uncalibrated={uncal_brier:.4f}, Platt (5-fold CV)={platt_brier:.4f}, "
        f"Isotonic (5-fold CV)={iso_brier:.4f}",
        f"- Selected calibrator: **{selected_calibrator_type}** ({tie_note})",
        "- See decisions/v2/0010-p7-calibration-method-and-operational-threshold.md for the full ADR",
        "",
        "## Threshold optimization",
        "",
        f"- Cost model: FN=${COST_FALSE_NEGATIVE:,.0f}, FP=${COST_FALSE_POSITIVE:,.0f}, "
        f"max review rate={MAX_REVIEW_RATE:.0%}",
        f"- Unconstrained cost-optimal threshold: {unconstrained_best['threshold']:.3f} "
        f"(review_rate={unconstrained_best['review_rate']:.1%})",
        f"- Operational threshold (review_rate <= {MAX_REVIEW_RATE:.0%}): **{selected_threshold:.3f}**",
        f"- Val: precision={best_feasible['precision']:.4f}, recall={best_feasible['recall']:.4f}, "
        f"review_rate={best_feasible['review_rate']:.1%}",
        f"- Val cost savings vs naive 0.5: ${naive['total_cost'] - best_feasible['total_cost']:,.0f}",
        "",
        "## Final test set results (one-time, honest)",
        "",
        f"- PR-AUC: {test_metrics['pr_auc']:.4f}",
        f"- ROC-AUC: {test_metrics['roc_auc']:.4f}",
        f"- Brier: {test_metrics['brier_score']:.4f}",
        f"- Precision: {test_metrics['precision']:.4f}",
        f"- Recall: {test_metrics['recall']:.4f}",
        f"- F1: {test_metrics['f1']:.4f}",
        f"- Confusion matrix: {test_metrics['confusion_matrix']}",
        f"- Cost @ threshold {selected_threshold:.3f}: ${test_optimal['total_cost']:,.0f}",
        f"- Cost @ naive 0.5: ${test_naive['total_cost']:,.0f}",
        f"- Cost savings vs naive 0.5: ${test_cost_savings:,.0f}",
        "",
        "## V1 vs V2 (test set)",
        "",
        "| Metric | V1 | V2 | Delta |",
        "|---|---|---|---|",
    ] + [
        f"| {name} | {v1:,.4f} | {v2:,.4f} | {v2 - v1:+,.4f} |" if abs(v1) < 100
        else f"| {name} | ${v1:,.0f} | ${v2:,.0f} | ${v2 - v1:+,.0f} |"
        for name, v1, v2 in comparison_rows
    ] + [
        "",
        "## Artifacts",
        "",
        f"- {REPORT_DIR}/threshold_cost_curve.png",
        f"- {REPORT_DIR}/reliability_diagram.png",
        f"- {MODELS_DIR}/lgbm_calibrated.pkl",
        f"- {MODELS_DIR}/model_metadata.json",
    ]
    with open(f"{REPORT_DIR}/p7_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    print("\n" + "=" * 60)
    print("V1 vs V2 (TEST SET)")
    print("=" * 60)
    print(f"{'Metric':<28}{'V1':>16}{'V2':>16}{'Delta':>16}")
    for name, v1, v2 in comparison_rows:
        if abs(v1) < 100:
            print(f"{name:<28}{v1:>16.4f}{v2:>16.4f}{v2 - v1:>+16.4f}")
        else:
            print(f"{name:<28}{'$' + format(v1, ',.0f'):>16}{'$' + format(v2, ',.0f'):>16}"
                  f"{'$' + format(v2 - v1, '+,.0f'):>16}")

    print(f"\nSaved: {REPORT_DIR}/threshold_cost_curve.png")
    print(f"Saved: {REPORT_DIR}/reliability_diagram.png")
    print(f"Saved: {REPORT_DIR}/p7_summary.md")
    print(f"Saved: {MODELS_DIR}/lgbm_calibrated.pkl")
    print(f"Saved: {MODELS_DIR}/model_metadata.json")


if __name__ == "__main__":
    main()
