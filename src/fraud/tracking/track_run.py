import os
import subprocess

import mlflow
import mlflow.xgboost

from fraud.config import (
    RANDOM_STATE,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    MAX_REVIEW_RATE,
)
from fraud.data import get_engine, load_data
from fraud.evaluations.splits import temporal_split
from calibration.calibrate import run as run_calibration, load_winning_model
from calibration.threshold_optimization import run as run_threshold_optimization

FIGURES_DIR = "src/reports/figures"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run():
    mlflow.set_experiment("fraud-risk-scoring")

    with mlflow.start_run(run_name="xgboost-m3-m4-tracked"):

        engine = get_engine()
        df = load_data(engine)
        train_df, val_df, test_df = temporal_split(df, dt_col="txn_ts")
        mlflow.log_param("n_rows_total", len(df))
        mlflow.log_param("fraud_rate", round(df["is_fraud"].mean(), 6))
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_val", len(val_df))
        mlflow.log_param("n_test", len(test_df))
        mlflow.log_param(
            "train_dt_range",
            f"{train_df['txn_ts'].min()} to {train_df['txn_ts'].max()}",
        )

        model = load_winning_model()
        xgb_params = model.get_params()
        print("Loaded model params (verify these aren't silently-reset defaults):")
        print(xgb_params)
        for k in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "scale_pos_weight",
            "tree_method",
        ]:
            if xgb_params.get(k) is not None:
                mlflow.log_param(k, xgb_params[k])
        mlflow.log_param("random_state", RANDOM_STATE)

        cal_results = run_calibration()
        for method in ["uncalibrated", "platt", "isotonic"]:
            m = cal_results[method][1]
            mlflow.log_metric(f"{method}_brier", m["brier_score"])
            mlflow.log_metric(f"{method}_pr_auc", m["pr_auc"])
            mlflow.log_metric(f"{method}_roc_auc", m["roc_auc"])

        mlflow.log_param("cost_false_negative", COST_FALSE_NEGATIVE)
        mlflow.log_param("cost_false_positive", COST_FALSE_POSITIVE)
        mlflow.log_param("max_review_rate", MAX_REVIEW_RATE)

        best_feasible, sensitivity_df = run_threshold_optimization()
        mlflow.log_metric("optimal_threshold", best_feasible["threshold"])
        mlflow.log_metric("optimal_total_cost", best_feasible["total_cost"])
        mlflow.log_metric("optimal_fn", best_feasible["fn"])
        mlflow.log_metric("optimal_fp", best_feasible["fp"])
        mlflow.log_metric("optimal_review_rate", best_feasible["review_rate"])

        mlflow.set_tag("git_commit", get_git_commit())

        for fname in [
            "calibration_curves.png",
            "threshold_cost_curve.png",
            "shap_global_summary.png",
            "shap_local_true_positive.png",
            "shap_local_false_positive.png",
            "shap_local_borderline.png",
        ]:
            fpath = os.path.join(FIGURES_DIR, fname)
            if os.path.exists(fpath):
                mlflow.log_artifact(fpath, artifact_path="figures")
            else:
                print(f"  [skip] {fpath} not found")

        if os.path.exists("data/processed/threshold_sensitivity.csv"):
            mlflow.log_artifact("data/processed/threshold_sensitivity.csv")

        try:
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name="fraud-xgboost",
            )
        except TypeError:
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name="fraud-xgboost",
                input_example=None,
            )

        print("\nRun logged.")
        print("View with: mlflow ui   (then open http://localhost:5000)")
        print("Model registered as 'fraud-xgboost' in the local MLflow registry.")


if __name__ == "__main__":
    run()