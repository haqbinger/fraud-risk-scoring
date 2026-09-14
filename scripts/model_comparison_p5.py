"""
P5: Model comparison at full scale -- tuned XGBoost (P4's best_params.json)
vs. a freshly-tuned LightGBM, both on the P2-selected 182-feature set and
the exact P3/P4 rolling temporal CV folds (imported, not re-derived).
"""
import json
import os
import sys
import time

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
from feature_selection_p2 import CAT_COLS, to_categorical  # noqa: E402
from temporal_cv_p3 import (  # noqa: E402
    N_FOLDS, load_data_splits, load_selected_features, make_rolling_folds,
)
from tuning_p4 import build_xy, train_xgb_with_params  # noqa: E402

from fraud.config import RANDOM_STATE  # noqa: E402

REPORT_DIR = "reports/model_comparison"
XGB_BEST_PARAMS_PATH = "reports/tuning/best_params.json"
N_TRIALS = 50
N_PARALLEL_TRIALS = 10
P4_XGB_CV_MEAN = 0.6167
P4_XGB_VAL_PR_AUC = 0.6317
TIE_THRESHOLD = 0.005


def train_lgbm_with_params(X_train, y_train, params: dict) -> LGBMClassifier:
    model = LGBMClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
        # subsample (bagging_fraction) is a no-op in LightGBM unless a bagging
        # frequency is set -- this is mechanism, not a tuned hyperparameter.
        subsample_freq=1,
    )
    model.fit(X_train, y_train)
    return model


def suggest_lgbm_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 20, 80),
    }


def cv_eval(train_fn, params, prepared_folds, model_name, result_rows):
    fold_scores, fold_fit_times = [], []
    for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(prepared_folds):
        t0 = time.time()
        model = train_fn(X_train, y_train, params)
        fit_time_s = time.time() - t0
        prob = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, prob)
        fold_scores.append(pr_auc)
        fold_fit_times.append(fit_time_s)
        result_rows.append({
            "model": model_name, "fold": fold_idx + 1,
            "train_size": len(X_train), "val_size": len(X_val),
            "val_pr_auc": pr_auc, "fit_time_s": fit_time_s,
        })
    return fold_scores, fold_fit_times


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    selected_features = load_selected_features()
    cat_cols = [c for c in CAT_COLS if c in selected_features]
    print(f"Loaded {len(selected_features)} selected features ({len(cat_cols)} categorical).")

    with open(XGB_BEST_PARAMS_PATH, encoding="utf-8") as fh:
        xgb_best_params = json.load(fh)
    print(f"Loaded XGBoost best params from {XGB_BEST_PARAMS_PATH}: {xgb_best_params}")

    train_full, val_full, cv_pool = load_data_splits()
    folds, block_size = make_rolling_folds(cv_pool, N_FOLDS)
    print(f"Replicated P3/P4's {N_FOLDS} rolling folds (~{block_size} rows/block) -- not re-derived.\n")

    prepared_folds = []
    for train_df, val_df in folds:
        categories_map = {c: sorted(train_df[c].dropna().unique().tolist()) for c in cat_cols}
        X_train, y_train = build_xy(train_df, selected_features, cat_cols, categories_map)
        X_val, y_val = build_xy(val_df, selected_features, cat_cols, categories_map)
        prepared_folds.append((X_train, y_train, X_val, y_val))

    result_rows = []

    print("=" * 60)
    print("Step 1: Tuned XGBoost CV sanity check (P4 best_params.json)")
    print("=" * 60)
    xgb_fold_scores, xgb_fit_times = cv_eval(
        train_xgb_with_params, xgb_best_params, prepared_folds, "XGBoost", result_rows
    )
    xgb_cv_mean = sum(xgb_fold_scores) / len(xgb_fold_scores)
    xgb_cv_std = pd.Series(xgb_fold_scores).std()
    xgb_avg_fit_time = sum(xgb_fit_times) / len(xgb_fit_times)
    print(f"XGBoost CV mean PR-AUC: {xgb_cv_mean:.4f} (P4 recorded: {P4_XGB_CV_MEAN}, "
          f"delta: {xgb_cv_mean - P4_XGB_CV_MEAN:+.4f})")
    for i, (s, t) in enumerate(zip(xgb_fold_scores, xgb_fit_times), 1):
        print(f"  fold {i}: val PR-AUC={s:.4f}  fit_time={t:.1f}s")

    print("\n" + "=" * 60)
    print(f"Step 2: LightGBM Optuna tuning ({N_TRIALS} trials, {N_PARALLEL_TRIALS} parallel)")
    print("=" * 60 + "\n")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params(trial)
        scores = []
        for X_train, y_train, X_val, y_val in prepared_folds:
            model = train_lgbm_with_params(X_train, y_train, params)
            prob = model.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, prob))
        return sum(scores) / len(scores)

    def report_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        print(f"Trial {trial.number:>3}: mean CV PR-AUC = {trial.value:.4f}  "
              f"(best so far: {study.best_value:.4f})")

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_PARALLEL_TRIALS, callbacks=[report_callback])
    lgbm_tuning_time_s = time.time() - t0

    lgbm_best_params = study.best_params
    print(f"\nLightGBM tuning finished in {lgbm_tuning_time_s:.1f}s ({lgbm_tuning_time_s / 60:.1f} min).")
    print(f"Best trial: #{study.best_trial.number}  mean CV PR-AUC = {study.best_value:.4f}")
    print(f"Best params: {lgbm_best_params}")

    with open(f"{REPORT_DIR}/lgbm_best_params.json", "w", encoding="utf-8") as fh:
        json.dump(lgbm_best_params, fh, indent=2)

    # Re-run CV with best params to get per-fold fit times/scores for the report
    # (the Optuna trial's own fold runs weren't retained individually).
    lgbm_fold_scores, lgbm_fit_times = cv_eval(
        train_lgbm_with_params, lgbm_best_params, prepared_folds, "LightGBM", result_rows
    )
    lgbm_cv_mean = sum(lgbm_fold_scores) / len(lgbm_fold_scores)
    lgbm_cv_std = pd.Series(lgbm_fold_scores).std()
    lgbm_avg_fit_time = sum(lgbm_fit_times) / len(lgbm_fit_times)

    print("\nRetraining best LightGBM on full train split, evaluating on val (single split) ...")
    categories_map = {c: sorted(train_full[c].dropna().unique().tolist()) for c in cat_cols}
    X_train_full, y_train_full = build_xy(train_full, selected_features, cat_cols, categories_map)
    X_val_full, y_val_full = build_xy(val_full, selected_features, cat_cols, categories_map)

    t0 = time.time()
    lgbm_final = train_lgbm_with_params(X_train_full, y_train_full, lgbm_best_params)
    lgbm_final_fit_time_s = time.time() - t0
    lgbm_val_pr_auc = average_precision_score(y_val_full, lgbm_final.predict_proba(X_val_full)[:, 1])

    cv_delta = lgbm_cv_mean - xgb_cv_mean
    if abs(cv_delta) <= TIE_THRESHOLD:
        winner = "LightGBM (tie within 0.005, preferred for faster inference)"
    elif cv_delta > 0:
        winner = "LightGBM (higher CV mean PR-AUC)"
    else:
        winner = "XGBoost (higher CV mean PR-AUC)"

    comparison_rows = [
        {
            "model": "Tuned XGBoost", "cv_mean_pr_auc": xgb_cv_mean, "cv_std_pr_auc": xgb_cv_std,
            "val_pr_auc": P4_XGB_VAL_PR_AUC, "avg_fit_time_s": xgb_avg_fit_time,
        },
        {
            "model": "Tuned LightGBM", "cv_mean_pr_auc": lgbm_cv_mean, "cv_std_pr_auc": lgbm_cv_std,
            "val_pr_auc": lgbm_val_pr_auc, "avg_fit_time_s": lgbm_avg_fit_time,
        },
    ]

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(f"{REPORT_DIR}/p5_results.csv", index=False)

    comparison_table_md = (
        "| Model | CV mean PR-AUC | CV std | Val PR-AUC | Avg fit time per fold |\n"
        "|---|---|---|---|---|\n"
        f"| Tuned XGBoost | {xgb_cv_mean:.4f} | {xgb_cv_std:.4f} | {P4_XGB_VAL_PR_AUC:.4f} | {xgb_avg_fit_time:.1f}s |\n"
        f"| Tuned LightGBM | {lgbm_cv_mean:.4f} | {lgbm_cv_std:.4f} | {lgbm_val_pr_auc:.4f} | {lgbm_avg_fit_time:.1f}s |\n"
    )

    summary_lines = [
        "# P5 Model Comparison Summary",
        "",
        "## Comparison table",
        "",
        comparison_table_md,
        f"**Winner: {winner}**",
        "",
        "## Details",
        "",
        f"- XGBoost CV mean sanity check vs P4: {xgb_cv_mean:.4f} vs {P4_XGB_CV_MEAN} "
        f"(delta {xgb_cv_mean - P4_XGB_CV_MEAN:+.4f})",
        f"- LightGBM tuning: {N_TRIALS} trials, {N_PARALLEL_TRIALS} parallel, "
        f"{lgbm_tuning_time_s:.1f}s ({lgbm_tuning_time_s / 60:.1f} min)",
        f"- LightGBM final model fit time (full train split): {lgbm_final_fit_time_s:.1f}s",
        "",
        "## LightGBM best params",
        "",
        "```json",
        json.dumps(lgbm_best_params, indent=2),
        "```",
    ]
    with open(f"{REPORT_DIR}/p5_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    print("\n" + "=" * 60)
    print("P5 FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<16} {'CV mean PR-AUC':>15} {'CV std':>8} {'Val PR-AUC':>11} {'Avg fit time/fold':>18}")
    for row in comparison_rows:
        print(f"{row['model']:<16} {row['cv_mean_pr_auc']:>15.4f} {row['cv_std_pr_auc']:>8.4f} "
              f"{row['val_pr_auc']:>11.4f} {row['avg_fit_time_s']:>17.1f}s")
    print(f"\nWinner: {winner}")
    print(f"\nSaved: {REPORT_DIR}/p5_results.csv")
    print(f"Saved: {REPORT_DIR}/lgbm_best_params.json")
    print(f"Saved: {REPORT_DIR}/p5_summary.md")


if __name__ == "__main__":
    main()
