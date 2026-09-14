"""
P4: Optuna hyperparameter tuning on the P2-selected 182-feature set,
optimized against the exact P3 rolling-CV fold boundaries (not re-derived --
imported from temporal_cv_p3 so the folds are identical row-for-row).

Each trial trains one XGBoost config across all 5 folds and is scored on
the mean val PR-AUC. n_jobs=1 on XGBoost itself; Optuna parallelizes trials
across threads instead (XGBoost's fit releases the GIL, so this genuinely
uses multiple cores without oversubscribing them).
"""
import json
import os
import sys
import time

import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, "scripts")
from feature_selection_p2 import CAT_COLS, to_categorical  # noqa: E402
from temporal_cv_p3 import (  # noqa: E402
    N_FOLDS, load_data_splits, load_selected_features, make_rolling_folds,
)

from fraud.config import RANDOM_STATE  # noqa: E402

REPORT_DIR = "reports/tuning"
N_TRIALS = 50
N_PARALLEL_TRIALS = 10
DEFAULT_CV_MEAN_PR_AUC = 0.5642  # P3 result, default hyperparams, same 182 features


def build_xy(df: pd.DataFrame, feature_cols, cat_cols, categories_map: dict):
    X = to_categorical(df[feature_cols], cat_cols, categories_map)
    y = df["is_fraud"].values
    return X, y


def train_xgb_with_params(X_train, y_train, params: dict) -> XGBClassifier:
    model = XGBClassifier(
        **params,
        random_state=RANDOM_STATE,
        enable_categorical=True,
        tree_method="hist",
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 20, 60),
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    selected_features = load_selected_features()
    cat_cols = [c for c in CAT_COLS if c in selected_features]
    print(f"Loaded {len(selected_features)} selected features ({len(cat_cols)} categorical).")

    train_full, val_full, cv_pool = load_data_splits()
    folds, block_size = make_rolling_folds(cv_pool, N_FOLDS)
    print(f"Replicated P3's {N_FOLDS} rolling folds (~{block_size} rows/block) -- not re-derived.\n")

    # Data prep is independent of hyperparams -- build each fold's X/y once,
    # shared read-only across all trials/threads.
    prepared_folds = []
    for train_df, val_df in folds:
        categories_map = {c: sorted(train_df[c].dropna().unique().tolist()) for c in cat_cols}
        X_train, y_train = build_xy(train_df, selected_features, cat_cols, categories_map)
        X_val, y_val = build_xy(val_df, selected_features, cat_cols, categories_map)
        prepared_folds.append((X_train, y_train, X_val, y_val))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        fold_scores = []
        for X_train, y_train, X_val, y_val in prepared_folds:
            model = train_xgb_with_params(X_train, y_train, params)
            prob = model.predict_proba(X_val)[:, 1]
            fold_scores.append(average_precision_score(y_val, prob))
        mean_score = sum(fold_scores) / len(fold_scores)
        trial.set_user_attr("fold_scores", fold_scores)
        return mean_score

    def report_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        print(f"Trial {trial.number:>3}: mean CV PR-AUC = {trial.value:.4f}  "
              f"(best so far: {study.best_value:.4f})")

    print(f"Starting Optuna study: {N_TRIALS} trials, {N_PARALLEL_TRIALS} parallel, "
          f"objective = mean PR-AUC across {N_FOLDS} CV folds ...\n")
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_PARALLEL_TRIALS, callbacks=[report_callback])
    total_tuning_time_s = time.time() - t0

    best_params = study.best_params
    best_cv_pr_auc = study.best_value

    print(f"\nTuning finished in {total_tuning_time_s:.1f}s ({total_tuning_time_s / 60:.1f} min).")
    print(f"Best trial: #{study.best_trial.number}  mean CV PR-AUC = {best_cv_pr_auc:.4f}")
    print(f"Best params: {best_params}")

    print("\nRetraining best config on the full train split, evaluating on val (single split, same as P2) ...")
    categories_map = {c: sorted(train_full[c].dropna().unique().tolist()) for c in cat_cols}
    X_train_full, y_train_full = build_xy(train_full, selected_features, cat_cols, categories_map)
    X_val_full, y_val_full = build_xy(val_full, selected_features, cat_cols, categories_map)

    t0 = time.time()
    final_model = train_xgb_with_params(X_train_full, y_train_full, best_params)
    final_fit_time_s = time.time() - t0

    val_prob = final_model.predict_proba(X_val_full)[:, 1]
    val_pr_auc = average_precision_score(y_val_full, val_prob)

    trial_rows = [
        {"trial": t.number, **t.params, "mean_cv_pr_auc": t.value}
        for t in study.trials if t.value is not None
    ]
    trials_df = pd.DataFrame(trial_rows).sort_values("trial").reset_index(drop=True)
    trials_df.to_csv(f"{REPORT_DIR}/optuna_trials.csv", index=False)

    with open(f"{REPORT_DIR}/best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best_params, fh, indent=2)

    delta = best_cv_pr_auc - DEFAULT_CV_MEAN_PR_AUC
    summary_lines = [
        "# P4 Hyperparameter Tuning Summary",
        "",
        f"- Trials: {N_TRIALS} ({N_PARALLEL_TRIALS} parallel)",
        f"- Total tuning time: {total_tuning_time_s:.1f}s ({total_tuning_time_s / 60:.1f} min)",
        f"- Default (P3, default hyperparams) CV mean PR-AUC: {DEFAULT_CV_MEAN_PR_AUC:.4f}",
        f"- Tuned CV mean PR-AUC: {best_cv_pr_auc:.4f}",
        f"- Delta vs default: {delta:+.4f}",
        f"- Val PR-AUC (best config, single split, same as P2): {val_pr_auc:.4f}",
        f"- Final model fit time (full train split): {final_fit_time_s:.1f}s",
        "",
        "## Best params",
        "",
        "```json",
        json.dumps(best_params, indent=2),
        "```",
    ]
    with open(f"{REPORT_DIR}/tuning_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    print("\n" + "=" * 60)
    print("P4 SUMMARY")
    print("=" * 60)
    print(f"Default CV mean PR-AUC : {DEFAULT_CV_MEAN_PR_AUC:.4f}")
    print(f"Tuned CV mean PR-AUC   : {best_cv_pr_auc:.4f}  (delta: {delta:+.4f})")
    print(f"Val PR-AUC (best config): {val_pr_auc:.4f}")
    print(f"\nSaved: {REPORT_DIR}/optuna_trials.csv")
    print(f"Saved: {REPORT_DIR}/best_params.json")
    print(f"Saved: {REPORT_DIR}/tuning_summary.md")


if __name__ == "__main__":
    main()
