"""
P3: Rolling temporal cross-validation on the P2-selected 182-feature set.

6 equal-sized chronological blocks are cut from train+val (never test).
Each of the 5 folds uses one block as train and the immediately following
block as val (rolling/sliding window, not expanding) -- train and val are
the same size in every fold, val is always strictly after train, and there
is no gap between them since the data is already contiguous and sorted by
txn_ts.

Test set is loaded by temporal_split() but never touched.
"""
import os
import sys
import time

import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
from feature_selection_p2 import CAT_COLS, to_categorical, train_xgb  # noqa: E402

from fraud.data import get_engine, load_data  # noqa: E402
from fraud.evaluations.splits import temporal_split  # noqa: E402

REPORT_DIR = "reports/cv"
SELECTED_FEATURES_PATH = "reports/feature_selection/selected_features.txt"
N_FOLDS = 5
P2_SINGLE_SPLIT_PR_AUC = 0.5888
MEAN_TOLERANCE = 0.03
STD_TOLERANCE = 0.05


def build_xy(df: pd.DataFrame, feature_cols, cat_cols, categories_map: dict):
    X = to_categorical(df[feature_cols], cat_cols, categories_map)
    y = df["is_fraud"].values
    return X, y


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(SELECTED_FEATURES_PATH, encoding="utf-8") as fh:
        selected_features = [line.strip() for line in fh if line.strip()]
    cat_cols = [c for c in CAT_COLS if c in selected_features]
    print(f"Loaded {len(selected_features)} selected features from {SELECTED_FEATURES_PATH} "
          f"({len(cat_cols)} categorical).")

    engine = get_engine()
    df = load_data(engine)
    train_full, val_full, _test_full = temporal_split(df, dt_col="txn_ts")
    del _test_full  # P3 is CV over train+val only -- test set is loaded but never touched

    cv_pool = pd.concat([train_full, val_full], ignore_index=True)
    n = len(cv_pool)
    n_blocks = N_FOLDS + 1
    block_size = n // n_blocks
    block_bounds = [i * block_size for i in range(n_blocks)] + [n]  # last block absorbs remainder

    print(f"\nCV pool (train+val, test excluded): {n} rows -> {n_blocks} equal blocks of ~{block_size} rows")
    print(f"Rolling CV: {N_FOLDS} folds, each fold's val block immediately follows its train block.\n")

    fold_results = []
    for fold in range(N_FOLDS):
        train_start, train_end = block_bounds[fold], block_bounds[fold + 1]
        val_start, val_end = block_bounds[fold + 1], block_bounds[fold + 2]

        train_df = cv_pool.iloc[train_start:train_end]
        val_df = cv_pool.iloc[val_start:val_end]

        categories_map = {c: sorted(train_df[c].dropna().unique().tolist()) for c in cat_cols}
        X_train, y_train = build_xy(train_df, selected_features, cat_cols, categories_map)
        X_val, y_val = build_xy(val_df, selected_features, cat_cols, categories_map)

        t0 = time.time()
        model = train_xgb(X_train, y_train)
        fit_time_s = time.time() - t0

        val_prob = model.predict_proba(X_val)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_prob)

        result = {
            "fold": fold + 1,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "val_pr_auc": val_pr_auc,
            "fit_time_s": fit_time_s,
        }
        fold_results.append(result)
        print(f"Fold {fold + 1}/{N_FOLDS}: train={len(train_df):>7} | val={len(val_df):>7} | "
              f"val PR-AUC={val_pr_auc:.4f} | fit_time={fit_time_s:.1f}s | "
              f"train window [{train_df['txn_ts'].min()} .. {train_df['txn_ts'].max()}] | "
              f"val window [{val_df['txn_ts'].min()} .. {val_df['txn_ts'].max()}]")

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f"{REPORT_DIR}/fold_results.csv", index=False)

    mean_pr_auc = results_df["val_pr_auc"].mean()
    std_pr_auc = results_df["val_pr_auc"].std()
    min_pr_auc = results_df["val_pr_auc"].min()
    max_pr_auc = results_df["val_pr_auc"].max()

    delta = abs(mean_pr_auc - P2_SINGLE_SPLIT_PR_AUC)
    reliable = delta <= MEAN_TOLERANCE and std_pr_auc < STD_TOLERANCE
    verdict = (
        f"Single-split estimate (0.5888) was RELIABLE -- CV mean is within {MEAN_TOLERANCE} "
        f"({delta:.4f}) and std ({std_pr_auc:.4f}) is below {STD_TOLERANCE}."
        if reliable else
        f"Single-split estimate (0.5888) was NOT reliable -- "
        f"{'mean delta ' + format(delta, '.4f') + ' exceeds ' + str(MEAN_TOLERANCE) if delta > MEAN_TOLERANCE else ''}"
        f"{' and ' if delta > MEAN_TOLERANCE and std_pr_auc >= STD_TOLERANCE else ''}"
        f"{'std ' + format(std_pr_auc, '.4f') + ' exceeds ' + str(STD_TOLERANCE) if std_pr_auc >= STD_TOLERANCE else ''}."
    )

    summary_lines = [
        "# P3 Temporal Cross-Validation Summary",
        "",
        f"- Folds: {N_FOLDS} (rolling window, train/val block size ~{block_size} rows each)",
        f"- Mean val PR-AUC: {mean_pr_auc:.4f}",
        f"- Std val PR-AUC: {std_pr_auc:.4f}",
        f"- Min val PR-AUC: {min_pr_auc:.4f}",
        f"- Max val PR-AUC: {max_pr_auc:.4f}",
        f"- P2 single-split val PR-AUC: {P2_SINGLE_SPLIT_PR_AUC}",
        f"- |mean - single-split|: {delta:.4f}",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Per-fold results",
        "",
        "| fold | train_size | val_size | val_pr_auc | fit_time_s |",
        "|---|---|---|---|---|",
    ] + [
        f"| {r.fold} | {r.train_size} | {r.val_size} | {r.val_pr_auc:.4f} | {r.fit_time_s:.1f} |"
        for r in results_df.itertuples()
    ]
    with open(f"{REPORT_DIR}/cv_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    print("\n" + "=" * 60)
    print("P3 SUMMARY")
    print("=" * 60)
    print(f"Mean val PR-AUC: {mean_pr_auc:.4f} +/- {std_pr_auc:.4f}  (min={min_pr_auc:.4f}, max={max_pr_auc:.4f})")
    print(f"P2 single-split val PR-AUC: {P2_SINGLE_SPLIT_PR_AUC}  (delta: {delta:.4f})")
    print(verdict)
    print(f"\nSaved: {REPORT_DIR}/fold_results.csv")
    print(f"Saved: {REPORT_DIR}/cv_summary.md")


if __name__ == "__main__":
    main()
