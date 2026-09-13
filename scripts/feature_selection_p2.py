"""
P2: Feature selection on the full P1 feature matrix.

Kitchen-sink XGBoost (all features, default hyperparams) as a ceiling check,
gain vs. permutation importance disagreement analysis, then drop near-zero /
harmful / redundant features and confirm val PR-AUC still holds.

Val only throughout -- the test set is loaded by temporal_split() but never
touched here.
"""
import os

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from fraud.config import RANDOM_STATE
from fraud.data import get_engine, load_data
from fraud.evaluations.splits import temporal_split

REPORT_DIR = "reports/feature_selection"

EXCLUDE_COLS = {"transaction_id", "is_fraud", "txn_ts", "card1_prev_txn_ts"}
CAT_COLS = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "product_cd", "p_emaildomain", "device_type",
]

GAIN_DROP_THRESHOLD = 0.0001
CORR_THRESHOLD_INITIAL = 0.95
CORR_THRESHOLD_RELAXED = 0.98
PR_AUC_TOLERANCE = 0.01
CORR_SAMPLE_SIZE = 100_000


def to_categorical(df: pd.DataFrame, cols, categories_map: dict) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.Categorical(out[c], categories=categories_map[c])
    return out


def build_xy(df: pd.DataFrame, feature_cols, categories_map: dict, cat_cols):
    X = to_categorical(df[feature_cols], cat_cols, categories_map)
    y = df["is_fraud"].values
    return X, y


def train_xgb(X_train, y_train) -> XGBClassifier:
    # enable_categorical + tree_method are mechanism, not tuning -- required
    # for XGBoost to accept the raw M1-M9/product_cd/p_emaildomain/device_type
    # categorical columns natively. All other hyperparams are left at default.
    model = XGBClassifier(
        random_state=RANDOM_STATE,
        enable_categorical=True,
        tree_method="hist",
        importance_type="gain",
    )
    model.fit(X_train, y_train)
    return model


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    engine = get_engine()
    df = load_data(engine)
    train_df, val_df, _test_df = temporal_split(df, dt_col="txn_ts")
    del _test_df  # P2 is val-only -- test set is loaded but never touched

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    categories_map = {c: sorted(train_df[c].dropna().unique().tolist()) for c in CAT_COLS}

    X_train, y_train = build_xy(train_df, feature_cols, categories_map, CAT_COLS)
    X_val, y_val = build_xy(val_df, feature_cols, categories_map, CAT_COLS)

    print(f"\nKitchen sink: {len(feature_cols)} features | train={len(X_train)} | val={len(X_val)}")

    print("Training kitchen-sink XGBoost (default hyperparams, all features) ...")
    model = train_xgb(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    kitchen_sink_pr_auc = average_precision_score(y_val, val_prob)
    print(f"Kitchen sink val PR-AUC (ceiling): {kitchen_sink_pr_auc:.4f}")

    gain_df = pd.DataFrame({
        "feature": feature_cols,
        "gain_importance": model.feature_importances_,
    }).sort_values("gain_importance", ascending=False).reset_index(drop=True)
    gain_df.to_csv(f"{REPORT_DIR}/gain_importance.csv", index=False)

    print("Computing permutation importance (n_repeats=10) on val set ...")
    perm_result = permutation_importance(
        model, X_val, y_val, n_repeats=10, random_state=RANDOM_STATE,
        scoring="average_precision",
    )
    perm_df = pd.DataFrame({
        "feature": X_val.columns,
        "perm_importance_mean": perm_result.importances_mean,
        "perm_importance_std": perm_result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
    perm_df.to_csv(f"{REPORT_DIR}/permutation_importance.csv", index=False)

    # ---------------------------------------------------------------------
    # Disagreement analysis
    # ---------------------------------------------------------------------
    gain_df["gain_rank"] = gain_df["gain_importance"].rank(ascending=False, method="min").astype(int)
    perm_df["perm_rank"] = perm_df["perm_importance_mean"].rank(ascending=False, method="min").astype(int)

    merged = gain_df.merge(perm_df, on="feature")
    merged["rank_gap"] = (merged["gain_rank"] - merged["perm_rank"]).abs()
    merged = merged.sort_values("rank_gap", ascending=False).reset_index(drop=True)
    merged.to_csv(f"{REPORT_DIR}/disagreement_analysis.csv", index=False)

    disagreements = merged[merged["rank_gap"] > 50]
    print(f"\n{len(disagreements)} features disagree by >50 rank positions between gain and permutation importance.")
    print("\nTop 20 disagreements (gain rank vs. permutation rank):")
    print(disagreements.head(20)[[
        "feature", "gain_rank", "perm_rank", "rank_gap", "gain_importance", "perm_importance_mean",
    ]].to_string(index=False))

    # ---------------------------------------------------------------------
    # Drop candidates: (A) hurts on permutation, (B) near-zero gain,
    # (C) redundant with a higher-ranked feature (Spearman > threshold)
    # ---------------------------------------------------------------------
    drop_reasons = {}
    for _, row in merged.iterrows():
        f = row["feature"]
        if row["perm_importance_mean"] < 0:
            drop_reasons[f] = "permutation_importance < 0"
        elif row["gain_importance"] < GAIN_DROP_THRESHOLD:
            drop_reasons[f] = f"gain_importance < {GAIN_DROP_THRESHOLD}"

    numeric_candidates = [f for f in feature_cols if f not in CAT_COLS and f not in drop_reasons]

    print(f"\nComputing Spearman correlation matrix on a {min(CORR_SAMPLE_SIZE, len(train_df))}-row "
          f"train-only sample of {len(numeric_candidates)} numeric candidate features ...")
    corr_sample = train_df[numeric_candidates].sample(
        n=min(CORR_SAMPLE_SIZE, len(train_df)), random_state=RANDOM_STATE
    )
    corr_matrix = corr_sample.corr(method="spearman").abs()

    def correlation_prune(threshold):
        # Best-ranked first (permutation rank primary, gain rank tiebreak) so
        # that when two features are redundant, the better-ranked one is kept.
        rank_order = (
            merged.set_index("feature").loc[numeric_candidates]
            .sort_values(["perm_rank", "gain_rank"]).index.tolist()
        )
        kept, pruned = [], {}
        for f in rank_order:
            redundant_with = next((k for k in kept if corr_matrix.loc[f, k] > threshold), None)
            if redundant_with is not None:
                pruned[f] = f"correlation > {threshold} with higher-ranked '{redundant_with}'"
            else:
                kept.append(f)
        return kept, pruned

    kept_numeric, corr_drop_reasons = correlation_prune(CORR_THRESHOLD_INITIAL)
    drop_reasons.update(corr_drop_reasons)
    kept_categorical = [f for f in CAT_COLS if f not in drop_reasons]
    selected_features = kept_numeric + kept_categorical

    print(f"\nAfter drop rules (corr threshold {CORR_THRESHOLD_INITIAL}): "
          f"{len(selected_features)} of {len(feature_cols)} features kept.")

    def retrain_and_eval(selected):
        cat_in_selected = [c for c in CAT_COLS if c in selected]
        X_train_sel = to_categorical(train_df[selected], cat_in_selected, categories_map)
        X_val_sel = to_categorical(val_df[selected], cat_in_selected, categories_map)
        m = train_xgb(X_train_sel, y_train)
        p = m.predict_proba(X_val_sel)[:, 1]
        return average_precision_score(y_val, p)

    reduced_pr_auc = retrain_and_eval(selected_features)
    used_threshold = CORR_THRESHOLD_INITIAL
    print(f"Reduced-set val PR-AUC (corr threshold {used_threshold}): {reduced_pr_auc:.4f} "
          f"(ceiling {kitchen_sink_pr_auc:.4f}, delta {kitchen_sink_pr_auc - reduced_pr_auc:.4f})")

    if kitchen_sink_pr_auc - reduced_pr_auc > PR_AUC_TOLERANCE:
        print(f"\nPR-AUC dropped more than {PR_AUC_TOLERANCE} -- relaxing correlation threshold to "
              f"{CORR_THRESHOLD_RELAXED} and retrying before dropping further ...")
        drop_reasons = {f: r for f, r in drop_reasons.items() if not r.startswith("correlation")}
        kept_numeric, corr_drop_reasons = correlation_prune(CORR_THRESHOLD_RELAXED)
        drop_reasons.update(corr_drop_reasons)
        kept_categorical = [f for f in CAT_COLS if f not in drop_reasons]
        selected_features = kept_numeric + kept_categorical
        used_threshold = CORR_THRESHOLD_RELAXED
        reduced_pr_auc = retrain_and_eval(selected_features)
        print(f"Reduced-set val PR-AUC (corr threshold {used_threshold}): {reduced_pr_auc:.4f} "
              f"(ceiling {kitchen_sink_pr_auc:.4f}, delta {kitchen_sink_pr_auc - reduced_pr_auc:.4f})")

    dropped_features = [f for f in feature_cols if f not in selected_features]

    with open(f"{REPORT_DIR}/selected_features.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(selected_features) + "\n")

    delta = kitchen_sink_pr_auc - reduced_pr_auc
    reason_counts = pd.Series(list(drop_reasons.values())).apply(
        lambda r: r.split(" with ")[0] if r.startswith("correlation") else r
    ).value_counts()

    summary_lines = [
        "# P2 Feature Selection Summary",
        "",
        f"- Kitchen sink features: {len(feature_cols)}",
        f"- Kitchen sink val PR-AUC (ceiling): {kitchen_sink_pr_auc:.4f}",
        f"- Correlation threshold used: {used_threshold}",
        f"- Reduced-set val PR-AUC: {reduced_pr_auc:.4f}",
        f"- PR-AUC delta from ceiling: {delta:.4f} "
        f"({'within' if delta <= PR_AUC_TOLERANCE else 'EXCEEDS'} {PR_AUC_TOLERANCE} tolerance)",
        f"- Features kept: {len(selected_features)}",
        f"- Features dropped: {len(dropped_features)}",
        "",
        "## Drop reasons (counts)",
        "",
    ]
    for reason, count in reason_counts.items():
        summary_lines.append(f"- {reason}: {count}")
    summary_lines.append("")
    summary_lines.append("## Dropped features")
    summary_lines.append("")
    for f in dropped_features:
        summary_lines.append(f"- `{f}`: {drop_reasons.get(f, 'unknown')}")

    with open(f"{REPORT_DIR}/selection_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    print("\n" + "=" * 60)
    print("P2 SUMMARY")
    print("=" * 60)
    print(f"Kitchen sink val PR-AUC : {kitchen_sink_pr_auc:.4f}  ({len(feature_cols)} features)")
    print(f"Reduced set val PR-AUC  : {reduced_pr_auc:.4f}  ({len(selected_features)} features)")
    print(f"Features kept: {len(selected_features)}  |  dropped: {len(dropped_features)}")
    print(f"\nSaved: {REPORT_DIR}/gain_importance.csv")
    print(f"Saved: {REPORT_DIR}/permutation_importance.csv")
    print(f"Saved: {REPORT_DIR}/disagreement_analysis.csv")
    print(f"Saved: {REPORT_DIR}/selected_features.txt")
    print(f"Saved: {REPORT_DIR}/selection_summary.md")


if __name__ == "__main__":
    main()
