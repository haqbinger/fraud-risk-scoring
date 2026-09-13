"""
P1 patch verification: card2, card3, card4, card5, card6, addr2, dist1, dist2,
and R_emaildomain frequency encoding were added to features_m1.sql / load_data()
after P2 had already run its feature selection on the old (410-column) matrix.

This retrains the kitchen-sink XGBoost on the expanded matrix and compares its
val PR-AUC against the P2 ceiling (0.5725, all-default hyperparams, 411
features). If the ceiling moves by more than 0.005, P2's selection is stale
and gets rerun on the expanded feature set; otherwise the old selected_features
stand and P3 can proceed.
"""
import sys

from sklearn.metrics import average_precision_score

sys.path.insert(0, "scripts")
from feature_selection_p2 import (  # noqa: E402
    CAT_COLS, EXCLUDE_COLS, build_xy, main as run_p2_selection, train_xgb,
)

from fraud.data import get_engine, load_data  # noqa: E402
from fraud.evaluations.splits import temporal_split  # noqa: E402

OLD_CEILING = 0.5725
CEILING_MOVE_THRESHOLD = 0.005


def main():
    engine = get_engine()
    df = load_data(engine)
    train_df, val_df, _test_df = temporal_split(df, dt_col="txn_ts")
    del _test_df

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    categories_map = {c: sorted(train_df[c].dropna().unique().tolist()) for c in CAT_COLS}

    X_train, y_train = build_xy(train_df, feature_cols, categories_map, CAT_COLS)
    X_val, y_val = build_xy(val_df, feature_cols, categories_map, CAT_COLS)

    print(f"\nExpanded kitchen sink: {len(feature_cols)} features "
          f"(was 411 before card2-6/addr2/dist1-2) | train={len(X_train)} | val={len(X_val)}")

    print("Training kitchen-sink XGBoost on expanded matrix (default hyperparams) ...")
    model = train_xgb(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    new_ceiling = average_precision_score(y_val, val_prob)

    delta = new_ceiling - OLD_CEILING
    rerun_needed = abs(delta) > CEILING_MOVE_THRESHOLD
    verdict = "RERUN P2 selection on expanded feature set" if rerun_needed else "proceed to P3 (old selected_features.txt still stands)"

    print("\n" + "=" * 60)
    print("P1 PATCH VERIFICATION")
    print("=" * 60)
    print(f"Old ceiling (411 features): {OLD_CEILING:.4f}")
    print(f"New ceiling ({len(feature_cols)} features): {new_ceiling:.4f}")
    print(f"Delta: {delta:+.4f}  (threshold: {CEILING_MOVE_THRESHOLD})")
    print(f"Verdict: {verdict}")

    if rerun_needed:
        print("\nCeiling moved by more than the threshold -- rerunning full P2 "
              "selection (gain + permutation importance, correlation pruning) "
              "on the expanded feature set ...\n")
        run_p2_selection()


if __name__ == "__main__":
    main()
