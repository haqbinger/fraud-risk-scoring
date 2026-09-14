"""
P6 patch test: does adding product_cd-conditioned fraud rate (FN blind spot,
ADR V2-0008/0009) and card4-conditioned legitimacy rate (FP blind spot) to
features_m1 move val PR-AUC enough to be worth absorbing?

One quick kitchen-sink LightGBM retrain (existing lgbm_best_params.json, no
new tuning) on the 182 P2-selected features + the 4 new patch columns,
compared against the established 0.6741 ceiling. Val only.
"""
import json
import sys

sys.path.insert(0, "scripts")
from feature_selection_p2 import CAT_COLS  # noqa: E402
from model_comparison_p5 import train_lgbm_with_params  # noqa: E402
from temporal_cv_p3 import load_selected_features  # noqa: E402
from tuning_p4 import build_xy  # noqa: E402

from fraud.data import get_engine, load_data  # noqa: E402
from fraud.evaluations.metrics import compute_metrics  # noqa: E402
from fraud.evaluations.splits import temporal_split  # noqa: E402

LGBM_BEST_PARAMS_PATH = "reports/model_comparison/lgbm_best_params.json"
CEILING_VAL_PR_AUC = 0.6741
DELTA_ABSORB_THRESHOLD = 0.005
NEW_PATCH_COLS = [
    "product_cd_te_fraud_rate", "product_cd_te_txn_count",
    "card4_hist_legit_rate", "card4_hist_txn_count",
]


def main():
    selected_features = load_selected_features()
    expanded_features = selected_features + NEW_PATCH_COLS
    cat_cols = [c for c in CAT_COLS if c in expanded_features]
    print(f"Baseline: {len(selected_features)} features (ceiling val PR-AUC {CEILING_VAL_PR_AUC})")
    print(f"Patch test: {len(expanded_features)} features ({len(NEW_PATCH_COLS)} new: {NEW_PATCH_COLS})")

    with open(LGBM_BEST_PARAMS_PATH, encoding="utf-8") as fh:
        lgbm_best_params = json.load(fh)

    engine = get_engine()
    df = load_data(engine)
    for c in NEW_PATCH_COLS:
        if c not in df.columns:
            raise SystemExit(f"ERROR: {c} not found in load_data() output -- did the features_m1 rebuild finish?")
    train_full, val_full, _test_full = temporal_split(df, dt_col="txn_ts")
    del _test_full

    categories_map = {c: sorted(train_full[c].dropna().unique().tolist()) for c in cat_cols}
    X_train, y_train = build_xy(train_full, expanded_features, cat_cols, categories_map)
    X_val, y_val = build_xy(val_full, expanded_features, cat_cols, categories_map)

    print("Training kitchen-sink LightGBM (existing tuned params, no new tuning) on expanded feature set ...")
    model = train_lgbm_with_params(X_train, y_train, lgbm_best_params)
    val_prob = model.predict_proba(X_val)[:, 1]
    new_val_pr_auc = compute_metrics(y_val, val_prob)["pr_auc"]

    delta = new_val_pr_auc - CEILING_VAL_PR_AUC
    absorb = delta > DELTA_ABSORB_THRESHOLD

    print("\n" + "=" * 60)
    print("P6 PATCH TEST RESULT")
    print("=" * 60)
    print(f"Ceiling (182 features):         {CEILING_VAL_PR_AUC:.4f}")
    print(f"Patch test (186 features):      {new_val_pr_auc:.4f}")
    print(f"Delta:                          {delta:+.4f}")
    print(f"Verdict: {'ABSORB -- rerun P2/P5/P7 on expanded set' if absorb else 'NO LIFT -- revert SQL, close ADR 0008'}")

    with open("reports/error_analysis/p6_patch_test_result.json", "w", encoding="utf-8") as fh:
        json.dump({
            "ceiling_val_pr_auc": CEILING_VAL_PR_AUC,
            "patch_val_pr_auc": new_val_pr_auc,
            "delta": delta,
            "absorb_threshold": DELTA_ABSORB_THRESHOLD,
            "verdict": "absorb" if absorb else "no_lift_revert",
            "new_columns": NEW_PATCH_COLS,
        }, fh, indent=2)


if __name__ == "__main__":
    main()
