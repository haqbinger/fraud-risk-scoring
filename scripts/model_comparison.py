import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud.config import RANDOM_STATE
from fraud.data import get_engine, load_data
from fraud.features import build_feature_matrix, fit_median_impute
from fraud.evaluations.splits import temporal_split
from fraud.evaluations.metrics import compute_metrics, print_metrics, precision_at_recall


def time_fit(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - start


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

    # Only LogisticRegression needs scaling; computed once, reused for
    # whichever model needs it below.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    fit_t = time_fit(logreg, X_train_scaled, y_train)
    val_metrics = compute_metrics(y_val, logreg.predict_proba(X_val_scaled)[:, 1])
    print_metrics("LogisticRegression -- VAL", val_metrics)
    results["LogisticRegression"] = {
        "model": logreg, "val": val_metrics, "fit_seconds": fit_t, "needs_scaling": True,
    }

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    fit_t = time_fit(rf, X_train, y_train)
    val_metrics = compute_metrics(y_val, rf.predict_proba(X_val)[:, 1])
    print_metrics("RandomForest -- VAL", val_metrics)
    results["RandomForest"] = {
        "model": rf, "val": val_metrics, "fit_seconds": fit_t, "needs_scaling": False,
    }

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
    )
    fit_t = time_fit(xgb_model, X_train, y_train)
    val_metrics = compute_metrics(y_val, xgb_model.predict_proba(X_val)[:, 1])
    print_metrics("XGBoost -- VAL", val_metrics)
    results["XGBoost"] = {
        "model": xgb_model, "val": val_metrics, "fit_seconds": fit_t, "needs_scaling": False,
    }

    print("\n" + "=" * 70)
    print("MODEL SELECTION (VAL PR-AUC) -- test set not touched yet")
    print("=" * 70)
    print(f"{'Model':<20}{'PR-AUC':>10}{'ROC-AUC':>10}{'Brier':>10}{'Fit(s)':>10}")
    for name, r in results.items():
        m = r["val"]
        print(f"{name:<20}{m['pr_auc']:>10.4f}{m['roc_auc']:>10.4f}{m['brier_score']:>10.4f}{r['fit_seconds']:>10.2f}")

    winner_name = max(results.items(), key=lambda kv: kv[1]["val"]["pr_auc"])[0]
    print(f"\nHighest VAL PR-AUC: {winner_name}")

    # Test touched here, once, only for the model already selected above.
    winner = results[winner_name]
    X_test_final = X_test_scaled if winner["needs_scaling"] else X_test
    test_prob = winner["model"].predict_proba(X_test_final)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob)
    test_metrics["precision_at_80pct_recall"] = precision_at_recall(y_test, test_prob, 0.80)
    print_metrics(f"{winner_name} -- TEST (final, unbiased, touched once)", test_metrics)

    print(
        "\nThis is your M3 'done when' evidence: justify the choice in one "
        "paragraph using these actual numbers -- PR-AUC gap over the "
        "next-best model, fit-time cost, and interpretability tradeoffs "
        "(LogReg coefficients vs. RF/XGBoost feature importances + SHAP "
        "support coming in M5). Write it into decisions/0004."
    )

    return results, winner_name, test_metrics


if __name__ == "__main__":
    run()