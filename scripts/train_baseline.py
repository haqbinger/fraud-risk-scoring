"""
M2: Trains Logistic Regression + Random Forest baselines using the
TEMPORAL split — the one every model in this project keeps using from
here forward. Evaluates on BOTH val (for model selection) AND test (for
a final, honest, unbiased number) — reports both, never just one.

"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from fraud.config import RANDOM_STATE
from fraud.data import get_engine, load_data
from fraud.features import build_feature_matrix, fit_median_impute
from fraud.evaluations.splits import temporal_split
from fraud.evaluations.metrics import compute_metrics, print_metrics, precision_at_recall


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

    # LogisticRegression needs scaled features (gradient-based solver) --
    # RandomForest doesn't (splits are scale-invariant), so it trains on
    # the unscaled X directly.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    logreg.fit(X_train_scaled, y_train)
    val_metrics = compute_metrics(y_val, logreg.predict_proba(X_val_scaled)[:, 1])
    print_metrics("LogisticRegression — VAL (model selection)", val_metrics)
    test_prob = logreg.predict_proba(X_test_scaled)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob)
    test_metrics["precision_at_80pct_recall"] = precision_at_recall(y_test, test_prob, 0.80)
    print_metrics("LogisticRegression — TEST (final, unbiased)", test_metrics)
    print(f"Precision @ 80% recall (test): {test_metrics['precision_at_80pct_recall']:.4f}")
    results["LogisticRegression"] = {"val": val_metrics, "test": test_metrics}

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    val_metrics = compute_metrics(y_val, rf.predict_proba(X_val)[:, 1])
    print_metrics("RandomForest — VAL (model selection)", val_metrics)
    test_prob = rf.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob)
    test_metrics["precision_at_80pct_recall"] = precision_at_recall(y_test, test_prob, 0.80)
    print_metrics("RandomForest — TEST (final, unbiased)", test_metrics)
    print(f"Precision @ 80% recall (test): {test_metrics['precision_at_80pct_recall']:.4f}")
    results["RandomForest"] = {"val": val_metrics, "test": test_metrics}

    print("\n" + "=" * 60)
    print("SUMMARY — model selection uses VAL PR-AUC; TEST is reported, never used to pick")
    print("=" * 60)
    for name, r in results.items():
        print(f"{name:20s} VAL PR-AUC: {r['val']['pr_auc']:.4f}   TEST PR-AUC: {r['test']['pr_auc']:.4f}")

    return results


if __name__ == "__main__":
    run()