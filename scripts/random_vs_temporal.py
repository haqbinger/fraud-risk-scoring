"""
M2: The quantified random-vs-temporal comparison — arguably the single
most interview-valuable artifact from this milestone. Doesn't just claim
temporal splitting is more correct, proves how much the naive approach
would have overstated performance, on your actual data.

Trains the SAME model, SAME features — only the split strategy differs.

Usage:
    python scripts/random_vs_temporal.py
"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fraud.config import RANDOM_STATE
from fraud.data import get_engine, load_data
from fraud.features import build_feature_matrix, fit_median_impute
from fraud.evaluations.splits import temporal_split, random_split
from fraud.evaluations.metrics import compute_metrics, print_metrics


def evaluate_split_strategy(name, train_df, val_df):
    X_train, y_train, category_maps = build_feature_matrix(train_df)
    X_val, y_val, _ = build_feature_matrix(val_df, category_maps=category_maps)

    medians = X_train.median(numeric_only=True)
    X_train = fit_median_impute(X_train, X_train, medians)
    X_val = fit_median_impute(X_train, X_val, medians)

    # Same fix as train_baseline.py -- lbfgs needs scaled features or it
    # doesn't converge, and a non-converged model makes this whole
    # comparison meaningless regardless of which split "wins."
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]

    m = compute_metrics(y_val, y_prob)
    print_metrics(name, m)
    return m


def main():
    engine = get_engine()
    df = load_data(engine)

    print("### RANDOM SPLIT (the naive/standard-practice approach) ###")
    r_train, r_val, r_test = random_split(df)
    random_metrics = evaluate_split_strategy("Random split", r_train, r_val)

    print("\n### TEMPORAL SPLIT (the correct approach for this data) ###")
    t_train, t_val, t_test = temporal_split(df, dt_col="txn_ts")
    temporal_metrics = evaluate_split_strategy("Temporal split", t_train, t_val)

    gap = random_metrics["pr_auc"] - temporal_metrics["pr_auc"]
    print("\n" + "=" * 60)
    print("THE HEADLINE NUMBER")
    print("=" * 60)
    print(f"Random split PR-AUC:   {random_metrics['pr_auc']:.4f}")
    print(f"Temporal split PR-AUC: {temporal_metrics['pr_auc']:.4f}")
    print(f"Gap: {gap:.4f} PR-AUC points")
    print(
        "\nIf gap > 0: random splitting made the model look better than it "
        "would actually perform in production. Copy this number into "
        "decisions/0001-temporal-split-not-random.md's Consequences section."
    )


if __name__ == "__main__":
    main()