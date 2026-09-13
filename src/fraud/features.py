import pandas as pd

NUMERIC_COLS = [
    "velocity_5min", "velocity_30min", "velocity_1h", "velocity_24h",
    "card1_hist_avg_amt", "card1_hist_std_amt", "card1_hist_txn_count",
    "amount_zscore_vs_card1_history", "seconds_since_card1_prev_txn",
    "device_hist_fraud_rate", "device_hist_txn_count", "transaction_amt",
]
CATEGORICAL_COLS = ["product_cd", "p_emaildomain", "device_type"]


def compute_top_categories(df: pd.DataFrame, col: str, n: int = 8):
    return list(df[col].value_counts().nlargest(n).index)


def encode_categorical(df: pd.DataFrame, col: str, top_categories):
    capped = df[col].where(df[col].isin(top_categories), other="__other__")
    return pd.get_dummies(capped, prefix=col, dummy_na=True)


def build_feature_matrix(df: pd.DataFrame, category_maps: dict = None):
    """
    category_maps: if None (training mode), top-8 categories per column
    are computed from `df` and returned alongside X, y. If provided
    (val/test/serving mode), those exact categories are reused so new
    rows encode into the same columns the model was trained on.

    Returns: (X, y, category_maps_used)
    """
    X_num = df[NUMERIC_COLS].copy()

    X_cat = pd.DataFrame(index=df.index)
    used_maps = {}
    for col in CATEGORICAL_COLS:
        top_categories = category_maps[col] if category_maps else compute_top_categories(df, col)
        used_maps[col] = top_categories
        X_cat = pd.concat([X_cat, encode_categorical(df, col, top_categories)], axis=1)

    X = pd.concat([X_num, X_cat], axis=1)
    y = df["is_fraud"].values if "is_fraud" in df.columns else None
    return X, y, used_maps


def fit_median_impute(X_train, X_all, medians=None):
    """Fit medians on TRAIN only, apply everywhere else. Pass a saved
    `medians` dict (rather than refitting) for val/test/serving."""
    medians = medians if medians is not None else X_train.median(numeric_only=True)
    return X_all.fillna(medians)