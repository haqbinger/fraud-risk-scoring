import pandas as pd
from sqlalchemy import text

from fraud.features import CATEGORICAL_COLS, encode_categorical, fit_median_impute

CARD1_HISTORY_SQL = text(
    """
    SELECT
        COUNT(*) FILTER (WHERE txn_ts >= :ts - INTERVAL '5 minutes') AS velocity_5min,
        COUNT(*) FILTER (WHERE txn_ts >= :ts - INTERVAL '30 minutes') AS velocity_30min,
        COUNT(*) FILTER (WHERE txn_ts >= :ts - INTERVAL '1 hour') AS velocity_1h,
        COUNT(*) FILTER (WHERE txn_ts >= :ts - INTERVAL '24 hours') AS velocity_24h,
        AVG(transaction_amt) AS card1_hist_avg_amt,
        STDDEV(transaction_amt) AS card1_hist_std_amt,
        COUNT(*) AS card1_hist_txn_count,
        MAX(txn_ts) AS card1_prev_txn_ts
    FROM transactions WHERE card1 = :card1 AND txn_ts < :ts
    """
)

DEVICE_HISTORY_SQL = text(
    """
    SELECT AVG(is_fraud::float) AS device_hist_fraud_rate,
           COUNT(*) AS device_hist_txn_count
    FROM transactions WHERE device_info = :device_info AND txn_ts < :ts
    """
)


def build_live_feature_vector(raw_input, engine, category_maps, medians, feature_columns):
    card1 = raw_input["card1"]
    txn_ts = raw_input["txn_ts"]
    transaction_amt = raw_input["transaction_amt"]
    device_info = raw_input.get("device_info")

    with engine.connect() as conn:
        card1_hist = conn.execute(
            CARD1_HISTORY_SQL, {"card1": card1, "ts": txn_ts}
        ).mappings().first()

        if device_info:
            device_hist = conn.execute(
                DEVICE_HISTORY_SQL, {"device_info": device_info, "ts": txn_ts}
            ).mappings().first()
        else:
            device_hist = {"device_hist_fraud_rate": None, "device_hist_txn_count": 0}

    card1_hist_avg_amt = card1_hist["card1_hist_avg_amt"]
    card1_hist_std_amt = card1_hist["card1_hist_std_amt"]
    card1_prev_txn_ts = card1_hist["card1_prev_txn_ts"]

    if card1_hist_avg_amt is None or not card1_hist_std_amt:
        amount_zscore_vs_card1_history = None
    else:
        amount_zscore_vs_card1_history = (
            transaction_amt - card1_hist_avg_amt
        ) / card1_hist_std_amt

    if card1_prev_txn_ts is None:
        seconds_since_card1_prev_txn = None
    else:
        seconds_since_card1_prev_txn = (txn_ts - card1_prev_txn_ts).total_seconds()

    numeric_row = {
        "velocity_5min": card1_hist["velocity_5min"],
        "velocity_30min": card1_hist["velocity_30min"],
        "velocity_1h": card1_hist["velocity_1h"],
        "velocity_24h": card1_hist["velocity_24h"],
        "card1_hist_avg_amt": card1_hist_avg_amt,
        "card1_hist_std_amt": card1_hist_std_amt,
        "card1_hist_txn_count": card1_hist["card1_hist_txn_count"],
        "amount_zscore_vs_card1_history": amount_zscore_vs_card1_history,
        "seconds_since_card1_prev_txn": seconds_since_card1_prev_txn,
        "device_hist_fraud_rate": device_hist["device_hist_fraud_rate"],
        "device_hist_txn_count": device_hist["device_hist_txn_count"],
        "transaction_amt": transaction_amt,
    }
    X_num = pd.DataFrame([numeric_row]).astype(float)

    cat_row = {col: raw_input.get(col) for col in CATEGORICAL_COLS}
    cat_df = pd.DataFrame([cat_row])
    X_cat = pd.DataFrame(index=cat_df.index)
    for col in CATEGORICAL_COLS:
        X_cat = pd.concat(
            [X_cat, encode_categorical(cat_df, col, category_maps[col])], axis=1
        )

    X = pd.concat([X_num, X_cat], axis=1)
    X = X.reindex(columns=feature_columns, fill_value=0)
    X = fit_median_impute(None, X, medians)
    return X
