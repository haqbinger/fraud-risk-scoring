import pandas as pd
from sqlalchemy import text

# V-column missingness blocks -- same grouping as sql/features_m1.sql's
# v_missingness CTE (P1 EDA: these columns share an upstream field and go
# missing together, so one flag per block).
V_BLOCK_SENTINELS = {
    "v_block1_missing": "V1",
    "v_block2_missing": "V12",
    "v_block3_missing": "V35",
    "v_block4_missing": "V53",
    "v_block5_missing": "V95",
    "v_block6_missing": "V138",
    "v_block7_missing": "V167",
    "v_block8_missing": "V279",
    "v_block9_missing": "V322",
}

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
        MAX(txn_ts) AS card1_prev_txn_ts,
        AVG(is_fraud::float) AS card1_te_fraud_rate
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

ADDR1_TE_SQL = text(
    """
    SELECT AVG(is_fraud::float) AS addr1_te_fraud_rate,
           COUNT(*) AS addr1_te_txn_count
    FROM transactions WHERE addr1 = :addr1 AND txn_ts < :ts
    """
)

PEMAILDOMAIN_TE_SQL = text(
    """
    SELECT AVG(is_fraud::float) AS p_emaildomain_te_fraud_rate,
           COUNT(*) AS p_emaildomain_te_txn_count
    FROM transactions WHERE p_emaildomain = :p_emaildomain AND txn_ts < :ts
    """
)

# Frequency encoding: plain full-table counts (not target-derived, so no
# leakage-safe `txn_ts <` filter in features_m1.sql either) -- but since the
# incoming transaction hasn't been inserted into `transactions` yet, the
# live count is naturally short by one relative to the offline batch count
# (which includes the row's own partition membership).
PEMAILDOMAIN_FREQ_SQL = text(
    "SELECT COUNT(*) AS freq FROM transactions WHERE p_emaildomain = :p_emaildomain"
)
REMAILDOMAIN_FREQ_SQL = text(
    'SELECT COUNT(*) AS freq FROM transactions WHERE "R_emaildomain" = :r_emaildomain'
)

# Raw pass-through columns -- must arrive in the request payload, no live
# computation needed.
PASSTHROUGH_NUMERIC = ["card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]
PASSTHROUGH_CATEGORICAL = ["card4", "card6", "product_cd", "p_emaildomain", "device_type"]


def _historical_aggregates(raw_input, engine):
    card1 = raw_input["card1"]
    txn_ts = raw_input["txn_ts"]
    device_info = raw_input.get("device_info")
    addr1 = raw_input.get("addr1")
    p_emaildomain = raw_input.get("p_emaildomain")
    r_emaildomain = raw_input.get("r_emaildomain")

    with engine.connect() as conn:
        card1_hist = conn.execute(
            CARD1_HISTORY_SQL, {"card1": card1, "ts": txn_ts}
        ).mappings().first()

        if device_info:
            device_hist = conn.execute(
                DEVICE_HISTORY_SQL, {"device_info": device_info, "ts": txn_ts}
            ).mappings().first()
        else:
            device_hist = {"device_hist_fraud_rate": None, "device_hist_txn_count": None}

        if addr1 is not None:
            addr1_hist = conn.execute(
                ADDR1_TE_SQL, {"addr1": addr1, "ts": txn_ts}
            ).mappings().first()
        else:
            addr1_hist = {"addr1_te_fraud_rate": None, "addr1_te_txn_count": None}

        if p_emaildomain:
            pemail_hist = conn.execute(
                PEMAILDOMAIN_TE_SQL, {"p_emaildomain": p_emaildomain, "ts": txn_ts}
            ).mappings().first()
            p_emaildomain_freq = conn.execute(
                PEMAILDOMAIN_FREQ_SQL, {"p_emaildomain": p_emaildomain}
            ).scalar()
        else:
            pemail_hist = {"p_emaildomain_te_fraud_rate": None, "p_emaildomain_te_txn_count": None}
            p_emaildomain_freq = 0

        if r_emaildomain:
            r_emaildomain_freq = conn.execute(
                REMAILDOMAIN_FREQ_SQL, {"r_emaildomain": r_emaildomain}
            ).scalar()
        else:
            r_emaildomain_freq = 0

    return card1_hist, device_hist, addr1_hist, pemail_hist, p_emaildomain_freq, r_emaildomain_freq


def build_live_feature_vector(raw_input, engine, selected_features, cat_cols, categories_map):
    """Computes the full features_m1-equivalent row for one incoming
    transaction, then reindexes/casts it down to exactly the columns the
    deployed model expects (`selected_features`, in order)."""
    txn_ts = raw_input["txn_ts"]
    transaction_amt = raw_input["transaction_amt"]

    card1_hist, device_hist, addr1_hist, pemail_hist, p_emaildomain_freq, r_emaildomain_freq = (
        _historical_aggregates(raw_input, engine)
    )

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

    row = {
        "transaction_amt": transaction_amt,
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
        "card1_te_fraud_rate": card1_hist["card1_te_fraud_rate"],
        "addr1_te_fraud_rate": addr1_hist["addr1_te_fraud_rate"],
        "addr1_te_txn_count": addr1_hist["addr1_te_txn_count"],
        "p_emaildomain_te_fraud_rate": pemail_hist["p_emaildomain_te_fraud_rate"],
        "p_emaildomain_te_txn_count": pemail_hist["p_emaildomain_te_txn_count"],
        "p_emaildomain_freq": p_emaildomain_freq,
        "r_emaildomain_freq": r_emaildomain_freq,
    }

    for flag, sentinel_col in V_BLOCK_SENTINELS.items():
        row[flag] = 1 if raw_input.get(sentinel_col) is None else 0

    for col in PASSTHROUGH_NUMERIC:
        row[col] = raw_input.get(col)
    for col in PASSTHROUGH_CATEGORICAL:
        row[col] = raw_input.get(col)

    for name, value in raw_input.items():
        if name[0] in ("C", "D", "V", "M") and name[1:].isdigit():
            row.setdefault(name, value)

    X = pd.DataFrame([row])
    X = X.reindex(columns=selected_features)

    for col in cat_cols:
        X[col] = pd.Categorical(X[col], categories=categories_map[col])
    numeric_cols = [c for c in selected_features if c not in cat_cols]
    X[numeric_cols] = X[numeric_cols].astype(float)

    return X
