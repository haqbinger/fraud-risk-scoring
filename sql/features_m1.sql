DROP TABLE IF EXISTS features_m1;

CREATE TABLE features_m1 AS
WITH base AS (
    SELECT
        transaction_id,
        is_fraud,
        txn_ts,
        transaction_amt,
        card1,
        device_info
    FROM transactions
),
velocity AS (
    SELECT
        transaction_id,
        COUNT(*) OVER (
            PARTITION BY card1 ORDER BY txn_ts
            RANGE BETWEEN INTERVAL '5 minutes' PRECEDING AND INTERVAL '1 second' PRECEDING
        ) AS velocity_5min,
        COUNT(*) OVER (
            PARTITION BY card1 ORDER BY txn_ts
            RANGE BETWEEN INTERVAL '30 minutes' PRECEDING AND INTERVAL '1 second' PRECEDING
        ) AS velocity_30min,
        COUNT(*) OVER (
            PARTITION BY card1 ORDER BY txn_ts
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND INTERVAL '1 second' PRECEDING
        ) AS velocity_1h,
        COUNT(*) OVER (
            PARTITION BY card1 ORDER BY txn_ts
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 second' PRECEDING
        ) AS velocity_24h
    FROM base
),
amount_stats AS (
    SELECT
        transaction_id,
        transaction_amt,
        AVG(transaction_amt) OVER w_past AS card1_hist_avg_amt,
        STDDEV(transaction_amt) OVER w_past AS card1_hist_std_amt,
        COUNT(*) OVER w_past AS card1_hist_txn_count
    FROM base
    WINDOW w_past AS (
        PARTITION BY card1 ORDER BY txn_ts, transaction_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),
recency AS (
    SELECT
        transaction_id,
        txn_ts,
        LAG(txn_ts) OVER (PARTITION BY card1 ORDER BY txn_ts, transaction_id) AS card1_prev_txn_ts
    FROM base
),
device_fraud_rate AS (
    SELECT
        transaction_id,
        device_info,
        AVG(is_fraud::float) OVER w_dev_past AS device_hist_fraud_rate,
        COUNT(*) OVER w_dev_past AS device_hist_txn_count
    FROM base
    WHERE device_info IS NOT NULL
    WINDOW w_dev_past AS (
        PARTITION BY device_info ORDER BY txn_ts, transaction_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
)
SELECT
    b.transaction_id,
    b.is_fraud,
    b.txn_ts,
    b.card1,
    v.velocity_5min,
    v.velocity_30min,
    v.velocity_1h,
    v.velocity_24h,
    a.card1_hist_avg_amt,
    a.card1_hist_std_amt,
    a.card1_hist_txn_count,
    CASE
        WHEN a.card1_hist_std_amt IS NULL OR a.card1_hist_std_amt = 0 THEN NULL
        ELSE (b.transaction_amt - a.card1_hist_avg_amt) / a.card1_hist_std_amt
    END AS amount_zscore_vs_card1_history,
    r.card1_prev_txn_ts,
    EXTRACT(EPOCH FROM (b.txn_ts - r.card1_prev_txn_ts)) AS seconds_since_card1_prev_txn,
    d.device_hist_fraud_rate,
    d.device_hist_txn_count
FROM base b
LEFT JOIN velocity v ON b.transaction_id = v.transaction_id
LEFT JOIN amount_stats a ON b.transaction_id = a.transaction_id
LEFT JOIN recency r ON b.transaction_id = r.transaction_id
LEFT JOIN device_fraud_rate d ON b.transaction_id = d.transaction_id;

CREATE UNIQUE INDEX idx_features_m1_id ON features_m1 (transaction_id);

SELECT transaction_id, txn_ts, velocity_5min, velocity_24h,
       amount_zscore_vs_card1_history, seconds_since_card1_prev_txn,
       device_hist_fraud_rate
FROM features_m1
ORDER BY txn_ts
LIMIT 20;