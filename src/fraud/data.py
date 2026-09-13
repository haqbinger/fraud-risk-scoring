import pandas as pd
from sqlalchemy import create_engine

from fraud.config import DB_URL


def get_engine():
    return create_engine(DB_URL)


def load_data(engine=None):
    """
    features_m1 JOIN transactions — the joined, feature-engineered
    dataset every model-training script (M2 onward) consumes. Ordered
    by txn_ts so downstream chronological_split/temporal_split can just
    slice by row position without re-sorting.
    """
    engine = engine or get_engine()
    print("Loading joined features + raw columns from Postgres ...")
    query = """
        SELECT
            f.transaction_id,
            f.is_fraud,
            f.txn_ts,
            f.velocity_5min,
            f.velocity_30min,
            f.velocity_1h,
            f.velocity_24h,
            f.card1_hist_avg_amt,
            f.card1_hist_std_amt,
            f.card1_hist_txn_count,
            f.amount_zscore_vs_card1_history,
            f.seconds_since_card1_prev_txn,
            f.device_hist_fraud_rate,
            f.device_hist_txn_count,
            t.transaction_amt,
            t.product_cd,
            t.card1,
            t.addr1,
            t.p_emaildomain,
            t.device_type
        FROM features_m1 f
        JOIN transactions t ON f.transaction_id = t.transaction_id
        ORDER BY f.txn_ts
    """
    df = pd.read_sql(query, engine, parse_dates=["txn_ts"])
    print(f"Loaded {len(df)} rows, fraud rate = {df.is_fraud.mean():.4%}")
    return df