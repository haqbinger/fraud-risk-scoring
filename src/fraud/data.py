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

    features_m1 (P1, patched) carries all V/C/D/M raw columns, card2-6,
    addr2, dist1/dist2, plus the engineered velocity/amount/device/
    target-encoding/frequency-encoding/missingness-flag features, so `f.*`
    covers it all; only the handful of curated columns not present in
    features_m1 are pulled from `transactions`.
    """
    engine = engine or get_engine()
    print("Loading joined features + raw columns from Postgres ...")
    query = """
        SELECT
            f.*,
            t.transaction_amt,
            t.product_cd,
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