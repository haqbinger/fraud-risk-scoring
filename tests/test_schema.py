import pandas as pd
from sqlalchemy import text

from fraud.features import NUMERIC_COLS

TRANSACTIONS_REQUIRED_COLS = {
    "transaction_id", "is_fraud", "txn_ts", "transaction_amt", "card1", "device_info",
}


def _columns(engine, table_name):
    query = text(
        "SELECT column_name FROM information_schema.columns WHERE table_name = :table_name"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"table_name": table_name}).fetchall()
    return {r[0] for r in rows}


def test_transactions_table_has_required_columns(engine):
    cols = _columns(engine, "transactions")
    missing = TRANSACTIONS_REQUIRED_COLS - cols
    assert not missing, f"transactions is missing columns: {missing}"


def test_features_m1_table_has_required_columns(engine):
    # transaction_amt lives in `transactions`, not `features_m1` — data.py
    # joins it in from t.transaction_amt, so it's excluded here.
    cols = _columns(engine, "features_m1")
    required = (set(NUMERIC_COLS) - {"transaction_amt"}) | {"transaction_id", "is_fraud", "txn_ts"}
    missing = required - cols
    assert not missing, f"features_m1 is missing columns: {missing}"


def test_transaction_id_is_unique_in_transactions(engine):
    query = text(
        "SELECT COUNT(*) AS total, COUNT(DISTINCT transaction_id) AS distinct_total FROM transactions"
    )
    with engine.connect() as conn:
        total, distinct_total = conn.execute(query).one()
    assert total == distinct_total


def test_transaction_id_is_unique_in_features_m1(engine):
    query = text(
        "SELECT COUNT(*) AS total, COUNT(DISTINCT transaction_id) AS distinct_total FROM features_m1"
    )
    with engine.connect() as conn:
        total, distinct_total = conn.execute(query).one()
    assert total == distinct_total


def test_is_fraud_is_binary(engine):
    query = text("SELECT DISTINCT is_fraud FROM transactions")
    with engine.connect() as conn:
        values = {row[0] for row in conn.execute(query).fetchall()}
    assert values <= {0, 1}


def test_velocity_windows_monotonically_increasing(engine):
    df = pd.read_sql(
        "SELECT velocity_5min, velocity_30min, velocity_1h, velocity_24h FROM features_m1",
        engine,
    )
    assert (df.velocity_24h >= df.velocity_1h).all()
    assert (df.velocity_1h >= df.velocity_30min).all()
    assert (df.velocity_30min >= df.velocity_5min).all()
