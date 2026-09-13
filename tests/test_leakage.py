"""
Formal pytest version of leakage_test.py at the project root.

Does NOT trust the SQL in sql/features_m1.sql. Independently recomputes
each feature in pandas, using only rows strictly before the target
transaction's txn_ts, then diffs against what features_m1 produced.
"""
import numpy as np
import pandas as pd
import pytest

N_SAMPLES = 8
SEED = 42


@pytest.fixture(scope="module")
def txns(engine):
    return pd.read_sql("SELECT * FROM transactions", engine, parse_dates=["txn_ts"])


@pytest.fixture(scope="module")
def feats(engine):
    return pd.read_sql("SELECT * FROM features_m1", engine, parse_dates=["txn_ts"])


@pytest.fixture(scope="module")
def sample_ids(txns):
    rng = np.random.default_rng(SEED)
    return rng.choice(txns.transaction_id.values, size=N_SAMPLES, replace=False)


def _recompute_velocity(txns, card1, current_ts, window):
    past = txns[
        (txns.card1 == card1)
        & (txns.txn_ts < current_ts)
        & (txns.txn_ts >= current_ts - window)
    ]
    return len(past)


def _recompute_amount_zscore(txns, card1, current_ts, current_id, current_amt):
    past = txns[
        (txns.card1 == card1)
        & ((txns.txn_ts < current_ts) | ((txns.txn_ts == current_ts) & (txns.transaction_id < current_id)))
    ]
    std = past.transaction_amt.std()
    if len(past) == 0 or std in (0, None) or pd.isna(std):
        return None
    return (current_amt - past.transaction_amt.mean()) / std


def _recompute_device_fraud_rate(txns, device_info, current_ts, current_id):
    if pd.isna(device_info):
        return None
    past = txns[
        (txns.device_info == device_info)
        & ((txns.txn_ts < current_ts) | ((txns.txn_ts == current_ts) & (txns.transaction_id < current_id)))
    ]
    if len(past) == 0:
        return None
    return past.is_fraud.mean()


def _assert_matches(expected, actual):
    if expected is None or (isinstance(expected, float) and pd.isna(expected)):
        assert actual is None or pd.isna(actual)
    else:
        assert actual is not None and not pd.isna(actual)
        assert abs(float(expected) - float(actual)) < 1e-6


def test_velocity_5min_matches_pandas_recomputation(txns, feats, sample_ids):
    for tid in sample_ids:
        row = txns[txns.transaction_id == tid].iloc[0]
        feat_row = feats[feats.transaction_id == tid].iloc[0]
        expected = _recompute_velocity(txns, row.card1, row.txn_ts, pd.Timedelta(minutes=5))
        assert expected == feat_row.velocity_5min


def test_velocity_24h_matches_pandas_recomputation(txns, feats, sample_ids):
    for tid in sample_ids:
        row = txns[txns.transaction_id == tid].iloc[0]
        feat_row = feats[feats.transaction_id == tid].iloc[0]
        expected = _recompute_velocity(txns, row.card1, row.txn_ts, pd.Timedelta(hours=24))
        assert expected == feat_row.velocity_24h


def test_amount_zscore_matches_pandas_recomputation(txns, feats, sample_ids):
    for tid in sample_ids:
        row = txns[txns.transaction_id == tid].iloc[0]
        feat_row = feats[feats.transaction_id == tid].iloc[0]
        expected = _recompute_amount_zscore(
            txns, row.card1, row.txn_ts, row.transaction_id, row.transaction_amt
        )
        _assert_matches(expected, feat_row.amount_zscore_vs_card1_history)


def test_device_fraud_rate_matches_pandas_recomputation(txns, feats, sample_ids):
    for tid in sample_ids:
        row = txns[txns.transaction_id == tid].iloc[0]
        feat_row = feats[feats.transaction_id == tid].iloc[0]
        expected = _recompute_device_fraud_rate(txns, row.device_info, row.txn_ts, row.transaction_id)
        _assert_matches(expected, feat_row.device_hist_fraud_rate)


def test_first_card1_transaction_has_null_history(txns, feats):
    first_per_card1 = (
        txns.sort_values(["card1", "txn_ts", "transaction_id"])
        .groupby("card1")
        .first()
        .reset_index()[["card1", "transaction_id"]]
    )
    first_txn = first_per_card1.iloc[0]
    feat_row = feats[feats.transaction_id == first_txn.transaction_id].iloc[0]
    assert pd.isna(feat_row.card1_hist_avg_amt)
