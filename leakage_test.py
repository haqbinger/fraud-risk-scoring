"""
M1 leakage test.

This does NOT trust the SQL. It independently recomputes each feature in
pandas, using only rows strictly before the target transaction's txn_ts,
then diffs against what features_m1.sql produced. If these disagree,
something in the SQL is looking at the wrong side of "now."

Usage:
    python test_leakage.py --n 8
"""

import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

DB_URL = "postgresql://fraud:fraud_dev_password@localhost:5433/fraud_db"


def recompute_velocity(txns: pd.DataFrame, card1, current_ts, window):
    past = txns[(txns.card1 == card1) & (txns.txn_ts < current_ts) & (txns.txn_ts >= current_ts - window)]
    return len(past)


def recompute_amount_zscore(txns: pd.DataFrame, card1, current_ts, current_id, current_amt):
    past = txns[
        (txns.card1 == card1)
        & ((txns.txn_ts < current_ts) | ((txns.txn_ts == current_ts) & (txns.transaction_id < current_id)))
    ]
    if len(past) == 0 or past.transaction_amt.std() in (0, None) or pd.isna(past.transaction_amt.std()):
        return None
    return (current_amt - past.transaction_amt.mean()) / past.transaction_amt.std()


def recompute_device_fraud_rate(txns: pd.DataFrame, device_info, current_ts, current_id):
    if pd.isna(device_info):
        return None
    past = txns[
        (txns.device_info == device_info)
        & ((txns.txn_ts < current_ts) | ((txns.txn_ts == current_ts) & (txns.transaction_id < current_id)))
    ]
    if len(past) == 0:
        return None
    return past.is_fraud.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="Number of random transactions to check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = create_engine(DB_URL)

    print("Loading transactions + features_m1 from Postgres ...")
    txns = pd.read_sql("SELECT * FROM transactions", engine, parse_dates=["txn_ts"])
    feats = pd.read_sql("SELECT * FROM features_m1", engine, parse_dates=["txn_ts"])

    rng = np.random.default_rng(args.seed)
    sample_ids = rng.choice(txns.transaction_id.values, size=args.n, replace=False)

    print(f"\nChecking {args.n} random transactions against independently recomputed values...\n")
    n_fail = 0
    for tid in sample_ids:
        row = txns[txns.transaction_id == tid].iloc[0]
        feat_row = feats[feats.transaction_id == tid].iloc[0]

        exp_v5 = recompute_velocity(txns, row.card1, row.txn_ts, pd.Timedelta(minutes=5))
        exp_v24 = recompute_velocity(txns, row.card1, row.txn_ts, pd.Timedelta(hours=24))
        exp_z = recompute_amount_zscore(txns, row.card1, row.txn_ts, row.transaction_id, row.transaction_amt)
        exp_dfr = recompute_device_fraud_rate(txns, row.device_info, row.txn_ts, row.transaction_id)

        checks = [
            ("velocity_5min", exp_v5, feat_row.velocity_5min),
            ("velocity_24h", exp_v24, feat_row.velocity_24h),
            ("amount_zscore_vs_card1_history", exp_z, feat_row.amount_zscore_vs_card1_history),
            ("device_hist_fraud_rate", exp_dfr, feat_row.device_hist_fraud_rate),
        ]

        print(f"transaction_id={tid}  card1={row.card1}  txn_ts={row.txn_ts}")
        for name, expected, actual in checks:
            if expected is None and (actual is None or pd.isna(actual)):
                ok = True
            elif expected is None or actual is None or pd.isna(actual):
                ok = False
            else:
                ok = abs(float(expected) - float(actual)) < 1e-6
            status = "OK" if ok else "MISMATCH"
            if not ok:
                n_fail += 1
            print(f"    {name:35s} expected={expected!s:>12}  sql={actual!s:>12}  [{status}]")
        print()

    if n_fail == 0:
        print("All checks passed. Now go break something on purpose:")
        print("  1. In features_m1.sql, change 'AND INTERVAL 1 second PRECEDING' to")
        print("     'AND INTERVAL 1 second FOLLOWING' on velocity_5min, re-run, re-test.")
        print("  2. Confirm this test suite actually catches it.")
        print("  This is the M1 'done when' criterion -- do it once for real.")
    else:
        print(f"{n_fail} mismatches found. Do not proceed to M2 until these are resolved.")


if __name__ == "__main__":
    main()