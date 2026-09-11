import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://fraud:fraud_dev_password@localhost:5433/fraud_db"
REFERENCE_DATETIME = "2017-11-30 00:00:00"


def load_csv_to_table(engine, path: str, table_name: str):
    if not os.path.exists(path):
        print(f"ERROR: expected file not found: {path}")
        sys.exit(1)

    print(f"Reading {path} to infer schema ...")
    # Read the whole file once for pandas' dtype inference, then create
    # the table with ZERO rows (head(0)) -- schema-only, fast regardless
    # of file size. The actual bulk load uses Postgres's native COPY,
    # not to_sql -- to_sql's "multi" method builds one INSERT per chunk
    # with (rows x columns) parameters. For a ~394-column file like
    # train_transaction.csv, a 50,000-row chunk is ~19.7 million
    # parameters in a single statement, blowing past Postgres's
    # 65,535-parameter-per-statement limit. COPY has no such limit.
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    df.head(0).to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"  Table {table_name} created (schema only)")

    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        with open(path, "r", encoding="utf-8") as f:
            next(f)  # skip header -- COPY below doesn't expect one
            cursor.copy_expert(f'COPY "{table_name}" FROM STDIN WITH (FORMAT csv)', f)
        raw_conn.commit()
        print(f"  Bulk loaded via COPY: {table_name} ({len(df)} rows)")
    finally:
        raw_conn.close()


def build_transactions_table(engine):
    print("Building `transactions` table (join + synthetic timestamp) ...")
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS transactions;"))
        conn.execute(text(f"""
            CREATE TABLE transactions AS
            SELECT
                t."TransactionID"                               AS transaction_id,
                t."isFraud"                                     AS is_fraud,
                t."TransactionDT"                                AS transaction_dt,
                TIMESTAMP '{REFERENCE_DATETIME}'
                    + (t."TransactionDT" * INTERVAL '1 second')  AS txn_ts,
                t."TransactionAmt"                               AS transaction_amt,
                t."ProductCD"                                    AS product_cd,
                t."card1"                                        AS card1,
                t."card2"                                        AS card2,
                t."addr1"                                        AS addr1,
                t."P_emaildomain"                                AS p_emaildomain,
                i."DeviceType"                                   AS device_type,
                i."DeviceInfo"                                   AS device_info
            FROM raw_transaction t
            LEFT JOIN raw_identity i ON t."TransactionID" = i."TransactionID";
        """))
        conn.execute(text('CREATE INDEX idx_transactions_card1_ts ON transactions (card1, txn_ts);'))
        conn.execute(text('CREATE INDEX idx_transactions_device_ts ON transactions (device_info, txn_ts);'))
        conn.execute(text('CREATE UNIQUE INDEX idx_transactions_id ON transactions (transaction_id);'))
    print("Done: transactions table built and indexed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    engine = create_engine(DB_URL)
    load_csv_to_table(engine, os.path.join(args.data_dir, "train_transaction.csv"), "raw_transaction")
    load_csv_to_table(engine, os.path.join(args.data_dir, "train_identity.csv"), "raw_identity")
    build_transactions_table(engine)

    print("\nAll done. Row counts:")
    with engine.connect() as conn:
        for tbl in ["raw_transaction", "raw_identity", "transactions"]:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {tbl};")).scalar()
            print(f"  {tbl}: {n}")


if __name__ == "__main__":
    main()