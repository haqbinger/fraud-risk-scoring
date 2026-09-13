"""
P0 EDA over raw_transaction: column inventory, V-col missingness, categorical
cardinality, and a univariate signal proxy (rank-based AUC via Mann-Whitney U).

Standalone: only sqlalchemy, pandas, numpy, matplotlib. No fraud package imports.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://fraud:fraud_dev_password@localhost:5433/fraud_db"
TABLE = "raw_transaction"
REPORT_DIR = "reports/eda"
TARGET_COL = "isFraud"

import os

os.makedirs(REPORT_DIR, exist_ok=True)

engine = create_engine(DB_URL)

# ---------------------------------------------------------------------------
# 1. Column inventory from information_schema
# ---------------------------------------------------------------------------
with engine.connect() as conn:
    cols_df = pd.read_sql(
        text(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = :table
            ORDER BY ordinal_position
            """
        ),
        conn,
        params={"table": TABLE},
    )

all_cols = cols_df["column_name"].tolist()
numeric_types = {"integer", "bigint", "smallint", "double precision", "real", "numeric"}

v_cols = [c for c in all_cols if c.startswith("V") and c[1:].isdigit()]
categorical_cols = [
    c
    for c, dt in zip(cols_df["column_name"], cols_df["data_type"])
    if dt not in numeric_types and c not in v_cols and c not in ("TransactionID",)
]
numeric_non_v_cols = [
    c
    for c, dt in zip(cols_df["column_name"], cols_df["data_type"])
    if dt in numeric_types and c not in v_cols and c not in ("TransactionID", TARGET_COL)
]

print(f"Total columns: {len(all_cols)}")
print(f"V-columns: {len(v_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Non-V numeric columns: {len(numeric_non_v_cols)}")

# ---------------------------------------------------------------------------
# 2. Missingness rate for ALL V-cols in a single SQL pass
# ---------------------------------------------------------------------------
if v_cols:
    select_clauses = ",\n".join(
        f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)::float / COUNT(*) AS "{c}"'
        for c in v_cols
    )
    query = f'SELECT\n{select_clauses}\nFROM {TABLE}'
    with engine.connect() as conn:
        missing_row = pd.read_sql(text(query), conn)

    missing_series = missing_row.iloc[0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(8, len(v_cols) * 0.06)))
    ax.barh(missing_series.index, missing_series.values, color="steelblue")
    ax.set_xlabel("Missingness rate")
    ax.set_title(f"Missingness rate for {len(v_cols)} V-columns (sorted)")
    ax.invert_yaxis()
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(f"{REPORT_DIR}/missingness_heatmap.png", dpi=150)
    plt.close(fig)
else:
    missing_series = pd.Series(dtype=float)

# ---------------------------------------------------------------------------
# 3. Cardinality + null_rate for categorical columns
# ---------------------------------------------------------------------------
cardinality_rows = []
with engine.connect() as conn:
    for c in categorical_cols:
        result = conn.execute(
            text(
                f"""
                SELECT
                    COUNT(DISTINCT "{c}") AS cardinality,
                    SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)::float / COUNT(*) AS null_rate
                FROM {TABLE}
                """
            )
        ).fetchone()
        cardinality_rows.append({"column": c, "cardinality": result[0], "null_rate": result[1]})

cardinality_df = pd.DataFrame(cardinality_rows).sort_values("cardinality", ascending=False)
cardinality_df.to_csv(f"{REPORT_DIR}/cardinality_table.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Univariate AUC proxy via Mann-Whitney U (rank-based), sampled at 20%
# ---------------------------------------------------------------------------
candidate_cols = numeric_non_v_cols + v_cols[::5]

sample_cols = ", ".join(f'"{c}"' for c in candidate_cols)
sample_query = f"""
    SELECT {sample_cols}, "{TARGET_COL}"
    FROM {TABLE} TABLESAMPLE BERNOULLI(20)
"""
with engine.connect() as conn:
    sample_df = pd.read_sql(text(sample_query), conn)

def mann_whitney_auc(values: pd.Series, labels: pd.Series) -> float:
    mask = values.notna()
    values = values[mask]
    labels = labels[mask]
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0 or n_neg == 0 or len(values) < 2:
        return np.nan
    ranks = pd.Series(values).rank().values
    rank_sum_pos = ranks[labels.values == 1].sum()
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2
    auc = u_stat / (n_pos * n_neg)
    return max(auc, 1 - auc)

labels = sample_df[TARGET_COL]
auc_scores = {}
for c in candidate_cols:
    auc_scores[c] = mann_whitney_auc(sample_df[c], labels)

auc_series = pd.Series(auc_scores).dropna().sort_values(ascending=False)
top20 = auc_series.head(20)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top20.index, top20.values, color="darkorange")
ax.set_xlabel("Univariate AUC proxy (|Mann-Whitney U based AUC|)")
ax.set_title("Top 20 features by univariate signal")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(f"{REPORT_DIR}/top20_signal.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print("\n=== EDA P0 Summary ===")
print(f"Total columns: {len(all_cols)}")
print(f"V-columns: {len(v_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Non-V numeric columns: {len(numeric_non_v_cols)}")

print("\nTop 10 missing V-columns by rate:")
print(missing_series.head(10).to_string())

print("\nTop 5 high-cardinality categoricals:")
print(cardinality_df.head(5).to_string(index=False))

print("\nTop 20 features by univariate signal:")
print(top20.to_string())

print(f"\nSaved: {REPORT_DIR}/missingness_heatmap.png")
print(f"Saved: {REPORT_DIR}/cardinality_table.csv")
print(f"Saved: {REPORT_DIR}/top20_signal.png")
