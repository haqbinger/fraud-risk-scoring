import os

DB_URL = os.environ.get(
    "FRAUD_DB_URL",
    "postgresql://fraud:fraud_dev_password@localhost:5433/fraud_db",
)

RANDOM_STATE = 42

# --- Business cost assumptions (M4) ---
# These are intentionally conservative, but realistic for a fraud operation:
# - a missed fraud can trigger chargeback, investigation, regulatory exposure, and reputation damage
# - a false positive is a manual review or blocked legitimate transaction, which still has cost
COST_FALSE_NEGATIVE = 2000.0
COST_FALSE_POSITIVE = 150.0

# Operational cap for manual review volume.
# Most fraud teams cannot review more than ~5-10% of transactions without scaling operations.
MAX_REVIEW_RATE = 0.05