import os

DB_URL = os.environ.get(
    "FRAUD_DB_URL",
    "postgresql://fraud:fraud_dev_password@localhost:5433/fraud_db",
)

RANDOM_STATE = 42

# --- Business cost assumptions (M4) ---
# Assumed numbers, stated explicitly (see docs/m0_problem_framing.md).
# M4's threshold and its sensitivity table are the only things
# conditional on these — M1-M3 don't depend on them being "correct."
COST_FALSE_NEGATIVE = 500.0   # missed fraud
COST_FALSE_POSITIVE = 5.0     # wrongly blocked legitimate transaction