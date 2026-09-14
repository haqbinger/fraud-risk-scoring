"""
Standalone ROC-AUC check on the test set, using the saved calibrated
pipeline and metadata from P7. See reports/calibration/p7_summary.md
for why PR-AUC (not ROC-AUC) is the primary metric for this model.
"""
import json

import joblib
from sklearn.metrics import roc_auc_score

from fraud.data import get_engine, load_data
from fraud.evaluations.splits import temporal_split

MODELS_DIR = "models"
REPORT_PATH = "reports/calibration/p7_summary.md"


def main():
    with open(f"{MODELS_DIR}/model_metadata.json", encoding="utf-8") as fh:
        metadata = json.load(fh)

    pipeline = joblib.load(f"{MODELS_DIR}/lgbm_calibrated.pkl")

    engine = get_engine()
    df = load_data(engine)
    _, _, test_full = temporal_split(df, dt_col="txn_ts")

    y_test = test_full["is_fraud"].to_numpy()
    calibrated_prob = pipeline.predict_proba(test_full)

    roc_auc = roc_auc_score(y_test, calibrated_prob)
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    with open(REPORT_PATH, "a", encoding="utf-8") as fh:
        fh.write("\n## ROC-AUC (context)\n\n")
        fh.write(f"- Test ROC-AUC: {roc_auc:.4f}\n")
        fh.write(
            f"- PR-AUC (test_pr_auc={metadata['test_pr_auc']:.4f}) remains the primary "
            "metric for this model because ROC-AUC is dominated by the large true-negative "
            "count under ~3.5% fraud prevalence and so overstates ranking quality on the "
            "minority (fraud) class relative to PR-AUC.\n"
        )


if __name__ == "__main__":
    main()
