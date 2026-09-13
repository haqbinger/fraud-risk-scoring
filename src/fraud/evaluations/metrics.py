import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, f1_score, brier_score_loss,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)   # this is PR-AUC
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,            # primary ranking metric, see ADR 0003
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier,        # lower is better; revisited properly in M4
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def precision_at_recall(y_true, y_prob, target_recall: float = 0.80) -> float:
    """Business-readable: 'at 80% fraud caught, what's our precision?'"""
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    valid = recalls >= target_recall
    if not valid.any():
        return float("nan")
    return float(precisions[valid].max())


def print_metrics(name: str, m: dict):
    print(f"\n--- {name} ---")
    print(f"ROC-AUC:  {m['roc_auc']:.4f}")
    print(f"PR-AUC:   {m['pr_auc']:.4f}  (primary metric — see decisions/0003)")
    print(f"Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}")
    print(f"Brier score: {m['brier_score']:.4f}")
    print(f"Confusion matrix: {m['confusion_matrix']}")