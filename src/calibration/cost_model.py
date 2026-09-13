import numpy as np
import pandas as pd


def expected_cost(y_true, y_prob, threshold: float,
                 cost_fn: float, cost_fp: float) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true)

    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    total_cost = fn * cost_fn + fp * cost_fp
    return {
        "threshold": threshold,
        "fn": fn,
        "fp": fp,
        "tp": tp,
        "tn": tn,
        "total_cost": total_cost,
    }


def sweep_thresholds(y_true, y_prob, cost_fn: float, cost_fp: float,
                    n_steps: int = 199, min_threshold: float = 0.001,
                    max_threshold: float = 0.99) -> pd.DataFrame:
    thresholds = np.linspace(min_threshold, max_threshold, n_steps)
    rows = [expected_cost(y_true, y_prob, t, cost_fn, cost_fp) for t in thresholds]
    return pd.DataFrame(rows)


def find_optimal_threshold(y_true, y_prob, cost_fn: float, cost_fp: float) -> dict:
    df = sweep_thresholds(y_true, y_prob, cost_fn, cost_fp)
    return df.loc[df["total_cost"].idxmin()].to_dict()


def sensitivity_analysis(y_true, y_prob, cost_fn_values: list, cost_fp: float = 5.0) -> pd.DataFrame:
    rows = []
    for cost_fn in cost_fn_values:
        best = find_optimal_threshold(y_true, y_prob, cost_fn, cost_fp)
        rows.append({
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
            "optimal_threshold": best["threshold"],
            "total_cost": best["total_cost"],
            "fn": best["fn"],
            "fp": best["fp"],
        })
    return pd.DataFrame(rows)