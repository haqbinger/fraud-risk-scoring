import matplotlib.pyplot as plt
from pathlib import Path

from fraud.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    MAX_REVIEW_RATE,
)
from src.calibration.calibrate import run as run_calibration
from src.calibration.cost_model import (
    find_optimal_threshold,
    sensitivity_analysis,
    sweep_thresholds,
)

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures"
DATA_DIR = ROOT / "data" / "processed"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def choose_best_calibration(cal_results):
    baseline_pr = cal_results["uncalibrated"][1]["pr_auc"]
    candidate_methods = ["uncalibrated", "platt", "isotonic"]

    valid = {}
    for method in candidate_methods:
        pr_auc = cal_results[method][1]["pr_auc"]
        if pr_auc >= baseline_pr * 0.99:
            valid[method] = cal_results[method][1]["brier_score"]

    if not valid:
        raise ValueError(
            "No calibration method preserved PR-AUC within the acceptable tolerance. "
            "Check calibration logic and cost assumptions."
        )

    return min(valid, key=valid.get)


def run():
    cal_results = run_calibration()

    best_method = choose_best_calibration(cal_results)
    print(f"\nBest-calibrated method by PR-AUC: {best_method}")
    y_prob = cal_results[best_method][0]
    y_test = cal_results["y_test"]

    print("\n" + "=" * 60)
    print(
        f"COST-SENSITIVE THRESHOLD SWEEP "
        f"(cost_fn=${COST_FALSE_NEGATIVE}, cost_fp=${COST_FALSE_POSITIVE})"
    )
    print("=" * 60)

    sweep_df = sweep_thresholds(
        y_test,
        y_prob,
        COST_FALSE_NEGATIVE,
        COST_FALSE_POSITIVE,
    )
    optimal = find_optimal_threshold(
        y_test,
        y_prob,
        COST_FALSE_NEGATIVE,
        COST_FALSE_POSITIVE,
    )

    review_rate = float((y_prob >= optimal["threshold"]).mean())
    if review_rate > MAX_REVIEW_RATE:
        print(
            f"Warning: review rate {review_rate:.1%} exceeds the operational cap "
            f"of {MAX_REVIEW_RATE:.1%}."
        )

    if optimal["threshold"] <= 0.001 or optimal["threshold"] >= 0.99:
        print(
            "Warning: optimal threshold is at the search boundary; "
            "review the cost model or add a stronger operational constraint."
        )

    review_rates = [(y_prob >= threshold).mean() for threshold in sweep_df["threshold"]]
    sweep_df = sweep_df.assign(review_rate=review_rates)

    feasible = sweep_df[sweep_df["review_rate"] <= MAX_REVIEW_RATE].copy()
    if feasible.empty:
        raise ValueError("No feasible threshold under the review-rate cap.")

    best_feasible = feasible.loc[feasible["total_cost"].idxmin()].to_dict()

    print(f"Operational threshold under review cap: {best_feasible['threshold']:.3f}")
    print(f"Review rate: {best_feasible['review_rate']:.1%}")
    print(f"FN={int(best_feasible['fn'])}, FP={int(best_feasible['fp'])}, total cost=${best_feasible['total_cost']:.2f}")

    naive_pred = (y_prob >= 0.5).astype(int)
    naive_fn = int(((naive_pred == 0) & (y_test == 1)).sum())
    naive_fp = int(((naive_pred == 1) & (y_test == 0)).sum())
    naive_total = naive_fn * COST_FALSE_NEGATIVE + naive_fp * COST_FALSE_POSITIVE
    print(f"\nNaive threshold=0.5: total cost=${naive_total:,.2f}")
    print(f"Cost-aware threshold saves: ${naive_total - best_feasible['total_cost']:,.2f} vs. naive 0.5")

    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["threshold"], sweep_df["total_cost"], label="Cost sweep")
    plt.axvline(optimal["threshold"], color="red", linestyle="--", label="Raw optimum")
    plt.axvline(best_feasible["threshold"], color="blue", linestyle="-.", label="Operational optimum")
    plt.axvline(0.5, color="gray", linestyle=":", label="Naive 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Total expected cost ($)")
    plt.title("Cost-sensitive threshold sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "threshold_cost_curve.png")
    print(f"\nSaved threshold curve to {FIG_DIR / 'threshold_cost_curve.png'}")

    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)
    sensitivity_df = sensitivity_analysis(
        y_test,
        y_prob,
        cost_fn_values=[50, 100, 250, 500, 1000, 2000],
        cost_fp=COST_FALSE_POSITIVE,
    )
    print(sensitivity_df.to_string(index=False))
    sensitivity_df.to_csv(DATA_DIR / "threshold_sensitivity.csv", index=False)
    print(f"\nSaved sensitivity table to {DATA_DIR / 'threshold_sensitivity.csv'}")

    return best_feasible, sensitivity_df


if __name__ == "__main__":
    run()