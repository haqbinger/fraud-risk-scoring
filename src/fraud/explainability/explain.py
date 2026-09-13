import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap

from src.calibration.calibrate import load_winning_model
from fraud.data import get_engine, load_data
from fraud.evaluations.splits import temporal_split
from fraud.features import build_feature_matrix, fit_median_impute

ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fname: str):
    path = FIGURES_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def _shap_values_2d(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            return np.asarray(shap_values[0])
        return np.asarray(shap_values)
    return np.asarray(shap_values)


def run():
    engine = get_engine()
    df = load_data(engine)
    train_df, _, test_df = temporal_split(df, dt_col="txn_ts")

    X_train, y_train, category_maps = build_feature_matrix(train_df)
    X_test, y_test, _ = build_feature_matrix(test_df, category_maps=category_maps)

    medians = X_train.median(numeric_only=True)
    X_train = fit_median_impute(X_train, X_train, medians)
    X_test = fit_median_impute(X_train, X_test, medians)

    model = load_winning_model()

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(3000, len(X_test)), replace=False)
    X_shap = X_test.iloc[sample_idx]

    print("Computing SHAP values (sampled test set) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    shap_values_2d = _shap_values_2d(shap_values)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.ravel(expected_value)[0])

    plt.figure()
    shap.summary_plot(shap_values_2d, X_shap, show=False)
    plt.tight_layout()
    _save("shap_global_summary.png")

    mean_abs = np.abs(shap_values_2d).mean(axis=0)
    importance = sorted(zip(X_shap.columns, mean_abs), key=lambda kv: -kv[1])
    print("\n=== Global importance (mean |SHAP|, top 15) ===")
    for name, val in importance[:15]:
        print(f"  {name:<45} {val:.5f}")

    y_prob_full = model.predict_proba(X_test)[:, 1]
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    sample_prob = y_prob_full[sample_idx]
    sample_y = y_test_arr[sample_idx]

    examples = {
        "true_positive": np.where((sample_prob > 0.8) & (sample_y == 1))[0],
        "false_positive": np.where((sample_prob > 0.8) & (sample_y == 0))[0],
        "borderline": np.where((sample_prob > 0.40) & (sample_prob < 0.60))[0],
    }

    print("\n=== Local explanations ===")
    for name, candidates in examples.items():
        if len(candidates) == 0:
            print(f"  [{name}] No example found -- loosen the probability window")
            continue

        local_idx = int(candidates[0])
        full_idx = int(sample_idx[local_idx])

        explanation = shap.Explanation(
            values=shap_values_2d[local_idx],
            base_values=expected_value,
            data=X_shap.iloc[local_idx],
            feature_names=X_shap.columns.tolist(),
        )

        plt.figure()
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        _save(f"shap_local_{name}.png")

        top3 = sorted(
            zip(X_shap.columns, shap_values_2d[local_idx]),
            key=lambda kv: -abs(kv[1]),
        )[:3]

        print(
            f"  [{name}] full_test_idx={full_idx} "
            f"fraud_prob={sample_prob[local_idx]:.3f} "
            f"actual={sample_y[local_idx]}"
        )
        print(f"    top-3 contributors: {[(n, round(v, 4)) for n, v in top3]}")

    return {
        "global_importance": importance[:15],
        "local_examples": {
            "true_positive": {
                "full_idx": int(
                    sample_idx[int(np.where((sample_prob > 0.8) & (sample_y == 1))[0][0])]
                ),
                "probability": float(
                    sample_prob[np.where((sample_prob > 0.8) & (sample_y == 1))[0][0]]
                ),
                "actual": int(
                    sample_y[np.where((sample_prob > 0.8) & (sample_y == 1))[0][0]]
                ),
            },
            "false_positive": {
                "full_idx": int(
                    sample_idx[int(np.where((sample_prob > 0.8) & (sample_y == 0))[0][0])]
                ),
                "probability": float(
                    sample_prob[np.where((sample_prob > 0.8) & (sample_y == 0))[0][0]]
                ),
                "actual": int(
                    sample_y[np.where((sample_prob > 0.8) & (sample_y == 0))[0][0]]
                ),
            },
            "borderline": {
                "full_idx": int(
                    sample_idx[int(np.where((sample_prob > 0.40) & (sample_prob < 0.60))[0][0])]
                ),
                "probability": float(
                    sample_prob[np.where((sample_prob > 0.40) & (sample_prob < 0.60))[0][0]]
                ),
                "actual": int(
                    sample_y[np.where((sample_prob > 0.40) & (sample_prob < 0.60))[0][0]]
                ),
            },
        },
    }


if __name__ == "__main__":
    run()