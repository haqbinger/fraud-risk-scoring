"""
P8: SHAP explainability on the final calibrated LightGBM model (P7).

SHAP runs on the raw (uncalibrated) LightGBM output, not the calibrated
probabilities -- same decision as V1 ADR 0007: calibration is a monotonic
post-hoc transform of the score, SHAP should attribute the model's actual
learned splits, not the calibration layer on top of them.

Val set only (loaded via load_data() + temporal_split(), test discarded).
"""
import os
import re
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

sys.path.insert(0, "scripts")  # noqa: E402

from fraud.data import get_engine, load_data  # noqa: E402
from fraud.evaluations.splits import temporal_split  # noqa: E402

FIG_DIR = "reports/figures"
PIPELINE_PATH = "models/lgbm_calibrated.pkl"
TOP_N_GLOBAL = 30
SHAP_SAMPLE_SIZE = 3000
RANDOM_STATE = 42
BORDERLINE_LOW, BORDERLINE_HIGH = 0.45, 0.55


def shap_values_2d(sv):
    if isinstance(sv, list):
        return np.asarray(sv[1] if len(sv) == 2 else sv[0])
    arr = np.asarray(sv)
    if arr.ndim == 3:
        return arr[:, :, -1]
    return arr


def save_fig(fname: str):
    path = f"{FIG_DIR}/{fname}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def interpret_feature(name: str) -> str:
    if name.endswith("_te_fraud_rate") or name == "device_hist_fraud_rate":
        who = name.replace("_te_fraud_rate", "").replace("device_hist_fraud_rate", "device")
        return f"Expanding-window historical fraud rate for {who} -- direct target-encoded risk signal."
    if name.endswith("_hist_legit_rate"):
        return "Expanding-window historical legitimacy rate -- direct target-encoded trust signal."
    if name.endswith("_txn_count") or name.endswith("_hist_txn_count"):
        return "Historical transaction count for this entity -- a maturity/volume proxy (new vs. established)."
    if name.endswith("_freq"):
        return "Full-table frequency count -- common vs. rare category signal."
    if name.startswith("velocity_"):
        return "Transaction count in a trailing time window for this card -- burst/velocity fraud signal."
    if name == "amount_zscore_vs_card1_history":
        return "How unusual this amount is vs. this card's own historical spending pattern."
    if name == "seconds_since_card1_prev_txn":
        return "Time since this card's previous transaction -- rapid repeat use is a fraud signal."
    if name == "transaction_amt":
        return "Raw transaction amount."
    if name in ("card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"):
        return "Raw numeric identifier/distance field from the payment processor."
    if name in ("card4", "card6", "product_cd", "p_emaildomain", "device_type"):
        return "Raw categorical field (card network/type, product, email domain, or device)."
    if re.match(r"^v_block\d+_missing$", name):
        return "Missingness indicator for a V-column block -- structured missingness by payment processor."
    if re.match(r"^V\d+$", name):
        return "Anonymized Vesta-engineered feature (identity/behavior signal, meaning undisclosed by source data)."
    if re.match(r"^C\d+$", name):
        return "Anonymized count feature (e.g. addresses/emails associated with this card)."
    if re.match(r"^D\d+$", name):
        return "Anonymized time-delta feature (days since some reference event)."
    if re.match(r"^M\d+$", name):
        return "Anonymized match flag (e.g. name/address match between card and billing)."
    return "Engineered feature."


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    pipeline = joblib.load(PIPELINE_PATH)
    print(f"Loaded pipeline: {len(pipeline.selected_features)} features, "
          f"calibrator={pipeline.calibrator_type}, threshold={pipeline.threshold}")

    engine = get_engine()
    df = load_data(engine)
    _train_full, val_full, _test_full = temporal_split(df, dt_col="txn_ts")
    del _train_full, _test_full
    val_full = val_full.reset_index(drop=True)

    X_val = pipeline._to_model_matrix(val_full)
    y_val = val_full["is_fraud"].values

    raw_prob = pipeline.model.predict_proba(X_val)[:, 1]
    calibrated_prob = pipeline.predict_proba(val_full)
    pred = (calibrated_prob >= pipeline.threshold).astype(int)

    tp_idx = np.where((y_val == 1) & (pred == 1))[0]
    fp_idx = np.where((y_val == 0) & (pred == 1))[0]
    fn_idx = np.where((y_val == 1) & (pred == 0))[0]
    border_idx = np.where((calibrated_prob >= BORDERLINE_LOW) & (calibrated_prob <= BORDERLINE_HIGH))[0]

    tp_top = tp_idx[np.argsort(-calibrated_prob[tp_idx])][:3]
    fp_top = fp_idx[np.argsort(-calibrated_prob[fp_idx])][:3]
    fn_top = fn_idx[np.argsort(calibrated_prob[fn_idx])][:3]
    border_top = border_idx[np.argsort(np.abs(calibrated_prob[border_idx] - 0.5))][:3]

    print(f"Local example pools: TP={len(tp_idx)}  FP={len(fp_idx)}  FN={len(fn_idx)}  "
          f"borderline={len(border_idx)}")

    rng = np.random.default_rng(RANDOM_STATE)
    global_sample_idx = rng.choice(len(X_val), size=min(SHAP_SAMPLE_SIZE, len(X_val)), replace=False)
    local_idx_all = np.concatenate([tp_top, fp_top, fn_top, border_top])
    shap_compute_idx = np.unique(np.concatenate([global_sample_idx, local_idx_all]))

    X_shap = X_val.iloc[shap_compute_idx].reset_index(drop=True)
    idx_map = {int(orig): pos for pos, orig in enumerate(shap_compute_idx)}

    print(f"Computing SHAP values for {len(X_shap)} rows via TreeExplainer ...")
    explainer = shap.TreeExplainer(pipeline.model)
    shap_values = explainer.shap_values(X_shap)
    shap_2d = shap_values_2d(shap_values)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.ravel(expected_value)[-1])

    global_pos = np.array([idx_map[int(i)] for i in global_sample_idx])
    shap_global = shap_2d[global_pos]
    X_global = X_shap.iloc[global_pos]

    plt.figure()
    shap.summary_plot(shap_global, X_global, max_display=TOP_N_GLOBAL, show=False)
    plt.tight_layout()
    save_fig("p8_shap_global_summary.png")

    plt.figure()
    shap.summary_plot(shap_global, X_global, plot_type="bar", max_display=TOP_N_GLOBAL, show=False)
    plt.tight_layout()
    save_fig("p8_shap_importance.png")

    mean_abs = np.abs(shap_global).mean(axis=0)
    importance = sorted(zip(X_global.columns, mean_abs), key=lambda kv: -kv[1])

    print("\nTop 10 features by mean |SHAP|:")
    for name, val in importance[:10]:
        print(f"  {name:<40} {val:.5f}")

    groups = {"tp": tp_top, "fp": fp_top, "fn": fn_top, "borderline": border_top}
    local_records = {g: [] for g in groups}
    for group_name, idxs in groups.items():
        for i, orig_idx in enumerate(idxs, start=1):
            pos = idx_map[int(orig_idx)]
            explanation = shap.Explanation(
                values=shap_2d[pos],
                base_values=expected_value,
                data=X_shap.iloc[pos],
                feature_names=X_shap.columns.tolist(),
            )
            plt.figure()
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            save_fig(f"p8_waterfall_{group_name}_{i}.png")

            top3 = sorted(zip(X_shap.columns, shap_2d[pos]), key=lambda kv: -abs(kv[1]))[:3]
            record = {
                "val_row_idx": int(orig_idx), "raw_prob": float(raw_prob[orig_idx]),
                "calibrated_prob": float(calibrated_prob[orig_idx]), "actual": int(y_val[orig_idx]),
                "top3": top3,
            }
            local_records[group_name].append(record)
            print(f"[{group_name} {i}] idx={orig_idx} calibrated_prob={calibrated_prob[orig_idx]:.4f} "
                  f"actual={y_val[orig_idx]}  top3: {[(n, round(v, 4)) for n, v in top3]}")

    def top3_freq(group_name):
        counts = {}
        for r in local_records[group_name]:
            for n, _ in r["top3"]:
                counts[n] = counts.get(n, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    summary_lines = [
        "# P8 SHAP Explainability Summary",
        "",
        f"- Model: LightGBM (P5/P7 final), raw (uncalibrated) output used for SHAP per V1 ADR 0007",
        f"- Explainer: TreeExplainer, {len(X_shap)} rows ({SHAP_SAMPLE_SIZE} random + 12 local examples)",
        f"- Operational threshold: {pipeline.threshold:.3f}",
        "",
        "## Top 10 features by mean |SHAP|",
        "",
        "| Rank | Feature | Mean \\|SHAP\\| | Interpretation |",
        "|---|---|---|---|",
    ] + [
        f"| {i} | `{name}` | {val:.5f} | {interpret_feature(name)} |"
        for i, (name, val) in enumerate(importance[:10], start=1)
    ] + [
        "",
        "## Local explanation patterns",
        "",
    ]

    for group_name, label in [("tp", "True positives"), ("fp", "False positives"),
                               ("fn", "False negatives"), ("borderline", "Borderline (0.45-0.55)")]:
        recs = local_records[group_name]
        summary_lines.append(f"### {label}")
        summary_lines.append("")
        for i, r in enumerate(recs, start=1):
            top3_str = ", ".join(f"{n} ({v:+.4f})" for n, v in r["top3"])
            summary_lines.append(
                f"- Example {i}: val_row={r['val_row_idx']}, calibrated_prob={r['calibrated_prob']:.4f}, "
                f"actual={r['actual']} -- top drivers: {top3_str}"
            )
        freq = top3_freq(group_name)
        if freq:
            summary_lines.append(f"- Recurring top-3 drivers across these 3 examples: "
                                  f"{', '.join(f'{n} (x{c})' for n, c in freq[:5])}")
        summary_lines.append("")

    fn_recs = local_records["fn"]
    fp_recs = local_records["fp"]
    fn_note = (
        "FN waterfalls show the model's top drivers pushing *toward* legitimate for these missed frauds -- "
        "worth checking whether the recurring drivers above overlap with known FN blind spots from P6 "
        "(product_cd=W dominance, April temporal drift) even though the dedicated product_cd/card4 features "
        "tested worse in the P6 patch test."
        if fn_recs else "No FN examples found in the sampled pool -- widen the sample or check the threshold."
    )
    fp_note = (
        "FP waterfalls show which features most strongly (and wrongly) pushed toward fraud for these false "
        "alarms -- compare against the P6 card4=discover / product_cd=C blind spots."
        if fp_recs else "No FP examples found in the sampled pool."
    )
    summary_lines += [
        "## Notes",
        "",
        f"- {fn_note}",
        f"- {fp_note}",
    ]

    with open(f"{FIG_DIR}/p8_shap_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"\nSaved: {FIG_DIR}/p8_shap_summary.md")


if __name__ == "__main__":
    main()
