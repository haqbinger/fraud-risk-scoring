"""
P6: Error analysis on the tuned LightGBM model (P5 winner) at threshold 0.5.

Pulls actual false negatives (fraud missed) and false positives (legit
flagged) from the P2/P4/P5 single val split, profiles them across amount,
product/card/email categoricals, time-of-day, month, and billing-shipping
distance, and flags any bucket where the error rate is >1.5x the base rate
for that error type. Test set is loaded by temporal_split() but never
touched.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "scripts")
from feature_selection_p2 import CAT_COLS  # noqa: E402
from model_comparison_p5 import train_lgbm_with_params  # noqa: E402
from temporal_cv_p3 import load_data_splits, load_selected_features  # noqa: E402
from tuning_p4 import build_xy  # noqa: E402

REPORT_DIR = "reports/error_analysis"
LGBM_BEST_PARAMS_PATH = "reports/model_comparison/lgbm_best_params.json"
THRESHOLD = 0.5
FLAG_RATIO = 1.5
TIME_BINS = [-0.1, 6, 12, 18, 24.1]
TIME_LABELS = ["0-6", "6-12", "12-18", "18-24"]


def amount_bucket_edges(s: pd.Series):
    edges = s.quantile([0, 0.25, 0.5, 0.75, 1.0]).to_numpy().copy()
    edges[0] -= 1e-6
    return edges


def bucket_rates(df: pd.DataFrame, cond_mask: pd.Series, error_mask: pd.Series,
                  bucket_series: pd.Series, dimension: str, top_n: int = None):
    pop = bucket_series[cond_mask]
    err = bucket_series[cond_mask & error_mask]
    n_pop_total = len(pop)
    n_err_total = len(err)
    overall_rate = n_err_total / n_pop_total if n_pop_total else float("nan")

    grp_pop = pop.value_counts(dropna=False)
    grp_err = err.value_counts(dropna=False).reindex(grp_pop.index, fill_value=0)

    rows = pd.DataFrame({
        "dimension": dimension,
        "bucket": grp_pop.index.astype(str),
        "population_count": grp_pop.values,
        "error_count": grp_err.values,
    })
    rows["pct_of_errors"] = rows["error_count"] / n_err_total if n_err_total else float("nan")
    rows["rate_in_bucket"] = rows["error_count"] / rows["population_count"]
    rows["overall_rate"] = overall_rate
    rows["ratio_vs_overall"] = rows["rate_in_bucket"] / overall_rate
    rows["flagged"] = rows["ratio_vs_overall"] > FLAG_RATIO
    rows = rows.sort_values("population_count", ascending=False).reset_index(drop=True)
    if top_n:
        rows = rows.head(top_n)
    return rows, overall_rate


def amount_stats_row(dimension: str, s: pd.Series):
    return pd.DataFrame([{
        "dimension": dimension, "bucket": "summary_stats",
        "population_count": s.notna().sum(), "error_count": None,
        "pct_of_errors": None,
        "mean": s.mean(), "p25": s.quantile(0.25), "p50": s.quantile(0.5), "p75": s.quantile(0.75),
        "overall_rate": None, "ratio_vs_overall": None, "flagged": False,
    }])


def analyze_error_type(val_full: pd.DataFrame, cond_mask: pd.Series, error_mask: pd.Series,
                        amt_edges, dist1_edges) -> pd.DataFrame:
    parts = []

    err_amt = val_full.loc[cond_mask & error_mask, "transaction_amt"]
    pop_amt = val_full.loc[cond_mask, "transaction_amt"]
    parts.append(amount_stats_row("transaction_amt (errors)", err_amt))
    parts.append(amount_stats_row("transaction_amt (population)", pop_amt))
    amt_bucket = pd.cut(val_full["transaction_amt"], bins=amt_edges,
                         labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
    rows, _ = bucket_rates(val_full, cond_mask, error_mask, amt_bucket, "transaction_amt_quartile")
    parts.append(rows)

    rows, _ = bucket_rates(val_full, cond_mask, error_mask, val_full["product_cd"], "product_cd")
    parts.append(rows)

    rows, _ = bucket_rates(val_full, cond_mask, error_mask, val_full["card4"], "card4")
    parts.append(rows)

    rows, _ = bucket_rates(val_full, cond_mask, error_mask, val_full["card6"], "card6")
    parts.append(rows)

    rows, _ = bucket_rates(val_full, cond_mask, error_mask, val_full["p_emaildomain"], "p_emaildomain", top_n=10)
    parts.append(rows)

    hour_bucket = pd.cut(val_full["txn_ts"].dt.hour, bins=TIME_BINS, labels=TIME_LABELS)
    rows, _ = bucket_rates(val_full, cond_mask, error_mask, hour_bucket, "time_of_day")
    parts.append(rows)

    err_dist1 = val_full.loc[cond_mask & error_mask, "dist1"]
    pop_dist1 = val_full.loc[cond_mask, "dist1"]
    parts.append(amount_stats_row("dist1 (errors)", err_dist1))
    parts.append(amount_stats_row("dist1 (population)", pop_dist1))
    dist1_bucket = pd.cut(val_full["dist1"], bins=dist1_edges,
                           labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
    dist1_bucket = dist1_bucket.astype("object")
    dist1_bucket[val_full["dist1"].isna()] = "missing"
    rows, _ = bucket_rates(val_full, cond_mask, error_mask, dist1_bucket, "dist1_quartile")
    parts.append(rows)

    month = val_full["txn_ts"].dt.strftime("%Y-%m")
    rows, _ = bucket_rates(val_full, cond_mask, error_mask, month, "month")
    parts.append(rows)

    return pd.concat(parts, ignore_index=True)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    selected_features = load_selected_features()
    cat_cols = [c for c in CAT_COLS if c in selected_features]
    print(f"Loaded {len(selected_features)} selected features ({len(cat_cols)} categorical).")

    with open(LGBM_BEST_PARAMS_PATH, encoding="utf-8") as fh:
        lgbm_best_params = json.load(fh)

    train_full, val_full, _cv_pool = load_data_splits()
    del _cv_pool
    val_full = val_full.reset_index(drop=True)

    categories_map = {c: sorted(train_full[c].dropna().unique().tolist()) for c in cat_cols}
    X_train, y_train = build_xy(train_full, selected_features, cat_cols, categories_map)
    X_val, y_val = build_xy(val_full, selected_features, cat_cols, categories_map)

    print("Training LightGBM (P5 best params) on full train split ...")
    model = train_lgbm_with_params(X_train, y_train, lgbm_best_params)
    val_prob = model.predict_proba(X_val)[:, 1]
    val_full["pred_prob"] = val_prob
    val_full["pred"] = (val_prob >= THRESHOLD).astype(int)

    is_fraud = val_full["is_fraud"] == 1
    is_legit = val_full["is_fraud"] == 0
    fn_mask = is_fraud & (val_full["pred"] == 0)
    fp_mask = is_legit & (val_full["pred"] == 1)

    n_fraud, n_legit = is_fraud.sum(), is_legit.sum()
    n_fn, n_fp = fn_mask.sum(), fp_mask.sum()
    overall_fn_rate = n_fn / n_fraud
    overall_fp_rate = n_fp / n_legit

    print(f"\nVal set: {len(val_full)} rows, {n_fraud} actual fraud, {n_legit} actual legit")
    print(f"At threshold {THRESHOLD}: {n_fn} FN (miss rate {overall_fn_rate:.4f}), "
          f"{n_fp} FP (false-alarm rate {overall_fp_rate:.4f})")

    amt_edges = amount_bucket_edges(val_full["transaction_amt"])
    dist1_edges = amount_bucket_edges(val_full["dist1"].dropna())

    fn_analysis = analyze_error_type(val_full, is_fraud, fn_mask, amt_edges, dist1_edges)
    fp_analysis = analyze_error_type(val_full, is_legit, fp_mask, amt_edges, dist1_edges)

    fn_analysis.to_csv(f"{REPORT_DIR}/fn_analysis.csv", index=False)
    fp_analysis.to_csv(f"{REPORT_DIR}/fp_analysis.csv", index=False)

    fn_flags = fn_analysis[fn_analysis["flagged"]].sort_values("ratio_vs_overall", ascending=False)
    fp_flags = fp_analysis[fp_analysis["flagged"]].sort_values("ratio_vs_overall", ascending=False)

    print(f"\nBase rates: FN (miss) rate = {overall_fn_rate:.4f} | FP (false-alarm) rate = {overall_fp_rate:.4f}")
    print(f"\nFN blind spots (rate > {FLAG_RATIO}x the {overall_fn_rate:.4f} base miss rate):")
    print(fn_flags[["dimension", "bucket", "population_count", "error_count",
                     "rate_in_bucket", "ratio_vs_overall"]].to_string(index=False))
    print(f"\nFP blind spots (rate > {FLAG_RATIO}x the {overall_fp_rate:.4f} base false-alarm rate):")
    print(fp_flags[["dimension", "bucket", "population_count", "error_count",
                     "rate_in_bucket", "ratio_vs_overall"]].to_string(index=False))

    fn_month = fn_analysis[fn_analysis["dimension"] == "month"].sort_values("bucket")
    print("\nFN rate by month:")
    print(fn_month[["bucket", "population_count", "error_count", "rate_in_bucket"]].to_string(index=False))

    fn_by_volume = fn_analysis[
        (fn_analysis["dimension"] == "product_cd") & fn_analysis["error_count"].notna()
    ].sort_values("error_count", ascending=False)
    fn_top_ratio = fn_analysis[fn_analysis["ratio_vs_overall"].notna()].sort_values(
        "ratio_vs_overall", ascending=False
    )

    if len(fn_flags):
        fn_pattern = (f"Top FN blind spot: {fn_flags.iloc[0]['dimension']}={fn_flags.iloc[0]['bucket']} "
                      f"({fn_flags.iloc[0]['ratio_vs_overall']:.2f}x base miss rate)")
        fn_target_dim = fn_flags.iloc[0]["dimension"]
    else:
        top_vol = fn_by_volume.iloc[0]
        top_ratio = fn_top_ratio.iloc[0]
        vol_share = top_vol["error_count"] / n_fn
        fn_pattern = (
            f"No bucket cleared the {FLAG_RATIO}x flag (overall miss rate is already high at "
            f"{overall_fn_rate:.2f}), but {top_vol['dimension']}={top_vol['bucket']} accounts for "
            f"{int(top_vol['error_count'])}/{n_fn} ({vol_share:.1%}) of all FNs at a "
            f"{top_vol['rate_in_bucket']:.2f} miss rate ({top_vol['ratio_vs_overall']:.2f}x base) -- the "
            f"dominant FN segment by volume even though its rate ratio falls just under the flag line. "
            f"Closest rate-based near-miss: {top_ratio['dimension']}={top_ratio['bucket']} "
            f"({top_ratio['ratio_vs_overall']:.2f}x)."
        )
        fn_target_dim = top_vol["dimension"]

    feature_candidates = [
        {
            "error_type": "FN",
            "pattern": fn_pattern,
            "feature_candidate": (
                f"Add a {fn_target_dim}-conditioned fraud-rate/velocity feature "
                f"(e.g. expanding-window fraud rate partitioned by {fn_target_dim}) so the "
                f"model has a direct signal for this segment instead of relying on it correlating with "
                f"existing features."
            ),
        },
        {
            "error_type": "FP",
            "pattern": (f"Top FP blind spot: {fp_flags.iloc[0]['dimension']}={fp_flags.iloc[0]['bucket']} "
                        f"({fp_flags.iloc[0]['ratio_vs_overall']:.2f}x base false-alarm rate)"
                        if len(fp_flags) else "No FP bucket exceeded the 1.5x threshold"),
            "feature_candidate": (
                f"Add a {fp_flags.iloc[0]['dimension']}-conditioned historical legitimacy feature "
                f"(e.g. per-{fp_flags.iloc[0]['dimension']} expanding-window non-fraud rate or txn count) "
                f"so common-but-unusual-looking legitimate patterns in this segment stop triggering false "
                f"alarms." if len(fp_flags) else "N/A"
            ),
        },
    ]

    with open(f"{REPORT_DIR}/blind_spots.md", "w", encoding="utf-8") as fh:
        fh.write("# P6 Blind Spots\n\n")
        fh.write(f"Threshold: {THRESHOLD}. Flag rule: bucket rate > {FLAG_RATIO}x the base rate "
                  f"(FN rate is conditioned on actual fraud, FP rate on actual legit).\n\n")
        fh.write(f"Base FN (miss) rate: {overall_fn_rate:.4f}\n")
        fh.write(f"Base FP (false-alarm) rate: {overall_fp_rate:.4f}\n\n")
        fh.write("## FN blind spots\n\n")
        if len(fn_flags):
            fh.write("| dimension | bucket | population | errors | rate | ratio vs base |\n|---|---|---|---|---|---|\n")
            for _, r in fn_flags.iterrows():
                fh.write(f"| {r['dimension']} | {r['bucket']} | {r['population_count']} | {r['error_count']} | "
                          f"{r['rate_in_bucket']:.4f} | {r['ratio_vs_overall']:.2f}x |\n")
        else:
            fh.write(f"No bucket exceeded {FLAG_RATIO}x the base miss rate ({overall_fn_rate:.4f}) -- see the "
                      f"volume-dominant and near-miss findings in the feature candidates section below.\n")
        fh.write("\n## FP blind spots\n\n")
        fh.write("| dimension | bucket | population | errors | rate | ratio vs base |\n|---|---|---|---|---|---|\n")
        for _, r in fp_flags.iterrows():
            fh.write(f"| {r['dimension']} | {r['bucket']} | {r['population_count']} | {r['error_count']} | "
                      f"{r['rate_in_bucket']:.4f} | {r['ratio_vs_overall']:.2f}x |\n")
        fh.write("\n## Feature candidates\n\n")
        for fc in feature_candidates:
            fh.write(f"- **{fc['error_type']}**: {fc['pattern']}\n  - Candidate: {fc['feature_candidate']}\n")

    summary_lines = [
        "# P6 Error Analysis Summary",
        "",
        f"- Model: LightGBM (P5 winner), params from {LGBM_BEST_PARAMS_PATH}",
        f"- Threshold: {THRESHOLD} (unoptimized -- P7 will tune this)",
        f"- Val set: {len(val_full)} rows, {n_fraud} fraud, {n_legit} legit",
        f"- FN: {n_fn} (miss rate {overall_fn_rate:.4f})",
        f"- FP: {n_fp} (false-alarm rate {overall_fp_rate:.4f})",
        "",
        "## FN rate by month",
        "",
        "| month | population | FN | FN rate |",
        "|---|---|---|---|",
    ] + [
        f"| {r.bucket} | {r.population_count} | {r.error_count} | {r.rate_in_bucket:.4f} |"
        for r in fn_month.itertuples()
    ] + [
        "",
        "## Feature candidates",
        "",
    ] + [
        f"- **{fc['error_type']}** -- found: {fc['pattern']}; did: {fc['feature_candidate']}"
        for fc in feature_candidates
    ] + [
        "",
        "## Feed back into P1",
        "",
        "If the blind spots above point at a categorical or velocity feature not yet in features_m1 "
        "(e.g. a per-product_cd or per-card4 expanding-window fraud rate), it belongs in P1's target/frequency "
        "encoding pass, then P2-P5 should be re-run on the expanded feature set to confirm it survives "
        "selection and moves CV PR-AUC.",
    ]
    with open(f"{REPORT_DIR}/p6_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    print("\n" + "=" * 60)
    print("P6 SUMMARY")
    print("=" * 60)
    print(f"FN: {n_fn} (miss rate {overall_fn_rate:.4f})  |  FP: {n_fp} (false-alarm rate {overall_fp_rate:.4f})")
    print("\nFeature candidates:")
    for fc in feature_candidates:
        print(f"  [{fc['error_type']}] found: {fc['pattern']}")
        print(f"       did: {fc['feature_candidate']}")
    print(f"\nSaved: {REPORT_DIR}/fn_analysis.csv")
    print(f"Saved: {REPORT_DIR}/fp_analysis.csv")
    print(f"Saved: {REPORT_DIR}/blind_spots.md")
    print(f"Saved: {REPORT_DIR}/p6_summary.md")


if __name__ == "__main__":
    main()
