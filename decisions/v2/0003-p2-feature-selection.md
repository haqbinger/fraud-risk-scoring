ADR V2-0003: P2 Feature Selection

Status

Accepted

Context

P1 produced a 415-column feature matrix. Need to reduce to a clean, justified set without losing PR-AUC.

Method

1. Kitchen-sink XGBoost (all 415 features, temporal split, default hyperparams) — val PR-AUC as ceiling
2. Permutation importance (n_repeats=10, val set) alongside gain importance
3. Disagreement analysis: features where gain rank vs permutation rank differ by >50 positions flagged as correlation proxies
4. Drop rules (any one sufficient):
   - Permutation importance mean < 0
   - Gain importance < 0.0001
   - Spearman correlation > 0.95 with a higher-ranked feature (computed on train set only)
5. Retrain on reduced set — must stay within 0.01 PR-AUC of ceiling

Results

| Metric                            | Value                     |
| --------------------------------- | ------------------------- |
| Kitchen-sink val PR-AUC (ceiling) | 0.5725 (411 features)     |
| Reduced-set val PR-AUC            | 0.5691 (182 features)     |
| Delta                             | 0.0034 — within tolerance |
| Features kept / dropped           | 182 / 229                 |

Drop breakdown:

- 95 near-zero gain (< 0.0001)
- 81 negative permutation importance (actively hurt val performance)
- 53 redundant (Spearman > 0.95 with higher-ranked feature)

Key Finding

297 of 411 features disagreed by >50 rank positions between gain and permutation importance. Several V-columns (V13, V217, V233, V301 etc.) ranked top-60 by gain but bottom-15 by permutation with negative permutation importance — classic symptom of features a tree splits on for structural/correlation reasons that don't generalize. All dropped.

Known Gap

card2-6, addr2, dist1/dist2, R_emaildomain were never piped through features_m1.sql into load_data(). These exist in transactions but are absent from the current feature matrix. dist1/dist2 (billing/shipping distance) is a known fraud signal — worth a P1 patch before P3.

Decision

Proceed with 182-feature set. Investigate missing columns (card2-6, addr2, dist1/2, R_emaildomain) via quick P1 patch before temporal CV. If ceiling doesn't move, drop and proceed to P3.

Artifacts

- reports/feature_selection/gain_importance.csv
- reports/feature_selection/permutation_importance.csv
- reports/feature_selection/disagreement_analysis.csv
- reports/feature_selection/selected_features.txt (182 features)
- reports/feature_selection/selection_summary.md
