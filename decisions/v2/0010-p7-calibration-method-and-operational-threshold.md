0010 — P7 Calibration Method and Operational Threshold

Status: Accepted
Date: 2026-09-14
Supersedes/relates to: ADR 0008 (P6 patch recommendation — product_cd/card4 features — deferred, see Consequences)

Context

P7 required selecting a probability calibration method for the LightGBM model
and setting an operational review threshold under a 5% review-rate cap.

Two calibration methods were evaluated out-of-fold:

- Platt scaling: Brier 0.0181
- Isotonic regression: Brier 0.0174

The gap (0.0007) fell inside the pre-registered tie-break threshold of 0.002,
meaning the two methods are not meaningfully distinguishable on this metric
at this sample size.

Decision

Platt scaling was selected per the tie-break rule (prefer the simpler,
parametric method when the performance gap is within the tie threshold).
Isotonic's typical failure mode — overfitting in sparse regions of the score
distribution — is a real risk avoided by this choice, even though it wasn't
the deciding factor here (the tie-break rule was).

Operational threshold set to 0.036 (vs. V1's 0.141), calibrated to hit
the 5% review-rate cap. This threshold is far lower than V1's in absolute
terms because Platt-calibrated LightGBM scores cluster tightly near the true
~3.5% fraud base rate — meaningful separation between fraud/non-fraud occurs
at much lower absolute scores than V1's less-calibrated XGBoost produced.
The threshold's low absolute value is a property of calibration quality, not
evidence of a weaker model.

Consequences

Test set (opened once) — V1 vs V2:

| Metric                     | V1         | V2       | Delta     |
| -------------------------- | ---------- | -------- | --------- |
| PR-AUC                     | 0.2189     | 0.5941   | +0.3752   |
| Brier                      | 0.0306     | 0.0209   | −0.0097   |
| Threshold                  | 0.141      | 0.036    | −0.105    |
| Cost savings vs. naive 0.5 | $1,532,200 | $833,850 | −$698,350 |

At threshold 0.036, V2 operates at 64.4% recall / 44.1% precision.

Important caveat on the cost-savings delta: the drop in dollar savings
vs. V1 does not indicate a worse model. The naive-0.5 baseline is not a
fixed reference — it moves with the model's calibration. V2's
well-calibrated scores rarely exceed 0.5 at all (true base rate ~3.5%), so
V2's naive-0.5 baseline already avoids most of the false positives that
V1's poorly-calibrated naive baseline made. The comparison baseline improved
alongside the model, compressing the visible savings gap despite PR-AUC
nearly tripling and Brier improving 32%. The 5%-review-rate operating point
(64.4% recall / 44.1% precision) is the more honest, load-bearing comparison
than the naive-0.5 dollar figure.

Outstanding: ADR 0008's recommended P6 patch (add product_cd/card4
features) was not applied before this P7 run, per explicit direction to
proceed directly to P7. Still outstanding for P8 consideration.

Artifacts:

- `reports/calibration/threshold_cost_curve.png`
- `reports/calibration/reliability_diagram.png`
- `reports/calibration/p7_summary.md`
- `models/lgbm_calibrated.pkl` (self-contained `CalibratedLGBMPipeline`, verified load/predict)
- `models/model_metadata.json`
