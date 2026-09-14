0002 — Entity proxies and cold-start convention

Status: Accepted
Milestone: M1

Context

IEEE-CIS has no real customer ID or merchant ID column. Feature
engineering that depends on "this entity's history" (velocity, average
spend, historical fraud rate) needs _some_ grouping key, and the choice
of key is a real modeling decision with real limitations, not a neutral
technical detail.

Decision

- `card1` is used as the customer-proxy entity for velocity and
  amount-history features. It's the closest available field to a
  repeatable identity, but it's explicitly a proxy — multiple real
  people could plausibly share a `card1` value, and one real person
  could plausibly use multiple `card1` values. This limitation is
  stated here rather than glossed over.
- `device_info` is used as the device-proxy entity for historical
  fraud-rate features, with the same caveat: inconsistent device string
  reporting under-counts history for the same physical device reported
  differently across sessions.
- Cold-start convention: NULL, never a default value. When an
  entity has no prior history (`card1_hist_avg_amt`, `device_hist_fraud_rate`,
  etc.), the feature is left `NULL`, not defaulted to `0` or any other
  placeholder. `0` for a fraud-rate feature would be indistinguishable
  from "this entity has a real track record of zero fraud" — a
  meaningfully different, false claim. `NULL` lets the model (via
  imputation, decided explicitly at train time) and SHAP (M5) treat
  "unknown" as genuinely different from "known-and-zero."

Alternatives considered

- A composite proxy (e.g. `ProductCD + card4 + addr1`) for a
  merchant-like entity — considered, but rejected for this project:
  it groups by product/network/region rather than any real transacting
  counterparty, which is a coarser and arguably less meaningful segment
  than a genuine merchant ID would be. Not adopted here.
- Defaulting cold-start fraud rate to `0` — rejected for the reason
  above. Kept as NULL and handled via train-set-only median imputation
  (`fit_median_impute`) at the feature-matrix stage, which is a decision
  boundary the model can learn from rather than one baked silently into
  the SQL.

Consequences

Every entity-history feature in `features_m1.sql` uses this same NULL
convention consistently — there is no feature in this project where
cold-start silently defaults to a non-null placeholder. This
consistency is itself verified: nothing in `leakage_test.py` or
`features_m1.sql` currently violates it, unlike a mixed convention
across different entity types, which would be a real inconsistency bug.
