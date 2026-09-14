P6 Blind Spots

Threshold: 0.5. Flag rule: bucket rate > 1.5x the base rate (FN rate is conditioned on actual fraud, FP rate on actual legit).

Base FN (miss) rate: 0.4579
Base FP (false-alarm) rate: 0.0055

FN blind spots

No bucket exceeded 1.5x the base miss rate (0.4579) -- see the volume-dominant and near-miss findings in the feature candidates section below.

FP blind spots

| dimension | bucket | population | errors | rate | ratio vs base |
|---|---|---|---|---|---|
| card4 | discover | 841 | 24 | 0.0285 | 5.22x |
| product_cd | C | 7975 | 224 | 0.0281 | 5.13x |
| p_emaildomain | comcast.net | 1059 | 24 | 0.0227 | 4.14x |
| product_cd | H | 2210 | 42 | 0.0190 | 3.47x |
| p_emaildomain | outlook.com | 682 | 12 | 0.0176 | 3.22x |
| card6 | credit | 17422 | 265 | 0.0152 | 2.78x |
| p_emaildomain | hotmail.com | 6054 | 85 | 0.0140 | 2.57x |
| transaction_amt_quartile | Q4 | 20888 | 190 | 0.0091 | 1.66x |
| product_cd | R | 3205 | 29 | 0.0090 | 1.65x |
| time_of_day | 0-6 | 19965 | 174 | 0.0087 | 1.59x |
| dist1_quartile | missing | 45358 | 392 | 0.0086 | 1.58x |

Feature candidates

- FN: No bucket cleared the 1.5x flag (overall miss rate is already high at 0.46), but product_cd=W accounts for 981/1393 (70.4%) of all FNs at a 0.66 miss rate (1.44x base) -- the dominant FN segment by volume even though its rate ratio falls just under the flag line. Closest rate-based near-miss: p_emaildomain=icloud.com (1.49x).
  - Candidate: Add a product_cd-conditioned fraud-rate/velocity feature (e.g. expanding-window fraud rate partitioned by product_cd) so the model has a direct signal for this segment instead of relying on it correlating with existing features.
- FP: Top FP blind spot: card4=discover (5.22x base false-alarm rate)
  - Candidate: Add a card4-conditioned historical legitimacy feature (e.g. per-card4 expanding-window non-fraud rate or txn count) so common-but-unusual-looking legitimate patterns in this segment stop triggering false alarms.
