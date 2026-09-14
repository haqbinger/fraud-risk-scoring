ADR V2-0004: P1 Patch — Missing Columns

Status

Accepted

Context

P2 found card2-6, addr2, dist1, dist2 existed in transactions but never reached the feature matrix.

Changes

- Added card2, card3, card4, card5, card6, addr2, dist1, dist2 as raw pass-throughs in features_m1.sql (418 cols, 590,540 rows)
- R_emaildomain frequency encoding was already present from original P1
- load_data() required no query change — f.\* absorbs new columns automatically

Results

|                      | Before | After                                         |
| -------------------- | ------ | --------------------------------------------- |
| Kitchen-sink ceiling | 0.5725 | 0.5796                                        |
| Delta                | —      | +0.0071                                       |
| Verdict              | —      | Exceeded 0.005 threshold — P2 rerun triggered |

P2 Rerun Result

- Features: 419 in → 182 selected (237 dropped)
- Reduced-set val PR-AUC: 0.5888 (beats ceiling — dropped features were adding noise)
- All 8 new columns survived selection
- reports/feature_selection/\* overwritten with expanded-set results

Decision

182-feature set at 0.5888 val PR-AUC is the new baseline. Proceed to P3 temporal CV.
