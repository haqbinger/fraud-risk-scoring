# ADR V2-0001: P0 EDA Findings

## Status

Accepted

## Context

V1 used 20 of 394 available features (PR-AUC 0.22). P0 EDA audited raw_transaction to understand what was unused and why.

## Findings

- 394 total columns: 339 V-cols, 14 categoricals, 39 non-V numerics
- V-col missingness is block-structured (9 blocks by null rate), not random — V138-166 cluster at ~86.1% null, indicating a shared upstream source field
- R_emaildomain (60 values, 77% null) and P_emaildomain (59 values, 16% null) are the only high-cardinality categoricals; all others have ≤5 distinct values
- Top univariate signal (AUC proxy 0.65-0.75): D2, D3, D5, D8 and several V/C columns — none of these were in the V1 feature set
- transactions table had only 12 columns; raw_transaction has 394

## Decision

Proceed to P1: rebuild transactions table and extend features_m1.sql to include all V/C/D/M columns, block missingness flags, frequency encoding for email domains, and target encoding for card1/addr1/p_emaildomain.
