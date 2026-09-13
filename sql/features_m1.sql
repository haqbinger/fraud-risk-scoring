DROP TABLE IF EXISTS features_m1;

-- V-column missingness blocks (from P0 EDA, grouped by shared null-rate --
-- these columns come from the same upstream payment-processor field and
-- are missing together, so one flag per block instead of per column):
--   block 1: V1-V11
--   block 2: V12-V34
--   block 3: V35-V52
--   block 4: V53-V94
--   block 5: V95-V137
--   block 6: V138-V166
--   block 7: V167-V278
--   block 8: V279-V321
--   block 9: V322-V339

CREATE TABLE features_m1 AS
WITH base AS (
    SELECT *
    FROM transactions
),
velocity AS (
    SELECT
        transaction_id,
        COUNT(*) OVER (
            PARTITION BY card1 ORDER BY txn_ts
            RANGE BETWEEN INTERVAL '5 minutes' PRECEDING AND INTERVAL '1 second' PRECEDING
        ) AS velocity_5min,
        COUNT(*) OVER (
            PARTITION BY card1 ORDER BY txn_ts
            RANGE BETWEEN INTERVAL '30 minutes' PRECEDING AND INTERVAL '1 second' PRECEDING
        ) AS velocity_30min,
        COUNT(*) OVER (
            PARTITION BY card1 ORDER BY txn_ts
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND INTERVAL '1 second' PRECEDING
        ) AS velocity_1h,
        COUNT(*) OVER (
            PARTITION BY card1 ORDER BY txn_ts
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 second' PRECEDING
        ) AS velocity_24h
    FROM base
),
amount_stats AS (
    SELECT
        transaction_id,
        transaction_amt,
        AVG(transaction_amt) OVER w_past AS card1_hist_avg_amt,
        STDDEV(transaction_amt) OVER w_past AS card1_hist_std_amt,
        COUNT(*) OVER w_past AS card1_hist_txn_count
    FROM base
    WINDOW w_past AS (
        PARTITION BY card1 ORDER BY txn_ts, transaction_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),
recency AS (
    SELECT
        transaction_id,
        txn_ts,
        LAG(txn_ts) OVER (PARTITION BY card1 ORDER BY txn_ts, transaction_id) AS card1_prev_txn_ts
    FROM base
),
device_fraud_rate AS (
    SELECT
        transaction_id,
        device_info,
        AVG(is_fraud::float) OVER w_dev_past AS device_hist_fraud_rate,
        COUNT(*) OVER w_dev_past AS device_hist_txn_count
    FROM base
    WHERE device_info IS NOT NULL
    WINDOW w_dev_past AS (
        PARTITION BY device_info ORDER BY txn_ts, transaction_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),
-- Target encoding: expanding-window fraud rate, same leakage-safe pattern as
-- device_hist_fraud_rate above (only rows strictly before the current one,
-- ordered by txn_ts/transaction_id, ever contribute to a given row's rate).
card1_fraud_rate AS (
    SELECT
        transaction_id,
        AVG(is_fraud::float) OVER w_card1_past AS card1_te_fraud_rate,
        COUNT(*) OVER w_card1_past AS card1_te_txn_count
    FROM base
    WHERE card1 IS NOT NULL
    WINDOW w_card1_past AS (
        PARTITION BY card1 ORDER BY txn_ts, transaction_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),
addr1_fraud_rate AS (
    SELECT
        transaction_id,
        AVG(is_fraud::float) OVER w_addr1_past AS addr1_te_fraud_rate,
        COUNT(*) OVER w_addr1_past AS addr1_te_txn_count
    FROM base
    WHERE addr1 IS NOT NULL
    WINDOW w_addr1_past AS (
        PARTITION BY addr1 ORDER BY txn_ts, transaction_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),
p_emaildomain_fraud_rate AS (
    SELECT
        transaction_id,
        AVG(is_fraud::float) OVER w_pemail_past AS p_emaildomain_te_fraud_rate,
        COUNT(*) OVER w_pemail_past AS p_emaildomain_te_txn_count
    FROM base
    WHERE p_emaildomain IS NOT NULL
    WINDOW w_pemail_past AS (
        PARTITION BY p_emaildomain ORDER BY txn_ts, transaction_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
),
-- Frequency encoding for the two high-cardinality categoricals from the P0
-- cardinality audit (R_emaildomain: 60 distinct, P_emaildomain: 59 distinct
-- -- everything else is <=5). Plain full-table counts, not target-derived,
-- so no leakage-safe windowing is needed here.
freq_encoding AS (
    SELECT
        transaction_id,
        COUNT(*) OVER (PARTITION BY p_emaildomain) AS p_emaildomain_freq,
        COUNT(*) OVER (PARTITION BY "R_emaildomain") AS r_emaildomain_freq
    FROM base
),
v_missingness AS (
    SELECT
        transaction_id,
        CASE WHEN "V1" IS NULL THEN 1 ELSE 0 END AS v_block1_missing,
        CASE WHEN "V12" IS NULL THEN 1 ELSE 0 END AS v_block2_missing,
        CASE WHEN "V35" IS NULL THEN 1 ELSE 0 END AS v_block3_missing,
        CASE WHEN "V53" IS NULL THEN 1 ELSE 0 END AS v_block4_missing,
        CASE WHEN "V95" IS NULL THEN 1 ELSE 0 END AS v_block5_missing,
        CASE WHEN "V138" IS NULL THEN 1 ELSE 0 END AS v_block6_missing,
        CASE WHEN "V167" IS NULL THEN 1 ELSE 0 END AS v_block7_missing,
        CASE WHEN "V279" IS NULL THEN 1 ELSE 0 END AS v_block8_missing,
        CASE WHEN "V322" IS NULL THEN 1 ELSE 0 END AS v_block9_missing
    FROM base
)
SELECT
    b.transaction_id,
    b.is_fraud,
    b.txn_ts,
    b.card1,
    v.velocity_5min,
    v.velocity_30min,
    v.velocity_1h,
    v.velocity_24h,
    a.card1_hist_avg_amt,
    a.card1_hist_std_amt,
    a.card1_hist_txn_count,
    CASE
        WHEN a.card1_hist_std_amt IS NULL OR a.card1_hist_std_amt = 0 THEN NULL
        ELSE (b.transaction_amt - a.card1_hist_avg_amt) / a.card1_hist_std_amt
    END AS amount_zscore_vs_card1_history,
    r.card1_prev_txn_ts,
    EXTRACT(EPOCH FROM (b.txn_ts - r.card1_prev_txn_ts)) AS seconds_since_card1_prev_txn,
    d.device_hist_fraud_rate,
    d.device_hist_txn_count,
    c1.card1_te_fraud_rate,
    c1.card1_te_txn_count,
    a1.addr1_te_fraud_rate,
    a1.addr1_te_txn_count,
    pe.p_emaildomain_te_fraud_rate,
    pe.p_emaildomain_te_txn_count,
    fe.p_emaildomain_freq,
    fe.r_emaildomain_freq,
    vm.v_block1_missing,
    vm.v_block2_missing,
    vm.v_block3_missing,
    vm.v_block4_missing,
    vm.v_block5_missing,
    vm.v_block6_missing,
    vm.v_block7_missing,
    vm.v_block8_missing,
    vm.v_block9_missing,
    b."C1",
    b."C2",
    b."C3",
    b."C4",
    b."C5",
    b."C6",
    b."C7",
    b."C8",
    b."C9",
    b."C10",
    b."C11",
    b."C12",
    b."C13",
    b."C14",
    b."D1",
    b."D2",
    b."D3",
    b."D4",
    b."D5",
    b."D6",
    b."D7",
    b."D8",
    b."D9",
    b."D10",
    b."D11",
    b."D12",
    b."D13",
    b."D14",
    b."D15",
    b."M1",
    b."M2",
    b."M3",
    b."M4",
    b."M5",
    b."M6",
    b."M7",
    b."M8",
    b."M9",
    b."V1",
    b."V2",
    b."V3",
    b."V4",
    b."V5",
    b."V6",
    b."V7",
    b."V8",
    b."V9",
    b."V10",
    b."V11",
    b."V12",
    b."V13",
    b."V14",
    b."V15",
    b."V16",
    b."V17",
    b."V18",
    b."V19",
    b."V20",
    b."V21",
    b."V22",
    b."V23",
    b."V24",
    b."V25",
    b."V26",
    b."V27",
    b."V28",
    b."V29",
    b."V30",
    b."V31",
    b."V32",
    b."V33",
    b."V34",
    b."V35",
    b."V36",
    b."V37",
    b."V38",
    b."V39",
    b."V40",
    b."V41",
    b."V42",
    b."V43",
    b."V44",
    b."V45",
    b."V46",
    b."V47",
    b."V48",
    b."V49",
    b."V50",
    b."V51",
    b."V52",
    b."V53",
    b."V54",
    b."V55",
    b."V56",
    b."V57",
    b."V58",
    b."V59",
    b."V60",
    b."V61",
    b."V62",
    b."V63",
    b."V64",
    b."V65",
    b."V66",
    b."V67",
    b."V68",
    b."V69",
    b."V70",
    b."V71",
    b."V72",
    b."V73",
    b."V74",
    b."V75",
    b."V76",
    b."V77",
    b."V78",
    b."V79",
    b."V80",
    b."V81",
    b."V82",
    b."V83",
    b."V84",
    b."V85",
    b."V86",
    b."V87",
    b."V88",
    b."V89",
    b."V90",
    b."V91",
    b."V92",
    b."V93",
    b."V94",
    b."V95",
    b."V96",
    b."V97",
    b."V98",
    b."V99",
    b."V100",
    b."V101",
    b."V102",
    b."V103",
    b."V104",
    b."V105",
    b."V106",
    b."V107",
    b."V108",
    b."V109",
    b."V110",
    b."V111",
    b."V112",
    b."V113",
    b."V114",
    b."V115",
    b."V116",
    b."V117",
    b."V118",
    b."V119",
    b."V120",
    b."V121",
    b."V122",
    b."V123",
    b."V124",
    b."V125",
    b."V126",
    b."V127",
    b."V128",
    b."V129",
    b."V130",
    b."V131",
    b."V132",
    b."V133",
    b."V134",
    b."V135",
    b."V136",
    b."V137",
    b."V138",
    b."V139",
    b."V140",
    b."V141",
    b."V142",
    b."V143",
    b."V144",
    b."V145",
    b."V146",
    b."V147",
    b."V148",
    b."V149",
    b."V150",
    b."V151",
    b."V152",
    b."V153",
    b."V154",
    b."V155",
    b."V156",
    b."V157",
    b."V158",
    b."V159",
    b."V160",
    b."V161",
    b."V162",
    b."V163",
    b."V164",
    b."V165",
    b."V166",
    b."V167",
    b."V168",
    b."V169",
    b."V170",
    b."V171",
    b."V172",
    b."V173",
    b."V174",
    b."V175",
    b."V176",
    b."V177",
    b."V178",
    b."V179",
    b."V180",
    b."V181",
    b."V182",
    b."V183",
    b."V184",
    b."V185",
    b."V186",
    b."V187",
    b."V188",
    b."V189",
    b."V190",
    b."V191",
    b."V192",
    b."V193",
    b."V194",
    b."V195",
    b."V196",
    b."V197",
    b."V198",
    b."V199",
    b."V200",
    b."V201",
    b."V202",
    b."V203",
    b."V204",
    b."V205",
    b."V206",
    b."V207",
    b."V208",
    b."V209",
    b."V210",
    b."V211",
    b."V212",
    b."V213",
    b."V214",
    b."V215",
    b."V216",
    b."V217",
    b."V218",
    b."V219",
    b."V220",
    b."V221",
    b."V222",
    b."V223",
    b."V224",
    b."V225",
    b."V226",
    b."V227",
    b."V228",
    b."V229",
    b."V230",
    b."V231",
    b."V232",
    b."V233",
    b."V234",
    b."V235",
    b."V236",
    b."V237",
    b."V238",
    b."V239",
    b."V240",
    b."V241",
    b."V242",
    b."V243",
    b."V244",
    b."V245",
    b."V246",
    b."V247",
    b."V248",
    b."V249",
    b."V250",
    b."V251",
    b."V252",
    b."V253",
    b."V254",
    b."V255",
    b."V256",
    b."V257",
    b."V258",
    b."V259",
    b."V260",
    b."V261",
    b."V262",
    b."V263",
    b."V264",
    b."V265",
    b."V266",
    b."V267",
    b."V268",
    b."V269",
    b."V270",
    b."V271",
    b."V272",
    b."V273",
    b."V274",
    b."V275",
    b."V276",
    b."V277",
    b."V278",
    b."V279",
    b."V280",
    b."V281",
    b."V282",
    b."V283",
    b."V284",
    b."V285",
    b."V286",
    b."V287",
    b."V288",
    b."V289",
    b."V290",
    b."V291",
    b."V292",
    b."V293",
    b."V294",
    b."V295",
    b."V296",
    b."V297",
    b."V298",
    b."V299",
    b."V300",
    b."V301",
    b."V302",
    b."V303",
    b."V304",
    b."V305",
    b."V306",
    b."V307",
    b."V308",
    b."V309",
    b."V310",
    b."V311",
    b."V312",
    b."V313",
    b."V314",
    b."V315",
    b."V316",
    b."V317",
    b."V318",
    b."V319",
    b."V320",
    b."V321",
    b."V322",
    b."V323",
    b."V324",
    b."V325",
    b."V326",
    b."V327",
    b."V328",
    b."V329",
    b."V330",
    b."V331",
    b."V332",
    b."V333",
    b."V334",
    b."V335",
    b."V336",
    b."V337",
    b."V338",
    b."V339"
FROM base b
LEFT JOIN velocity v ON b.transaction_id = v.transaction_id
LEFT JOIN amount_stats a ON b.transaction_id = a.transaction_id
LEFT JOIN recency r ON b.transaction_id = r.transaction_id
LEFT JOIN device_fraud_rate d ON b.transaction_id = d.transaction_id
LEFT JOIN card1_fraud_rate c1 ON b.transaction_id = c1.transaction_id
LEFT JOIN addr1_fraud_rate a1 ON b.transaction_id = a1.transaction_id
LEFT JOIN p_emaildomain_fraud_rate pe ON b.transaction_id = pe.transaction_id
LEFT JOIN freq_encoding fe ON b.transaction_id = fe.transaction_id
LEFT JOIN v_missingness vm ON b.transaction_id = vm.transaction_id;

CREATE UNIQUE INDEX idx_features_m1_id ON features_m1 (transaction_id);

SELECT transaction_id, txn_ts, velocity_5min, velocity_24h,
       amount_zscore_vs_card1_history, seconds_since_card1_prev_txn,
       device_hist_fraud_rate, card1_te_fraud_rate, addr1_te_fraud_rate,
       p_emaildomain_te_fraud_rate, p_emaildomain_freq, r_emaildomain_freq,
       v_block1_missing, v_block6_missing
FROM features_m1
ORDER BY txn_ts
LIMIT 20;
