CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    transaction_id TEXT,
    probability_raw DOUBLE PRECISION NOT NULL,
    probability_calibrated DOUBLE PRECISION NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('ALLOW', 'BLOCK')),
    model_version TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at ON prediction_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_transaction_id ON prediction_logs (transaction_id);
