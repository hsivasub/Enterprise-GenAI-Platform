-- ════════════════════════════════════════════════════
-- Enterprise GenAI Platform — PostgreSQL Schema Init
-- Runs automatically when the postgres container starts
-- ════════════════════════════════════════════════════

-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ──────────────────────────────────────────────────────
-- Audit log for all LLM interactions
-- ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS llm_audit_log (
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    request_id      VARCHAR(64) NOT NULL,
    session_id      VARCHAR(64),
    user_id         VARCHAR(64),
    question        TEXT        NOT NULL,
    answer          TEXT,
    model           VARCHAR(64),
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        NUMERIC(10, 6),
    latency_ms      NUMERIC(10, 2),
    tool_calls      JSONB,
    hallucination_risk VARCHAR(10),
    cached          BOOLEAN     DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_audit_session ON llm_audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_llm_audit_created ON llm_audit_log(created_at DESC);

-- ──────────────────────────────────────────────────────
-- Document registry (tracks ingested documents)
-- ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    doc_id          VARCHAR(64) UNIQUE NOT NULL,
    filename        VARCHAR(512),
    doc_type        VARCHAR(16),
    file_size_bytes BIGINT,
    content_hash    VARCHAR(64),
    chunks_created  INTEGER,
    strategy        VARCHAR(64),
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ──────────────────────────────────────────────────────
-- Financial metrics (sample data)
-- ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS financial_metrics (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(16)     NOT NULL,
    period      VARCHAR(16),
    revenue     NUMERIC(18, 2),
    net_income  NUMERIC(18, 2),
    eps         NUMERIC(10, 4),
    date        DATE
);

CREATE TABLE IF NOT EXISTS stock_prices (
    id      SERIAL PRIMARY KEY,
    ticker  VARCHAR(16)     NOT NULL,
    date    DATE            NOT NULL,
    open    NUMERIC(10, 4),
    high    NUMERIC(10, 4),
    low     NUMERIC(10, 4),
    close   NUMERIC(10, 4),
    volume  BIGINT,
    UNIQUE (ticker, date)
);

CREATE TABLE IF NOT EXISTS company_info (
    ticker      VARCHAR(16) PRIMARY KEY,
    name        VARCHAR(256),
    sector      VARCHAR(64),
    market_cap  NUMERIC(18, 2),
    exchange    VARCHAR(16)
);

-- ──────────────────────────────────────────────────────
-- Sample financial data
-- ──────────────────────────────────────────────────────
INSERT INTO company_info VALUES
    ('AAPL', 'Apple Inc.', 'Technology', 3100000000000.00, 'NASDAQ'),
    ('MSFT', 'Microsoft Corp.', 'Technology', 3050000000000.00, 'NASDAQ'),
    ('GOOGL', 'Alphabet Inc.', 'Technology', 1900000000000.00, 'NASDAQ'),
    ('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 1800000000000.00, 'NASDAQ')
ON CONFLICT DO NOTHING;

INSERT INTO financial_metrics (ticker, period, revenue, net_income, eps, date) VALUES
    ('AAPL', 'Q3-2024', 85777000000.00, 21448000000.00, 1.40, '2024-09-30'),
    ('AAPL', 'Q2-2024', 90753000000.00, 23636000000.00, 1.53, '2024-06-30'),
    ('MSFT', 'Q3-2024', 61858000000.00, 21939000000.00, 2.94, '2024-09-30'),
    ('GOOGL', 'Q3-2024', 88268000000.00, 19689000000.00, 1.55, '2024-09-30')
ON CONFLICT DO NOTHING;
