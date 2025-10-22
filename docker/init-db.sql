-- =============================================================================
-- TimescaleDB Initialization Script for UNDERDOG Trading System
-- Creates tables for OHLCV data, trades, positions, and metrics
-- =============================================================================

-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- OHLCV (Candlestick) Data Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,  -- e.g., '1H', '4H', '1D'
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    spread DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert to hypertable (partitioned by time)
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS ohlcv_symbol_time_idx ON ohlcv (symbol, time DESC);
CREATE INDEX IF NOT EXISTS ohlcv_timeframe_time_idx ON ohlcv (timeframe, time DESC);

-- =============================================================================
-- Sentiment Scores Table (LLM Sentiment Analysis)
-- =============================================================================
CREATE TABLE IF NOT EXISTS sentiment_scores (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    sentiment DOUBLE PRECISION NOT NULL,  -- [-1, 1] range
    source VARCHAR(50) NOT NULL,  -- 'reddit', 'news', 'twitter'
    confidence DOUBLE PRECISION,
    text_sample TEXT,  -- First 200 chars of analyzed text
    metadata JSONB,
    PRIMARY KEY (time, symbol, source)
);

-- Convert to hypertable
SELECT create_hypertable('sentiment_scores', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS sentiment_symbol_time_idx ON sentiment_scores (symbol, time DESC);
CREATE INDEX IF NOT EXISTS sentiment_source_idx ON sentiment_scores (source);

-- =============================================================================
-- Macro Indicators Table (FRED, Alpha Vantage)
-- =============================================================================
CREATE TABLE IF NOT EXISTS macro_indicators (
    time TIMESTAMPTZ NOT NULL,
    series_id VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    name VARCHAR(200),
    units VARCHAR(50),
    frequency VARCHAR(20),  -- 'daily', 'monthly', etc.
    metadata JSONB,
    PRIMARY KEY (time, series_id)
);

-- Convert to hypertable
SELECT create_hypertable('macro_indicators', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS macro_series_time_idx ON macro_indicators (series_id, time DESC);

-- =============================================================================
-- Regime Predictions Table (GMM + XGBoost Classifier)
-- =============================================================================
CREATE TABLE IF NOT EXISTS regime_predictions (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    regime VARCHAR(20) NOT NULL,  -- 'trend', 'range', 'transition'
    confidence DOUBLE PRECISION,
    features JSONB,  -- ATR, ADX, BB width, RSI, volume
    model_version VARCHAR(20),
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('regime_predictions', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS regime_symbol_time_idx ON regime_predictions (symbol, time DESC);
CREATE INDEX IF NOT EXISTS regime_type_idx ON regime_predictions (regime);

-- =============================================================================
-- Trades Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS trades (
    trade_id VARCHAR(50) PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'long' or 'short'
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    size DOUBLE PRECISION NOT NULL,
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION,
    commission DOUBLE PRECISION,
    strategy VARCHAR(50),
    confidence DOUBLE PRECISION,
    regime VARCHAR(20),
    result VARCHAR(10),  -- 'win' or 'loss'
    duration_seconds INTEGER,
    exit_reason VARCHAR(50),
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS trades_symbol_idx ON trades (symbol);
CREATE INDEX IF NOT EXISTS trades_strategy_idx ON trades (strategy);
CREATE INDEX IF NOT EXISTS trades_result_idx ON trades (result);
CREATE INDEX IF NOT EXISTS trades_time_idx ON trades (time DESC);

-- =============================================================================
-- Positions Table (Current Open Positions)
-- =============================================================================
CREATE TABLE IF NOT EXISTS positions (
    position_id VARCHAR(50) PRIMARY KEY,
    opened_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    current_price DOUBLE PRECISION,
    size DOUBLE PRECISION NOT NULL,
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    strategy VARCHAR(50),
    confidence DOUBLE PRECISION,
    regime VARCHAR(20),
    metadata JSONB
);

-- Create indexes
CREATE INDEX IF NOT EXISTS positions_symbol_idx ON positions (symbol);
CREATE INDEX IF NOT EXISTS positions_strategy_idx ON positions (strategy);

-- =============================================================================
-- Metrics Table (System Performance Metrics)
-- =============================================================================
CREATE TABLE IF NOT EXISTS metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    labels JSONB,
    PRIMARY KEY (time, metric_name)
);

-- Convert to hypertable
SELECT create_hypertable('metrics', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS metrics_name_time_idx ON metrics (metric_name, time DESC);
CREATE INDEX IF NOT EXISTS metrics_labels_idx ON metrics USING GIN (labels);

-- =============================================================================
-- Account Snapshots Table (Periodic Capital/Equity Snapshots)
-- =============================================================================
CREATE TABLE IF NOT EXISTS account_snapshots (
    time TIMESTAMPTZ NOT NULL PRIMARY KEY,
    balance DOUBLE PRECISION NOT NULL,
    equity DOUBLE PRECISION NOT NULL,
    margin DOUBLE PRECISION,
    free_margin DOUBLE PRECISION,
    margin_level DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    drawdown_pct DOUBLE PRECISION,
    total_exposure DOUBLE PRECISION,
    open_positions INTEGER,
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('account_snapshots', 'time', if_not_exists => TRUE);

-- =============================================================================
-- Continuous Aggregates (Materialized Views for Performance)
-- =============================================================================

-- Daily trading summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_trading_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    symbol,
    COUNT(*) AS total_trades,
    COUNT(*) FILTER (WHERE result = 'win') AS winning_trades,
    COUNT(*) FILTER (WHERE result = 'loss') AS losing_trades,
    SUM(realized_pnl) AS total_pnl,
    AVG(realized_pnl) AS avg_pnl,
    MIN(realized_pnl) AS min_pnl,
    MAX(realized_pnl) AS max_pnl,
    AVG(confidence) AS avg_confidence
FROM trades
GROUP BY day, symbol;

-- Refresh policy (update every hour)
SELECT add_continuous_aggregate_policy('daily_trading_summary',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Hourly sentiment summary (rolling 24h avg)
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_sentiment_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    symbol,
    source,
    AVG(sentiment) AS avg_sentiment,
    STDDEV(sentiment) AS sentiment_volatility,
    COUNT(*) AS sample_count
FROM sentiment_scores
GROUP BY hour, symbol, source;

-- Refresh policy (update every 15 minutes)
SELECT add_continuous_aggregate_policy('hourly_sentiment_summary',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE
);

-- =============================================================================
-- Compression Policy (Reduce Storage for Old Data)
-- =============================================================================

-- Compress OHLCV data older than 30 days
SELECT add_compression_policy('ohlcv', INTERVAL '30 days', if_not_exists => TRUE);

-- Compress trades older than 90 days
SELECT add_compression_policy('trades', INTERVAL '90 days', if_not_exists => TRUE);

-- Compress metrics older than 30 days
SELECT add_compression_policy('metrics', INTERVAL '30 days', if_not_exists => TRUE);

-- Compress sentiment scores older than 60 days
SELECT add_compression_policy('sentiment_scores', INTERVAL '60 days', if_not_exists => TRUE);

-- Compress macro indicators older than 180 days (6 months)
SELECT add_compression_policy('macro_indicators', INTERVAL '180 days', if_not_exists => TRUE);

-- Compress regime predictions older than 90 days
SELECT add_compression_policy('regime_predictions', INTERVAL '90 days', if_not_exists => TRUE);

-- =============================================================================
-- Retention Policy (Delete Very Old Data)
-- =============================================================================

-- Delete OHLCV data older than 2 years
SELECT add_retention_policy('ohlcv', INTERVAL '2 years', if_not_exists => TRUE);

-- Delete trades older than 3 years
SELECT add_retention_policy('trades', INTERVAL '3 years', if_not_exists => TRUE);

-- Delete metrics older than 1 year
SELECT add_retention_policy('metrics', INTERVAL '1 year', if_not_exists => TRUE);

-- Delete sentiment scores older than 1 year
SELECT add_retention_policy('sentiment_scores', INTERVAL '1 year', if_not_exists => TRUE);

-- Delete macro indicators older than 5 years
SELECT add_retention_policy('macro_indicators', INTERVAL '5 years', if_not_exists => TRUE);

-- Delete regime predictions older than 2 years
SELECT add_retention_policy('regime_predictions', INTERVAL '2 years', if_not_exists => TRUE);

-- =============================================================================
-- DRL Agent Decisions Log (rl_action_log) - CRÍTICA para MLOps
-- =============================================================================
CREATE TABLE IF NOT EXISTS rl_action_log (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    state_features JSONB NOT NULL,  -- Vector de estado completo (14 dimensiones)
    action_raw DOUBLE PRECISION NOT NULL,  -- Acción pedida por TD3 Agent
    action_final DOUBLE PRECISION NOT NULL,  -- Acción después de Safety Shield
    reward DOUBLE PRECISION,  -- Recompensa inmediata
    is_safe_projected BOOLEAN NOT NULL,  -- Si action_raw != action_final
    agent_version VARCHAR(20),
    metadata JSONB,
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('rl_action_log', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS rl_action_symbol_time_idx ON rl_action_log (symbol, time DESC);
CREATE INDEX IF NOT EXISTS rl_action_safe_idx ON rl_action_log (is_safe_projected);

-- =============================================================================
-- Risk/Compliance Audit Log (risk_compliance_log) - CRÍTICA para PropFirm
-- =============================================================================
CREATE TABLE IF NOT EXISTS risk_compliance_log (
    time TIMESTAMPTZ NOT NULL,
    rule_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- 'INTERVENTION', 'VIOLATION', 'LOCKOUT'
    symbol VARCHAR(20),
    action_original DOUBLE PRECISION,  -- Lote original solicitado
    action_modified DOUBLE PRECISION,  -- Lote final permitido (0.0 si bloqueado)
    reason_context JSONB NOT NULL,  -- {drawdown_pct, margin_free, total_exposure}
    severity VARCHAR(20),  -- 'INFO', 'WARNING', 'CRITICAL'
    PRIMARY KEY (time, rule_id)
);

-- Convert to hypertable
SELECT create_hypertable('risk_compliance_log', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS risk_compliance_event_idx ON risk_compliance_log (event_type);
CREATE INDEX IF NOT EXISTS risk_compliance_severity_idx ON risk_compliance_log (severity);
CREATE INDEX IF NOT EXISTS risk_compliance_time_idx ON risk_compliance_log (time DESC);

-- =============================================================================
-- Feature Store Metadata (feature_store_metadata) - MLOps Feature Drift
-- =============================================================================
CREATE TABLE IF NOT EXISTS feature_store_metadata (
    feature_id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10),
    source_table VARCHAR(50) NOT NULL,  -- 'ohlcv', 'sentiment_scores', 'macro_indicators'
    calculation_params JSONB,  -- e.g., {'period': 14, 'type': 'RSI'}
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    description TEXT
);

-- Create indexes
CREATE INDEX IF NOT EXISTS feature_store_symbol_idx ON feature_store_metadata (symbol);
CREATE INDEX IF NOT EXISTS feature_store_active_idx ON feature_store_metadata (is_active);

-- =============================================================================
-- Backtest & Deployment History (strategy_run_history) - A/B Testing
-- =============================================================================
CREATE TABLE IF NOT EXISTS strategy_run_history (
    run_id VARCHAR(50) PRIMARY KEY,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    agent_version VARCHAR(20) NOT NULL,
    config_hash VARCHAR(64) NOT NULL,  -- SHA256 hash del config para reproducibilidad
    final_metrics JSONB,  -- {sharpe, max_dd, total_pnl, win_rate, profit_factor}
    status VARCHAR(20) NOT NULL,  -- 'COMPLETED', 'RUNNING', 'ERROR'
    environment VARCHAR(20),  -- 'BACKTEST', 'DEMO', 'LIVE'
    error_message TEXT,
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('strategy_run_history', 'start_time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS strategy_run_agent_idx ON strategy_run_history (agent_version);
CREATE INDEX IF NOT EXISTS strategy_run_status_idx ON strategy_run_history (status);
CREATE INDEX IF NOT EXISTS strategy_run_env_idx ON strategy_run_history (environment);

-- =============================================================================
-- Data and Service Health Log (service_health_log) - Monitoreo
-- =============================================================================
CREATE TABLE IF NOT EXISTS service_health_log (
    time TIMESTAMPTZ NOT NULL,
    service_name VARCHAR(50) NOT NULL,
    status_code INTEGER NOT NULL,  -- 200 (OK), 4xx (Warning), 5xx (Error)
    latency_ms INTEGER,  -- Latencia en milisegundos
    error_message TEXT,
    metadata JSONB,
    PRIMARY KEY (time, service_name)
);

-- Convert to hypertable
SELECT create_hypertable('service_health_log', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS service_health_name_time_idx ON service_health_log (service_name, time DESC);
CREATE INDEX IF NOT EXISTS service_health_status_idx ON service_health_log (status_code);

-- =============================================================================
-- Compression Policies for New Tables
-- =============================================================================
SELECT add_compression_policy('rl_action_log', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_compression_policy('risk_compliance_log', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_compression_policy('strategy_run_history', INTERVAL '180 days', if_not_exists => TRUE);
SELECT add_compression_policy('service_health_log', INTERVAL '30 days', if_not_exists => TRUE);

-- =============================================================================
-- Retention Policies for New Tables
-- =============================================================================
SELECT add_retention_policy('rl_action_log', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('risk_compliance_log', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('service_health_log', INTERVAL '1 year', if_not_exists => TRUE);

-- =============================================================================
-- Utility Functions
-- =============================================================================

-- Function to calculate drawdown
CREATE OR REPLACE FUNCTION calculate_drawdown(
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ
)
RETURNS TABLE(
    time TIMESTAMPTZ,
    drawdown_pct DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    WITH equity_series AS (
        SELECT
            time,
            equity,
            MAX(equity) OVER (ORDER BY time) AS peak_equity
        FROM account_snapshots
        WHERE time BETWEEN start_time AND end_time
    )
    SELECT
        equity_series.time,
        CASE
            WHEN peak_equity > 0 THEN ((peak_equity - equity) / peak_equity) * 100
            ELSE 0
        END AS drawdown_pct
    FROM equity_series
    ORDER BY time;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Grants (Security)
-- =============================================================================

-- Grant permissions to underdog user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO underdog;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO underdog;

-- =============================================================================
-- Initial Data Check
-- =============================================================================

-- Show table info
SELECT
    hypertable_name,
    num_chunks,
    table_bytes / 1024 / 1024 AS table_mb,
    index_bytes / 1024 / 1024 AS index_mb,
    total_bytes / 1024 / 1024 AS total_mb
FROM timescaledb_information.hypertables
WHERE hypertable_schema = 'public';

-- Success message
DO $$
BEGIN
    RAISE NOTICE '=============================================================================';
    RAISE NOTICE 'TimescaleDB initialization complete!';
    RAISE NOTICE '=============================================================================';
    RAISE NOTICE 'Core tables: ohlcv, trades, positions, metrics, account_snapshots';
    RAISE NOTICE 'Alternative data: sentiment_scores, macro_indicators, regime_predictions';
    RAISE NOTICE 'DRL/MLOps: rl_action_log, feature_store_metadata, strategy_run_history';
    RAISE NOTICE 'Risk/Compliance: risk_compliance_log';
    RAISE NOTICE 'Monitoring: service_health_log';
    RAISE NOTICE 'Continuous aggregates: daily_trading_summary, hourly_sentiment_summary';
    RAISE NOTICE 'Compression and retention policies: ACTIVE';
    RAISE NOTICE '=============================================================================';
END
$$;
