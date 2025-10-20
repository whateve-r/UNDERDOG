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

-- =============================================================================
-- Compression Policy (Reduce Storage for Old Data)
-- =============================================================================

-- Compress OHLCV data older than 30 days
SELECT add_compression_policy('ohlcv', INTERVAL '30 days', if_not_exists => TRUE);

-- Compress trades older than 90 days
SELECT add_compression_policy('trades', INTERVAL '90 days', if_not_exists => TRUE);

-- Compress metrics older than 30 days
SELECT add_compression_policy('metrics', INTERVAL '30 days', if_not_exists => TRUE);

-- =============================================================================
-- Retention Policy (Delete Very Old Data)
-- =============================================================================

-- Delete OHLCV data older than 2 years
SELECT add_retention_policy('ohlcv', INTERVAL '2 years', if_not_exists => TRUE);

-- Delete trades older than 3 years
SELECT add_retention_policy('trades', INTERVAL '3 years', if_not_exists => TRUE);

-- Delete metrics older than 1 year
SELECT add_retention_policy('metrics', INTERVAL '1 year', if_not_exists => TRUE);

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
    RAISE NOTICE 'TimescaleDB initialization complete!';
    RAISE NOTICE 'Tables created: ohlcv, trades, positions, metrics, account_snapshots';
    RAISE NOTICE 'Compression and retention policies configured';
END
$$;
