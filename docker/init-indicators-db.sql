-- =============================================================================
-- TimescaleDB Schema for OHLCV Data with Technical Indicators
-- Creates separate hypertables for each timeframe with 36+ columns
-- =============================================================================

-- Create TimescaleDB extension (if not exists)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- OHLCV + Indicators Tables (One per Timeframe)
-- =============================================================================

-- 1-minute timeframe
CREATE TABLE IF NOT EXISTS ohlcv_1m (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    
    -- OHLCV Base Data
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    
    -- Moving Averages
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    ema_12 DOUBLE PRECISION,
    ema_26 DOUBLE PRECISION,
    ema_50 DOUBLE PRECISION,
    
    -- Momentum Indicators
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,
    
    -- Volatility Indicators
    atr_14 DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_middle DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    
    -- Trend Indicators
    supertrend DOUBLE PRECISION,
    supertrend_direction INTEGER,
    parabolic_sar DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    
    -- Volume Indicators
    obv BIGINT,
    vwap DOUBLE PRECISION,
    
    -- Keltner Channels
    keltner_upper DOUBLE PRECISION,
    keltner_middle DOUBLE PRECISION,
    keltner_lower DOUBLE PRECISION,
    
    -- Williams %R
    williams_r DOUBLE PRECISION,
    
    PRIMARY KEY (timestamp, symbol)
);

-- 5-minute timeframe
CREATE TABLE IF NOT EXISTS ohlcv_5m (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    ema_12 DOUBLE PRECISION,
    ema_26 DOUBLE PRECISION,
    ema_50 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_middle DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    supertrend DOUBLE PRECISION,
    supertrend_direction INTEGER,
    parabolic_sar DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    obv BIGINT,
    vwap DOUBLE PRECISION,
    keltner_upper DOUBLE PRECISION,
    keltner_middle DOUBLE PRECISION,
    keltner_lower DOUBLE PRECISION,
    williams_r DOUBLE PRECISION,
    PRIMARY KEY (timestamp, symbol)
);

-- 15-minute timeframe
CREATE TABLE IF NOT EXISTS ohlcv_15m (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    ema_12 DOUBLE PRECISION,
    ema_26 DOUBLE PRECISION,
    ema_50 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_middle DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    supertrend DOUBLE PRECISION,
    supertrend_direction INTEGER,
    parabolic_sar DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    obv BIGINT,
    vwap DOUBLE PRECISION,
    keltner_upper DOUBLE PRECISION,
    keltner_middle DOUBLE PRECISION,
    keltner_lower DOUBLE PRECISION,
    williams_r DOUBLE PRECISION,
    PRIMARY KEY (timestamp, symbol)
);

-- 30-minute timeframe
CREATE TABLE IF NOT EXISTS ohlcv_30m (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    ema_12 DOUBLE PRECISION,
    ema_26 DOUBLE PRECISION,
    ema_50 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_middle DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    supertrend DOUBLE PRECISION,
    supertrend_direction INTEGER,
    parabolic_sar DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    obv BIGINT,
    vwap DOUBLE PRECISION,
    keltner_upper DOUBLE PRECISION,
    keltner_middle DOUBLE PRECISION,
    keltner_lower DOUBLE PRECISION,
    williams_r DOUBLE PRECISION,
    PRIMARY KEY (timestamp, symbol)
);

-- 1-hour timeframe
CREATE TABLE IF NOT EXISTS ohlcv_1H (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    ema_12 DOUBLE PRECISION,
    ema_26 DOUBLE PRECISION,
    ema_50 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_middle DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    supertrend DOUBLE PRECISION,
    supertrend_direction INTEGER,
    parabolic_sar DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    obv BIGINT,
    vwap DOUBLE PRECISION,
    keltner_upper DOUBLE PRECISION,
    keltner_middle DOUBLE PRECISION,
    keltner_lower DOUBLE PRECISION,
    williams_r DOUBLE PRECISION,
    PRIMARY KEY (timestamp, symbol)
);

-- 4-hour timeframe
CREATE TABLE IF NOT EXISTS ohlcv_4H (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    ema_12 DOUBLE PRECISION,
    ema_26 DOUBLE PRECISION,
    ema_50 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_middle DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    supertrend DOUBLE PRECISION,
    supertrend_direction INTEGER,
    parabolic_sar DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    obv BIGINT,
    vwap DOUBLE PRECISION,
    keltner_upper DOUBLE PRECISION,
    keltner_middle DOUBLE PRECISION,
    keltner_lower DOUBLE PRECISION,
    williams_r DOUBLE PRECISION,
    PRIMARY KEY (timestamp, symbol)
);

-- 1-day timeframe
CREATE TABLE IF NOT EXISTS ohlcv_1D (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    ema_12 DOUBLE PRECISION,
    ema_26 DOUBLE PRECISION,
    ema_50 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_histogram DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_middle DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    supertrend DOUBLE PRECISION,
    supertrend_direction INTEGER,
    parabolic_sar DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    obv BIGINT,
    vwap DOUBLE PRECISION,
    keltner_upper DOUBLE PRECISION,
    keltner_middle DOUBLE PRECISION,
    keltner_lower DOUBLE PRECISION,
    williams_r DOUBLE PRECISION,
    PRIMARY KEY (timestamp, symbol)
);

-- =============================================================================
-- Convert to Hypertables (Partitioned by Timestamp)
-- =============================================================================

SELECT create_hypertable('ohlcv_1m', 'timestamp', 
    chunk_time_interval => INTERVAL '1 week', 
    if_not_exists => TRUE
);

SELECT create_hypertable('ohlcv_5m', 'timestamp', 
    chunk_time_interval => INTERVAL '1 week', 
    if_not_exists => TRUE
);

SELECT create_hypertable('ohlcv_15m', 'timestamp', 
    chunk_time_interval => INTERVAL '1 week', 
    if_not_exists => TRUE
);

SELECT create_hypertable('ohlcv_30m', 'timestamp', 
    chunk_time_interval => INTERVAL '1 week', 
    if_not_exists => TRUE
);

SELECT create_hypertable('ohlcv_1H', 'timestamp', 
    chunk_time_interval => INTERVAL '1 month', 
    if_not_exists => TRUE
);

SELECT create_hypertable('ohlcv_4H', 'timestamp', 
    chunk_time_interval => INTERVAL '1 month', 
    if_not_exists => TRUE
);

SELECT create_hypertable('ohlcv_1D', 'timestamp', 
    chunk_time_interval => INTERVAL '6 months', 
    if_not_exists => TRUE
);

-- =============================================================================
-- Create Indexes for Fast Queries
-- =============================================================================

-- Indexes for 1-minute
CREATE INDEX IF NOT EXISTS ohlcv_1m_symbol_time_idx ON ohlcv_1m (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS ohlcv_1m_symbol_idx ON ohlcv_1m (symbol);

-- Indexes for 5-minute
CREATE INDEX IF NOT EXISTS ohlcv_5m_symbol_time_idx ON ohlcv_5m (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS ohlcv_5m_symbol_idx ON ohlcv_5m (symbol);

-- Indexes for 15-minute
CREATE INDEX IF NOT EXISTS ohlcv_15m_symbol_time_idx ON ohlcv_15m (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS ohlcv_15m_symbol_idx ON ohlcv_15m (symbol);

-- Indexes for 30-minute
CREATE INDEX IF NOT EXISTS ohlcv_30m_symbol_time_idx ON ohlcv_30m (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS ohlcv_30m_symbol_idx ON ohlcv_30m (symbol);

-- Indexes for 1-hour
CREATE INDEX IF NOT EXISTS ohlcv_1H_symbol_time_idx ON ohlcv_1H (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS ohlcv_1H_symbol_idx ON ohlcv_1H (symbol);

-- Indexes for 4-hour
CREATE INDEX IF NOT EXISTS ohlcv_4H_symbol_time_idx ON ohlcv_4H (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS ohlcv_4H_symbol_idx ON ohlcv_4H (symbol);

-- Indexes for 1-day
CREATE INDEX IF NOT EXISTS ohlcv_1D_symbol_time_idx ON ohlcv_1D (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS ohlcv_1D_symbol_idx ON ohlcv_1D (symbol);

-- =============================================================================
-- Compression Policies (Reduce Storage for Old Data)
-- =============================================================================

-- Compress data older than 30 days (1m, 5m, 15m, 30m)
SELECT add_compression_policy('ohlcv_1m', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('ohlcv_5m', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('ohlcv_15m', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('ohlcv_30m', INTERVAL '30 days', if_not_exists => TRUE);

-- Compress data older than 90 days (1H, 4H)
SELECT add_compression_policy('ohlcv_1H', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_compression_policy('ohlcv_4H', INTERVAL '90 days', if_not_exists => TRUE);

-- Compress data older than 180 days (1D)
SELECT add_compression_policy('ohlcv_1D', INTERVAL '180 days', if_not_exists => TRUE);

-- =============================================================================
-- Retention Policies (Delete Very Old Data - Optional)
-- =============================================================================

-- Delete 1-minute data older than 6 months
SELECT add_retention_policy('ohlcv_1m', INTERVAL '6 months', if_not_exists => TRUE);

-- Delete 5-minute data older than 1 year
SELECT add_retention_policy('ohlcv_5m', INTERVAL '1 year', if_not_exists => TRUE);

-- Keep higher timeframes for 2+ years
SELECT add_retention_policy('ohlcv_15m', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('ohlcv_30m', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('ohlcv_1H', INTERVAL '3 years', if_not_exists => TRUE);
SELECT add_retention_policy('ohlcv_4H', INTERVAL '5 years', if_not_exists => TRUE);
-- No retention for 1D (keep forever)

-- =============================================================================
-- Grants (Security)
-- =============================================================================

-- Grant permissions to underdog user
GRANT ALL PRIVILEGES ON ohlcv_1m TO underdog;
GRANT ALL PRIVILEGES ON ohlcv_5m TO underdog;
GRANT ALL PRIVILEGES ON ohlcv_15m TO underdog;
GRANT ALL PRIVILEGES ON ohlcv_30m TO underdog;
GRANT ALL PRIVILEGES ON ohlcv_1H TO underdog;
GRANT ALL PRIVILEGES ON ohlcv_4H TO underdog;
GRANT ALL PRIVILEGES ON ohlcv_1D TO underdog;

-- =============================================================================
-- Summary
-- =============================================================================

DO $$
DECLARE
    table_record RECORD;
BEGIN
    RAISE NOTICE '=============================================================================';
    RAISE NOTICE 'OHLCV + Indicators Schema Initialization Complete!';
    RAISE NOTICE '=============================================================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables created:';
    
    FOR table_record IN 
        SELECT hypertable_name, num_chunks, 
               pg_size_pretty(total_bytes::bigint) AS total_size
        FROM timescaledb_information.hypertables
        WHERE hypertable_schema = 'public' 
          AND hypertable_name LIKE 'ohlcv_%'
        ORDER BY hypertable_name
    LOOP
        RAISE NOTICE '  - % (chunks: %, size: %)', 
            table_record.hypertable_name, 
            table_record.num_chunks,
            table_record.total_size;
    END LOOP;
    
    RAISE NOTICE '';
    RAISE NOTICE 'Compression policies: 30-180 days depending on timeframe';
    RAISE NOTICE 'Retention policies: 6 months - unlimited';
    RAISE NOTICE 'Indexes: (symbol, timestamp DESC) + (symbol)';
    RAISE NOTICE '';
    RAISE NOTICE 'Ready to insert data from Parquet files!';
    RAISE NOTICE '=============================================================================';
END
$$;
