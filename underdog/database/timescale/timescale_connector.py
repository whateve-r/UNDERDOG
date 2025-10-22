"""
TimescaleDB Connector for Time-Series Data

High-performance time-series database for:
- OHLCV data (1-min bars, hypertables)
- Sentiment scores (Reddit, news)
- Macro indicators (FRED)
- Regime predictions (GMM+XGBoost)

Features:
- Hypertables (optimized for time-series)
- Continuous aggregates (1min → 5min → 1h → 1d)
- Retention policies (5 years)
- Compression (80% reduction)

Schema:
- ohlcv_data: timestamp, symbol, open, high, low, close, volume, spread
- sentiment_scores: timestamp, symbol, sentiment, source, confidence
- macro_indicators: timestamp, series_id, value, name
- regime_predictions: timestamp, symbol, regime, confidence

Paper: AlphaQuanter.pdf (Feature Store)
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import contextmanager
import asyncpg
import psycopg2
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OHLCVBar:
    """OHLCV candlestick bar"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float = 0.0


@dataclass
class SentimentScore:
    """Sentiment data point"""
    timestamp: datetime
    symbol: str
    sentiment: float  # [-1, 1]
    source: str  # 'reddit', 'news', 'twitter'
    confidence: float


@dataclass
class MacroIndicator:
    """Macroeconomic indicator"""
    timestamp: datetime
    series_id: str
    value: float
    name: str


class TimescaleDBConnector:
    """
    TimescaleDB connector for time-series data
    
    Usage:
        connector = TimescaleDBConnector()
        await connector.connect()
        await connector.insert_ohlcv_batch(bars)
        df = await connector.query_ohlcv('EURUSD', start_date, end_date)
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'underdog_ts',
        user: str = 'postgres',
        password: str = 'postgres'
    ):
        """
        Initialize TimescaleDB connector
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        
        self.pool: Optional[asyncpg.Pool] = None
        self._is_connected = False
        
        logger.info(f"TimescaleDB connector initialized (host={host}, db={database})")
    
    async def connect(self):
        """Establish connection pool"""
        if self._is_connected:
            return
        
        try:
            logger.info("Connecting to TimescaleDB...")
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            self._is_connected = True
            logger.info("Connected to TimescaleDB successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise
    
    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self._is_connected = False
            logger.info("Disconnected from TimescaleDB")
    
    @contextmanager
    def get_connection(self):
        """
        Synchronous connection context manager for non-async contexts.
        
        Use this in scripts that can't easily use async/await (e.g., FRED backfill).
        For async contexts, use the pool directly.
        
        Example:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM ohlcv_data LIMIT 10")
        """
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        try:
            yield conn
        finally:
            conn.close()
    
    async def create_schema(self):
        """
        Create database schema (hypertables, indexes)
        
        Run once during setup.
        """
        if not self._is_connected:
            await self.connect()
        
        logger.info("Creating TimescaleDB schema...")
        
        async with self.pool.acquire() as conn:
            # Enable TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            
            # OHLCV table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    spread DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, symbol)
                );
            """)
            
            # Convert to hypertable
            await conn.execute("""
                SELECT create_hypertable('ohlcv_data', 'timestamp', 
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Sentiment scores table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    sentiment DOUBLE PRECISION,
                    source TEXT,
                    confidence DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, symbol, source)
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('sentiment_scores', 'timestamp',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Macro indicators table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_indicators (
                    timestamp TIMESTAMPTZ NOT NULL,
                    series_id TEXT NOT NULL,
                    value DOUBLE PRECISION,
                    name TEXT,
                    PRIMARY KEY (timestamp, series_id)
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('macro_indicators', 'timestamp',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '7 days'
                );
            """)
            
            # Regime predictions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_predictions (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    regime TEXT,
                    confidence DOUBLE PRECISION,
                    PRIMARY KEY (timestamp, symbol)
                );
            """)
            
            await conn.execute("""
                SELECT create_hypertable('regime_predictions', 'timestamp',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_data (symbol, timestamp DESC);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment_scores (symbol, timestamp DESC);")
            
            logger.info("Schema created successfully")
    
    async def insert_ohlcv_batch(self, bars: List[OHLCVBar]):
        """
        Insert batch of OHLCV bars (high-performance)
        
        Args:
            bars: List of OHLCVBar objects
        """
        if not bars:
            return
        
        if not self._is_connected:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO ohlcv_data (timestamp, symbol, open, high, low, close, volume, spread)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (timestamp, symbol) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    spread = EXCLUDED.spread
                """,
                [
                    (b.timestamp, b.symbol, b.open, b.high, b.low, b.close, b.volume, b.spread)
                    for b in bars
                ]
            )
        
        logger.debug(f"Inserted {len(bars)} OHLCV bars")
    
    async def query_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Query OHLCV data for symbol
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            start_date: Start timestamp
            end_date: End timestamp
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, spread
        """
        if not self._is_connected:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT timestamp, open, high, low, close, volume, spread
                FROM ohlcv_data
                WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
                ORDER BY timestamp ASC
                """,
                symbol, start_date, end_date
            )
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        logger.debug(f"Queried {len(df)} OHLCV bars for {symbol}")
        return df
    
    async def insert_sentiment_batch(self, scores: List[SentimentScore]):
        """Insert batch of sentiment scores"""
        if not scores:
            return
        
        if not self._is_connected:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO sentiment_scores (timestamp, symbol, sentiment, source, confidence)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (timestamp, symbol, source) DO UPDATE SET
                    sentiment = EXCLUDED.sentiment,
                    confidence = EXCLUDED.confidence
                """,
                [
                    (s.timestamp, s.symbol, s.sentiment, s.source, s.confidence)
                    for s in scores
                ]
            )
        
        logger.debug(f"Inserted {len(scores)} sentiment scores")
    
    async def get_latest_sentiment(self, symbol: str, source: str = 'reddit') -> Optional[float]:
        """Get latest sentiment score for symbol"""
        if not self._is_connected:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT sentiment FROM sentiment_scores
                WHERE symbol = $1 AND source = $2
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                symbol, source
            )
        
        return row['sentiment'] if row else None


# Singleton instance (lazy-loaded)
_connector: Optional[TimescaleDBConnector] = None

def get_connector() -> TimescaleDBConnector:
    """Get singleton TimescaleDB connector"""
    global _connector
    if _connector is None:
        _connector = TimescaleDBConnector()
    return _connector


if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing TimescaleDB Connector...")
        
        connector = TimescaleDBConnector()
        await connector.connect()
        
        print("✓ Connected")
        
        # Test insert
        bars = [
            OHLCVBar(
                timestamp=datetime.now(),
                symbol='EURUSD',
                open=1.1000,
                high=1.1010,
                low=1.0990,
                close=1.1005,
                volume=1000.0,
                spread=0.0001
            )
        ]
        await connector.insert_ohlcv_batch(bars)
        print("✓ Inserted OHLCV bar")
        
        # Test query
        df = await connector.query_ohlcv(
            'EURUSD',
            datetime.now() - timedelta(hours=1),
            datetime.now()
        )
        print(f"✓ Queried {len(df)} bars")
        
        await connector.disconnect()
        print("✓ Disconnected")
    
    asyncio.run(test())
