"""
Database Data Loader - TimescaleDB + Parquet Fallback

This module provides a unified interface for loading OHLCV data from:
1. TimescaleDB (preferred, fast queries)
2. Parquet files (fallback if DB unavailable)
3. CSV files (legacy support)

Usage:
    from underdog.database.db_loader import DBDataLoader
    
    loader = DBDataLoader()
    
    # Load from DB (fastest)
    df = loader.load_ohlcv('EURUSD', '5min', '2024-01-01', '2024-01-31')
    
    # Or let it auto-fallback if DB unavailable
    df = loader.load_data('EURUSD', '5min', start='2024-01-01')
"""
from typing import Optional, Union, Literal
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DBDataLoader:
    """
    Unified data loader with automatic fallback.
    
    Priority:
    1. TimescaleDB (fastest, most recent data)
    2. Parquet files (fast, compressed, stored locally)
    3. CSV files (slow, legacy, stored locally)
    """
    
    def __init__(
        self,
        db_host: str = 'localhost',
        db_port: int = 5432,
        db_name: str = 'underdog',
        db_user: str = 'underdog',
        db_password: str = 'underdog_password',
        parquet_dir: str = 'data/parquet',
        csv_dir: str = 'data/historical'
    ):
        """
        Initialize data loader.
        
        Args:
            db_host: PostgreSQL/TimescaleDB host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password
            parquet_dir: Directory containing Parquet files
            csv_dir: Directory containing CSV files (legacy)
        """
        self.db_params = {
            'host': db_host,
            'port': db_port,
            'database': db_name,
            'user': db_user,
            'password': db_password
        }
        self.parquet_dir = Path(parquet_dir)
        self.csv_dir = Path(csv_dir)
        
        # Check DB availability
        self.db_available = self._check_db_connection()
        
        if self.db_available:
            logger.info("✓ TimescaleDB connection available")
        else:
            logger.warning("⚠️  TimescaleDB unavailable, will use Parquet/CSV fallback")
    
    def _check_db_connection(self) -> bool:
        """
        Check if database is available.
        
        Returns:
            True if DB is reachable, False otherwise
        """
        try:
            import psycopg2
            
            with psycopg2.connect(**self.db_params, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        
        except (ImportError, Exception) as e:
            logger.debug(f"DB connection failed: {e}")
            return False
    
    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        source: Literal['auto', 'db', 'parquet', 'csv'] = 'auto'
    ) -> pd.DataFrame:
        """
        Load OHLCV data with automatic source selection.
        
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., '5min', '1h')
            start: Start date/datetime (optional)
            end: End date/datetime (optional)
            source: Data source ('auto', 'db', 'parquet', 'csv')
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        
        Raises:
            ValueError: If no data found in any source
        """
        # Convert dates if strings
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        
        # Try sources based on priority
        if source == 'auto':
            # Try DB first
            if self.db_available:
                try:
                    df = self._load_from_db(symbol, timeframe, start, end)
                    if not df.empty:
                        logger.info(f"✓ Loaded {len(df):,} rows from TimescaleDB")
                        return df
                except Exception as e:
                    logger.warning(f"DB load failed, trying Parquet: {e}")
            
            # Try Parquet next
            try:
                df = self._load_from_parquet(symbol, timeframe, start, end)
                if not df.empty:
                    logger.info(f"✓ Loaded {len(df):,} rows from Parquet")
                    return df
            except Exception as e:
                logger.warning(f"Parquet load failed, trying CSV: {e}")
            
            # Try CSV last
            try:
                df = self._load_from_csv(symbol, timeframe, start, end)
                if not df.empty:
                    logger.info(f"✓ Loaded {len(df):,} rows from CSV")
                    return df
            except Exception as e:
                logger.error(f"CSV load failed: {e}")
            
            raise ValueError(f"No data found for {symbol} {timeframe} in any source")
        
        elif source == 'db':
            return self._load_from_db(symbol, timeframe, start, end)
        elif source == 'parquet':
            return self._load_from_parquet(symbol, timeframe, start, end)
        elif source == 'csv':
            return self._load_from_csv(symbol, timeframe, start, end)
        else:
            raise ValueError(f"Invalid source: {source}")
    
    def _load_from_db(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load data from TimescaleDB.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            start: Start datetime (optional)
            end: End datetime (optional)
        
        Returns:
            DataFrame with OHLCV data
        """
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        query = """
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv_data
            WHERE symbol = %s
            AND timeframe = %s
        """
        
        params = [symbol, timeframe]
        
        if start:
            query += " AND timestamp >= %s"
            params.append(start)
        
        if end:
            query += " AND timestamp <= %s"
            params.append(end)
        
        query += " ORDER BY timestamp ASC"
        
        with psycopg2.connect(**self.db_params) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    def _load_from_parquet(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load data from Parquet files.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            start: Start datetime (optional)
            end: End datetime (optional)
        
        Returns:
            DataFrame with OHLCV data
        """
        # Find Parquet files matching pattern
        symbol_dir = self.parquet_dir / symbol
        
        if not symbol_dir.exists():
            raise FileNotFoundError(f"No Parquet directory for {symbol}")
        
        # Look for files matching: SYMBOL_YEAR_TIMEFRAME.parquet
        parquet_files = list(symbol_dir.glob(f"{symbol}_*_{timeframe}.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files for {symbol} {timeframe}")
        
        # Load all matching files
        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            dfs.append(df)
        
        # Concatenate
        df = pd.concat(dfs, ignore_index=False)
        df = df.sort_index()
        
        # Filter by date range
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        
        return df
    
    def _load_from_csv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load data from CSV files (legacy).
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            start: Start datetime (optional)
            end: End datetime (optional)
        
        Returns:
            DataFrame with OHLCV data
        """
        # Try common filename patterns
        patterns = [
            f"{symbol}_{timeframe}.csv",
            f"{symbol}_{timeframe.upper()}.csv",
            f"{symbol.lower()}_{timeframe}.csv"
        ]
        
        csv_file = None
        for pattern in patterns:
            path = self.csv_dir / pattern
            if path.exists():
                csv_file = path
                break
        
        if not csv_file:
            raise FileNotFoundError(f"No CSV file found for {symbol} {timeframe}")
        
        # Load CSV
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])
        df = df.set_index('timestamp')
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        
        # Filter by date range
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        
        return df
    
    def get_available_symbols(self, source: str = 'auto') -> list:
        """
        Get list of available symbols.
        
        Args:
            source: Data source to check
        
        Returns:
            List of available currency pairs
        """
        symbols = set()
        
        if source in ['auto', 'db'] and self.db_available:
            try:
                import psycopg2
                
                with psycopg2.connect(**self.db_params) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT DISTINCT symbol FROM ohlcv_data")
                        symbols.update([row[0] for row in cur.fetchall()])
            except Exception as e:
                logger.debug(f"Failed to get symbols from DB: {e}")
        
        if source in ['auto', 'parquet']:
            if self.parquet_dir.exists():
                symbols.update([d.name for d in self.parquet_dir.iterdir() if d.is_dir()])
        
        if source in ['auto', 'csv']:
            if self.csv_dir.exists():
                for file in self.csv_dir.glob("*_*.csv"):
                    symbol = file.stem.split('_')[0].upper()
                    symbols.add(symbol)
        
        return sorted(list(symbols))
    
    def get_available_timeframes(self, symbol: str, source: str = 'auto') -> list:
        """
        Get list of available timeframes for a symbol.
        
        Args:
            symbol: Currency pair
            source: Data source to check
        
        Returns:
            List of available timeframes
        """
        timeframes = set()
        
        if source in ['auto', 'db'] and self.db_available:
            try:
                import psycopg2
                
                with psycopg2.connect(**self.db_params) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT DISTINCT timeframe FROM ohlcv_data WHERE symbol = %s",
                            (symbol,)
                        )
                        timeframes.update([row[0] for row in cur.fetchall()])
            except Exception as e:
                logger.debug(f"Failed to get timeframes from DB: {e}")
        
        if source in ['auto', 'parquet']:
            symbol_dir = self.parquet_dir / symbol
            if symbol_dir.exists():
                for file in symbol_dir.glob(f"{symbol}_*_*.parquet"):
                    # Extract timeframe from filename: SYMBOL_YEAR_TIMEFRAME.parquet
                    parts = file.stem.split('_')
                    if len(parts) >= 3:
                        timeframes.add(parts[-1])
        
        if source in ['auto', 'csv']:
            if self.csv_dir.exists():
                for file in self.csv_dir.glob(f"{symbol}_*.csv"):
                    tf = file.stem.split('_')[1]
                    timeframes.add(tf.lower())
        
        return sorted(list(timeframes))


# Global instance for convenience
_loader = None

def get_loader() -> DBDataLoader:
    """
    Get global data loader instance (singleton).
    
    Returns:
        DBDataLoader instance
    """
    global _loader
    if _loader is None:
        _loader = DBDataLoader()
    return _loader
