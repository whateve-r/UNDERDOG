"""
Insert Parquet data with indicators into TimescaleDB

This script:
1. Reads all Parquet files from data/parquet/
2. Creates separate tables per timeframe (ohlcv_1min, ohlcv_5min, etc.)
3. Each table includes OHLCV + all technical indicators
4. Creates hypertables for time-series optimization
5. Creates indexes for fast symbol/timeframe queries

Usage:
    # Insert all data
    poetry run python scripts/insert_indicators_to_db.py --all
    
    # Insert specific symbol
    poetry run python scripts/insert_indicators_to_db.py --symbols EURUSD
    
    # Insert specific timeframe
    poetry run python scripts/insert_indicators_to_db.py --timeframes 5min 15min
"""
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import argparse
import logging
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# Add underdog to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndicatorDatabaseInserter:
    """
    Insert Parquet files with indicators into TimescaleDB.
    """
    
    TIMEFRAMES = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']
    
    def __init__(
        self,
        parquet_dir: str = 'data/parquet',
        db_host: str = 'localhost',
        db_port: int = 5432,
        db_name: str = 'underdog',
        db_user: str = 'underdog',
        db_password: str = 'underdog_password'
    ):
        """
        Initialize inserter.
        
        Args:
            parquet_dir: Directory containing Parquet files
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password
        """
        self.parquet_dir = Path(parquet_dir)
        self.conn_params = {
            'host': db_host,
            'port': db_port,
            'database': db_name,
            'user': db_user,
            'password': db_password
        }
        
        # Statistics
        self.stats = {
            'tables_created': 0,
            'total_rows_inserted': 0,
            'rows_per_timeframe': {},
            'errors': []
        }
        
        logger.info("=" * 80)
        logger.info("INDICATOR DATABASE INSERTION")
        logger.info("=" * 80)
        logger.info(f"Parquet directory: {parquet_dir}")
        logger.info(f"Database: {db_host}:{db_port}/{db_name}")
        logger.info("=" * 80)
    
    def insert_all(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None
    ) -> dict:
        """
        Insert all Parquet data into database.
        
        Args:
            symbols: Specific symbols to insert (None = all)
            timeframes: Specific timeframes to insert (None = all)
        
        Returns:
            Statistics dictionary
        """
        # Get all symbols from parquet directory
        if symbols is None:
            symbols = [d.name for d in self.parquet_dir.iterdir() if d.is_dir()]
        
        if timeframes is None:
            timeframes = self.TIMEFRAMES
        
        logger.info(f"\nSymbols to insert: {', '.join(symbols)}")
        logger.info(f"Timeframes to insert: {', '.join(timeframes)}\n")
        
        # Connect to database
        with psycopg2.connect(**self.conn_params) as conn:
            # Create tables for each timeframe
            for tf in timeframes:
                self._create_table(conn, tf)
            
            # Insert data
            for symbol in symbols:
                for tf in timeframes:
                    self._insert_symbol_timeframe(conn, symbol, tf)
        
        logger.info("\n" + "=" * 80)
        logger.info("INSERTION COMPLETE")
        logger.info("=" * 80)
        self._print_statistics()
        
        return self.stats
    
    def _create_table(self, conn, timeframe: str) -> None:
        """
        Create table for a timeframe with all indicator columns.
        
        Args:
            conn: psycopg2 connection
            timeframe: Timeframe name (e.g., '5min')
        """
        table_name = f"ohlcv_{timeframe.replace('min', 'm').replace('h', 'H').replace('d', 'D')}"
        
        logger.info(f"Creating table: {table_name}")
        
        with conn.cursor() as cur:
            # Drop existing table if needed (comment out for production)
            # cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            
            # Create table with all columns
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    
                    -- OHLCV
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
                    
                    -- Momentum
                    rsi DOUBLE PRECISION,
                    macd DOUBLE PRECISION,
                    macd_signal DOUBLE PRECISION,
                    macd_histogram DOUBLE PRECISION,
                    
                    -- Bollinger Bands
                    bb_middle DOUBLE PRECISION,
                    bb_upper DOUBLE PRECISION,
                    bb_lower DOUBLE PRECISION,
                    bb_width DOUBLE PRECISION,
                    
                    -- Volatility
                    atr DOUBLE PRECISION,
                    
                    -- Trend
                    supertrend DOUBLE PRECISION,
                    supertrend_direction INTEGER,
                    psar DOUBLE PRECISION,
                    psar_trend INTEGER,
                    
                    -- Channels
                    keltner_middle DOUBLE PRECISION,
                    keltner_upper DOUBLE PRECISION,
                    keltner_lower DOUBLE PRECISION,
                    
                    PRIMARY KEY (timestamp, symbol)
                );
            """)
            
            # Convert to hypertable
            try:
                cur.execute(f"""
                    SELECT create_hypertable(
                        '{table_name}',
                        'timestamp',
                        if_not_exists => TRUE,
                        chunk_time_interval => INTERVAL '1 week'
                    );
                """)
                logger.info(f"  ✓ Created hypertable: {table_name}")
            except psycopg2.errors.DuplicateTable:
                logger.info(f"  ✓ Hypertable already exists: {table_name}")
            
            # Create indexes
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_time 
                ON {table_name} (symbol, timestamp DESC);
            """)
            
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol 
                ON {table_name} (symbol);
            """)
            
            conn.commit()
            
            self.stats['tables_created'] += 1
    
    def _insert_symbol_timeframe(self, conn, symbol: str, timeframe: str) -> None:
        """
        Insert all Parquet files for a symbol/timeframe combination.
        
        Args:
            conn: psycopg2 connection
            symbol: Currency pair
            timeframe: Timeframe
        """
        table_name = f"ohlcv_{timeframe.replace('min', 'm').replace('h', 'H').replace('d', 'D')}"
        parquet_path = self.parquet_dir / symbol / timeframe
        
        if not parquet_path.exists():
            return
        
        parquet_files = list(parquet_path.glob("*.parquet"))
        
        if not parquet_files:
            return
        
        logger.info(f"\nInserting {symbol} {timeframe} ({len(parquet_files)} files)...")
        
        total_rows = 0
        
        for parquet_file in sorted(parquet_files):
            try:
                # Load Parquet
                df = pd.read_parquet(parquet_file)
                
                if df.empty:
                    continue
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Reset index to get timestamp as column
                df = df.reset_index()
                
                # Get all columns present in DataFrame
                columns = list(df.columns)
                
                # Prepare data tuples
                data_tuples = [tuple(row) for row in df.values]
                
                # Build INSERT query dynamically based on available columns
                columns_str = ', '.join(columns)
                placeholders = ', '.join(['%s'] * len(columns))
                
                # Insert with ON CONFLICT UPDATE
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {table_name} ({columns_str})
                        VALUES %s
                        ON CONFLICT (timestamp, symbol) DO UPDATE SET
                            {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['timestamp', 'symbol']])}
                        """,
                        data_tuples,
                        page_size=1000
                    )
                
                conn.commit()
                
                rows = len(df)
                total_rows += rows
                logger.info(f"  ✓ {parquet_file.name}: {rows:,} rows")
                
            except Exception as e:
                error_msg = f"Failed to insert {parquet_file.name}: {str(e)}"
                logger.error(f"  ✗ {error_msg}")
                self.stats['errors'].append(error_msg)
                conn.rollback()
        
        if total_rows > 0:
            logger.info(f"  ✓ Total inserted: {total_rows:,} rows")
            
            if timeframe not in self.stats['rows_per_timeframe']:
                self.stats['rows_per_timeframe'][timeframe] = 0
            self.stats['rows_per_timeframe'][timeframe] += total_rows
            self.stats['total_rows_inserted'] += total_rows
    
    def _print_statistics(self) -> None:
        """Print insertion statistics"""
        logger.info("\n📊 Insertion Summary:")
        logger.info("-" * 80)
        logger.info(f"Tables created: {self.stats['tables_created']}")
        logger.info(f"Total rows inserted: {self.stats['total_rows_inserted']:,}")
        
        if self.stats['rows_per_timeframe']:
            logger.info("\nRows per timeframe:")
            for tf, count in sorted(self.stats['rows_per_timeframe'].items()):
                logger.info(f"  {tf}: {count:,}")
        
        if self.stats['errors']:
            logger.warning(f"\n⚠️  Errors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:10]:
                logger.warning(f"  - {error}")
            if len(self.stats['errors']) > 10:
                logger.warning(f"  ... and {len(self.stats['errors'])-10} more")
        else:
            logger.info("\n✓ No errors")
        
        logger.info("-" * 80)


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description='Insert Parquet data with indicators into TimescaleDB'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Insert all available data'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to insert'
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        choices=['1min', '5min', '15min', '30min', '1h', '4h', '1d'],
        help='Specific timeframes to insert'
    )
    
    parser.add_argument(
        '--parquet-dir',
        type=str,
        default='data/parquet',
        help='Parquet directory (default: data/parquet)'
    )
    
    args = parser.parse_args()
    
    # Determine what to insert
    symbols = args.symbols if not args.all else None
    timeframes = args.timeframes
    
    # Run insertion
    inserter = IndicatorDatabaseInserter(parquet_dir=args.parquet_dir)
    
    stats = inserter.insert_all(symbols=symbols, timeframes=timeframes)
    
    return stats


if __name__ == '__main__':
    main()
