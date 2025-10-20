"""
Enhanced Backfill Script - HistData + Parquet + TimescaleDB

Features:
- Downloads forex data directly using histdata library
- Saves to Parquet format (70-80% smaller than CSV)
- Inserts into TimescaleDB for fast querying
- Supports multiple currency pairs and timeframes
- Progress tracking and error handling

Usage:
    poetry run python scripts/backfill_histdata_parquet.py --symbols EURUSD GBPUSD --year 2024
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd
import argparse
import logging
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF

# Add underdog to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistDataParquetBackfill:
    """
    Download forex data from HistData.com and save to Parquet + TimescaleDB.
    
    This class handles:
    1. Downloading tick/minute data using histdata library
    2. Converting to OHLCV format
    3. Saving to Parquet files (compressed, efficient)
    4. Inserting into TimescaleDB for querying
    """
    
    SUPPORTED_PAIRS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
        'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY'
    ]
    
    def __init__(
        self,
        symbols: List[str],
        year: int,
        months: Optional[List[int]] = None,
        output_dir: str = "data/parquet",
        timeframes: List[str] = ['1min', '5min', '15min', '1h']
    ):
        """
        Initialize backfill pipeline.
        
        Args:
            symbols: List of currency pairs (e.g., ['EURUSD', 'GBPUSD'])
            year: Year to download (e.g., 2024)
            months: Months to download (e.g., [1,2,3]). If None, downloads all 12 months
            output_dir: Directory to save Parquet files
            timeframes: List of timeframes to generate
        """
        self.symbols = [s.upper() for s in symbols]
        self.year = year
        self.months = months or list(range(1, 13))
        self.output_dir = Path(output_dir)
        self.timeframes = timeframes
        
        # Validate symbols
        for symbol in self.symbols:
            if symbol not in self.SUPPORTED_PAIRS:
                logger.warning(f"Symbol {symbol} may not be available on HistData.com")
        
        # Create output directories
        for symbol in self.symbols:
            (self.output_dir / symbol).mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("HistData + Parquet Backfill Pipeline")
        logger.info("=" * 70)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Year: {year}, Months: {self.months}")
        logger.info(f"Timeframes: {', '.join(timeframes)}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 70)
        
        # Statistics
        self.stats = {
            'downloaded': 0,
            'saved': 0,
            'inserted': 0,
            'errors': []
        }
    
    def download_and_process(self) -> Dict:
        """
        Main pipeline: Download → Parquet → TimescaleDB
        
        Returns:
            Dictionary with execution statistics
        """
        logger.info("\n[STEP 1/3] Downloading data from HistData.com")
        logger.info("-" * 70)
        
        for symbol in self.symbols:
            self._download_symbol(symbol)
        
        logger.info("\n[STEP 2/3] Converting to multiple timeframes")
        logger.info("-" * 70)
        
        self._convert_timeframes()
        
        logger.info("\n[STEP 3/3] Inserting into TimescaleDB")
        logger.info("-" * 70)
        
        self._insert_to_database()
        
        logger.info("\n" + "=" * 70)
        logger.info("BACKFILL COMPLETE")
        logger.info("=" * 70)
        self._print_statistics()
        
        return self.stats
    
    def _download_symbol(self, symbol: str) -> None:
        """
        Download data for a single symbol using histdata library.
        
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
        """
        logger.info(f"\nProcessing {symbol}...")
        
        for month in self.months:
            try:
                logger.info(f"  Downloading {self.year}/{month:02d}...")
                
                # Download using histdata library
                # Note: histdata library returns data in specific format
                # We'll download tick data and convert to OHLCV
                
                # Download tick data (this is a placeholder - actual API may differ)
                # The histdata library downloads and returns a DataFrame
                data = dl(
                    currency_pair=symbol.lower(),
                    year=self.year,
                    month=month,
                    platform=P.GENERIC_ASCII,  # Generic ASCII format
                    time_frame=TF.TICK_DATA     # Tick data
                )
                
                if data is None or data.empty:
                    logger.warning(f"    No data for {symbol} {self.year}/{month:02d}")
                    continue
                
                # Save raw tick data to Parquet
                filename = f"{symbol}_{self.year}{month:02d}_ticks.parquet"
                filepath = self.output_dir / symbol / filename
                
                data.to_parquet(
                    filepath,
                    engine='fastparquet',
                    compression='snappy',
                    index=True
                )
                
                logger.info(f"    ✓ Downloaded {len(data):,} ticks → {filename}")
                self.stats['downloaded'] += len(data)
                self.stats['saved'] += 1
                
            except Exception as e:
                error_msg = f"Failed to download {symbol} {self.year}/{month:02d}: {str(e)}"
                logger.error(f"    ✗ {error_msg}")
                self.stats['errors'].append(error_msg)
    
    def _convert_timeframes(self) -> None:
        """
        Convert tick data to OHLCV for multiple timeframes.
        """
        logger.info("\nConverting tick data to OHLCV timeframes...")
        
        for symbol in self.symbols:
            symbol_dir = self.output_dir / symbol
            tick_files = list(symbol_dir.glob(f"{symbol}_*_ticks.parquet"))
            
            if not tick_files:
                logger.warning(f"  No tick data found for {symbol}")
                continue
            
            logger.info(f"\n  Processing {symbol} ({len(tick_files)} files)...")
            
            # Load all tick data for the symbol
            all_ticks = []
            for tick_file in tick_files:
                try:
                    ticks = pd.read_parquet(tick_file)
                    all_ticks.append(ticks)
                except Exception as e:
                    logger.error(f"    Failed to load {tick_file.name}: {e}")
            
            if not all_ticks:
                continue
            
            # Concatenate all ticks
            combined_ticks = pd.concat(all_ticks, ignore_index=False)
            combined_ticks = combined_ticks.sort_index()
            
            # Ensure we have bid/ask columns (histdata format varies)
            # Typically: timestamp, bid, ask
            if 'bid' not in combined_ticks.columns:
                if 'close' in combined_ticks.columns:
                    combined_ticks['bid'] = combined_ticks['close']
                    combined_ticks['ask'] = combined_ticks['close']
            
            # Convert to each timeframe
            for tf in self.timeframes:
                try:
                    ohlcv = self._resample_to_timeframe(combined_ticks, tf)
                    
                    if ohlcv.empty:
                        continue
                    
                    # Save to Parquet
                    filename = f"{symbol}_{self.year}_{tf}.parquet"
                    filepath = symbol_dir / filename
                    
                    ohlcv.to_parquet(
                        filepath,
                        engine='fastparquet',
                        compression='snappy',
                        index=True
                    )
                    
                    logger.info(f"    ✓ {tf}: {len(ohlcv):,} bars → {filename}")
                    self.stats['saved'] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to convert {symbol} to {tf}: {str(e)}"
                    logger.error(f"    ✗ {error_msg}")
                    self.stats['errors'].append(error_msg)
    
    def _resample_to_timeframe(self, tick_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample tick data to OHLCV bars.
        
        Args:
            tick_data: DataFrame with tick data (bid/ask columns)
            timeframe: Target timeframe ('1min', '5min', '15min', '1h')
        
        Returns:
            DataFrame with OHLCV data
        """
        # Map timeframe to pandas resample rule
        tf_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        if timeframe not in tf_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        rule = tf_map[timeframe]
        
        # Use bid prices for OHLCV (or average of bid/ask)
        if 'bid' in tick_data.columns:
            price = tick_data['bid']
        elif 'close' in tick_data.columns:
            price = tick_data['close']
        else:
            raise ValueError("No price column found (bid, ask, or close)")
        
        # Resample to OHLCV
        ohlcv = pd.DataFrame({
            'open': price.resample(rule).first(),
            'high': price.resample(rule).max(),
            'low': price.resample(rule).min(),
            'close': price.resample(rule).last(),
            'volume': price.resample(rule).count()  # Tick count as volume
        })
        
        # Remove NaN rows
        ohlcv = ohlcv.dropna()
        
        return ohlcv
    
    def _insert_to_database(self) -> None:
        """
        Insert Parquet data into TimescaleDB.
        """
        logger.info("\nInserting data into TimescaleDB...")
        
        try:
            import psycopg2
            from psycopg2.extras import execute_values
            
            # Connection parameters (from docker-compose.yml)
            conn_params = {
                'host': 'localhost',
                'port': 5432,
                'database': 'underdog',
                'user': 'underdog',
                'password': 'underdog_password'
            }
            
            logger.info(f"  Connecting to TimescaleDB at {conn_params['host']}:{conn_params['port']}...")
            
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    # Create table if not exists
                    self._create_ohlcv_table(cur)
                    conn.commit()
                    
                    # Insert each Parquet file
                    for symbol in self.symbols:
                        symbol_dir = self.output_dir / symbol
                        
                        for tf in self.timeframes:
                            parquet_file = symbol_dir / f"{symbol}_{self.year}_{tf}.parquet"
                            
                            if not parquet_file.exists():
                                continue
                            
                            try:
                                # Load Parquet
                                df = pd.read_parquet(parquet_file)
                                
                                if df.empty:
                                    continue
                                
                                # Prepare data for insertion
                                df['symbol'] = symbol
                                df['timeframe'] = tf
                                df = df.reset_index()  # timestamp becomes column
                                
                                # Insert data
                                data_tuples = [
                                    (
                                        row['timestamp'] if 'timestamp' in row.index else row.name,
                                        row['symbol'],
                                        row['timeframe'],
                                        row['open'],
                                        row['high'],
                                        row['low'],
                                        row['close'],
                                        int(row['volume'])
                                    )
                                    for _, row in df.iterrows()
                                ]
                                
                                # Batch insert
                                execute_values(
                                    cur,
                                    """
                                    INSERT INTO ohlcv_data 
                                    (timestamp, symbol, timeframe, open, high, low, close, volume)
                                    VALUES %s
                                    ON CONFLICT (timestamp, symbol, timeframe) DO UPDATE SET
                                        open = EXCLUDED.open,
                                        high = EXCLUDED.high,
                                        low = EXCLUDED.low,
                                        close = EXCLUDED.close,
                                        volume = EXCLUDED.volume
                                    """,
                                    data_tuples
                                )
                                
                                conn.commit()
                                
                                logger.info(f"    ✓ Inserted {len(df):,} bars: {symbol} {tf}")
                                self.stats['inserted'] += len(df)
                                
                            except Exception as e:
                                error_msg = f"Failed to insert {parquet_file.name}: {str(e)}"
                                logger.error(f"    ✗ {error_msg}")
                                self.stats['errors'].append(error_msg)
                                conn.rollback()
            
            logger.info("  ✓ Database insertion complete")
            
        except ImportError:
            logger.warning("  psycopg2 not installed. Skipping database insertion.")
            logger.warning("  Install with: poetry add psycopg2-binary")
            self.stats['errors'].append("psycopg2 not available - database insertion skipped")
        
        except Exception as e:
            logger.error(f"  ✗ Database connection failed: {str(e)}")
            logger.warning("  Make sure TimescaleDB is running (docker-compose up -d)")
            self.stats['errors'].append(f"Database error: {str(e)}")
    
    def _create_ohlcv_table(self, cursor) -> None:
        """
        Create OHLCV table with TimescaleDB hypertable.
        
        Args:
            cursor: psycopg2 cursor
        """
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume BIGINT NOT NULL,
                PRIMARY KEY (timestamp, symbol, timeframe)
            );
        """)
        
        # Convert to hypertable (TimescaleDB)
        cursor.execute("""
            SELECT create_hypertable(
                'ohlcv_data',
                'timestamp',
                if_not_exists => TRUE,
                chunk_time_interval => INTERVAL '1 day'
            );
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time 
            ON ohlcv_data (symbol, timeframe, timestamp DESC);
        """)
        
        logger.info("  ✓ Created/verified ohlcv_data table (hypertable)")
    
    def _print_statistics(self) -> None:
        """Print execution summary"""
        logger.info("\n📊 Execution Summary:")
        logger.info("-" * 70)
        logger.info(f"Ticks downloaded: {self.stats['downloaded']:,}")
        logger.info(f"Files saved: {self.stats['saved']}")
        logger.info(f"Bars inserted to DB: {self.stats['inserted']:,}")
        
        if self.stats['errors']:
            logger.warning(f"\n⚠️  Errors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:5]:
                logger.warning(f"  - {error}")
            if len(self.stats['errors']) > 5:
                logger.warning(f"  ... and {len(self.stats['errors'])-5} more")
        else:
            logger.info("\n✓ No errors")
        
        logger.info("-" * 70)


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description='Download forex data from HistData.com to Parquet + TimescaleDB'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['EURUSD', 'GBPUSD'],
        help='Currency pairs to download (default: EURUSD GBPUSD)'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Year to download (e.g., 2024)'
    )
    
    parser.add_argument(
        '--months',
        nargs='+',
        type=int,
        default=None,
        help='Months to download (1-12). If not specified, downloads all months'
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['1min', '5min', '15min', '1h'],
        help='Timeframes to generate (default: 1min 5min 15min 1h)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/parquet',
        help='Output directory for Parquet files (default: data/parquet)'
    )
    
    args = parser.parse_args()
    
    # Run backfill
    pipeline = HistDataParquetBackfill(
        symbols=args.symbols,
        year=args.year,
        months=args.months,
        output_dir=args.output_dir,
        timeframes=args.timeframes
    )
    
    stats = pipeline.download_and_process()
    
    return stats


if __name__ == '__main__':
    main()
