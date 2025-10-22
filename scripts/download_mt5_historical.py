"""
MT5 Historical Data Downloader - Backfill Script

Downloads historical OHLCV data from MT5 broker and stores in TimescaleDB.
Supports multiple symbols, timeframes, and date ranges.

Usage:
    poetry run python scripts/download_mt5_historical.py --start 2024-01-01 --end 2024-12-31
    
    # Single symbol
    poetry run python scripts/download_mt5_historical.py --symbol EURUSD --timeframe M1
    
    # Multiple symbols with custom range
    poetry run python scripts/download_mt5_historical.py --symbols EURUSD GBPUSD --start 2023-01-01

Business Goal: 
    - Backfill 1 year of M1 data for DRL training (~500k bars/symbol)
    - Populate TimescaleDB ohlcv table for StateVectorBuilder
    - Enable historical regime detection and feature calculation
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
from typing import List
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from underdog.data.mt5_historical_loader import MT5HistoricalDataLoader
from underdog.database.timescale.timescale_connector import TimescaleDBConnector
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataBackfill:
    """
    Orchestrates historical data download and insertion to TimescaleDB
    """
    
    def __init__(self, config_path: str = "config/data_providers.yaml"):
        self.config = self._load_config(config_path)
        self.db = TimescaleDBConnector(
            host=self.config['timescaledb']['host'],
            port=self.config['timescaledb']['port'],
            database=self.config['timescaledb']['database'],
            user=self.config['timescaledb']['user'],
            password=self.config['timescaledb']['password']
        )
    
    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    async def download_and_insert(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframes: List[str] = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
    ):
        """
        Download data from MT5 and insert into TimescaleDB
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            timeframes: List of timeframes to download
        """
        logger.info(f"{'='*80}")
        logger.info(f"BACKFILL: {symbol} ({start_date} to {end_date})")
        logger.info(f"{'='*80}")
        
        # Connect to TimescaleDB
        await self.db.connect()
        
        # Download each timeframe
        for tf in timeframes:
            try:
                logger.info(f"\n[{symbol}] Downloading {tf} data...")
                
                # Download from MT5
                async with MT5HistoricalDataLoader() as loader:
                    df = await loader.get_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=tf
                    )
                
                if df.empty:
                    logger.warning(f"  No data returned for {symbol} {tf}")
                    continue
                
                logger.info(f"  Downloaded {len(df)} bars")
                logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
                # Insert into TimescaleDB (batch insert for performance)
                await self._batch_insert(df, symbol, tf)
                
                logger.info(f"  ✓ Inserted {len(df)} rows into ohlcv table")
                
            except Exception as e:
                logger.error(f"  ✗ Error downloading {symbol} {tf}: {e}")
                continue
        
        await self.db.disconnect()
    
    async def _batch_insert(self, df: pd.DataFrame, symbol: str, timeframe: str, batch_size: int = 10000):
        """
        Insert data in batches for performance
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe string
            batch_size: Number of rows per batch (10k recommended)
        """
        total_rows = len(df)
        batches = (total_rows + batch_size - 1) // batch_size
        
        logger.info(f"  Inserting {total_rows} rows in {batches} batches...")
        
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Prepare records for asyncpg
            records = [
                (
                    row['timestamp'],
                    symbol,
                    timeframe,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']),
                    float(row.get('spread', 0.0))
                )
                for _, row in batch.iterrows()
            ]
            
            # Batch insert with ON CONFLICT DO NOTHING (skip duplicates)
            await self.db.pool.executemany(
                """
                INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume, spread)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (time, symbol, timeframe) DO NOTHING
                """,
                records
            )
            
            logger.info(f"  Batch {i//batch_size + 1}/{batches} inserted ({len(records)} rows)")
    
    async def verify_data(self, symbol: str, timeframe: str):
        """
        Verify data was inserted correctly
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to verify
        """
        await self.db.connect()
        
        query = """
        SELECT 
            COUNT(*) as total_bars,
            MIN(time) as first_bar,
            MAX(time) as last_bar,
            AVG(volume) as avg_volume
        FROM ohlcv
        WHERE symbol = $1 AND timeframe = $2
        """
        
        result = await self.db.pool.fetchrow(query, symbol, timeframe)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"VERIFICATION: {symbol} {timeframe}")
        logger.info(f"{'='*80}")
        logger.info(f"Total bars: {result['total_bars']:,}")
        logger.info(f"Date range: {result['first_bar']} to {result['last_bar']}")
        logger.info(f"Avg volume: {result['avg_volume']:.2f}")
        logger.info(f"{'='*80}\n")
        
        await self.db.disconnect()


async def main():
    parser = argparse.ArgumentParser(description='Download MT5 historical data and store in TimescaleDB')
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
        help='Symbols to download (default: EURUSD GBPUSD USDJPY XAUUSD)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        help='Single symbol to download (overrides --symbols)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        help='Start date YYYY-MM-DD (default: 1 year ago)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date YYYY-MM-DD (default: today)'
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'],
        help='Timeframes to download (default: M1 M5 M15 M30 H1 H4 D1)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify data after download'
    )
    
    args = parser.parse_args()
    
    # Use single symbol if provided, otherwise use list
    symbols = [args.symbol] if args.symbol else args.symbols
    
    logger.info("="*80)
    logger.info("MT5 HISTORICAL DATA BACKFILL")
    logger.info("="*80)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Timeframes: {', '.join(args.timeframes)}")
    logger.info("="*80 + "\n")
    
    # Initialize backfill orchestrator
    backfill = HistoricalDataBackfill()
    
    # Download each symbol
    for symbol in symbols:
        try:
            await backfill.download_and_insert(
                symbol=symbol,
                start_date=args.start,
                end_date=args.end,
                timeframes=args.timeframes
            )
            
            # Verify if requested
            if args.verify:
                for tf in args.timeframes:
                    await backfill.verify_data(symbol, tf)
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    logger.info("\n" + "="*80)
    logger.info("BACKFILL COMPLETE")
    logger.info("="*80)
    logger.info("Next steps:")
    logger.info("  1. Verify data: docker exec timescaledb psql -U underdog -c 'SELECT * FROM ohlcv LIMIT 10'")
    logger.info("  2. Start orchestrator: poetry run python underdog/database/timescale/run_orchestrator.py")
    logger.info("  3. Train TD3 agent: poetry run python underdog/rl/train_drl.py")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())
