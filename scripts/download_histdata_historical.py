"""
HistData.com Historical Data Downloader - Robust Forex/Index Data

Downloads historical OHLCV data from HistData.com (reliable, established provider).
Known for complete data archives with no service interruptions.

Advantages:
- Stable HTTP downloads (no 503 errors)
- Complete historical archives (10+ years)
- M1 granularity available
- Free access for retail traders
- EST timezone (converted to UTC)

Asset Coverage (6 symbols):
- EURUSD, GBPUSD, USDJPY (Trading majors)
- XAUUSD (Safe haven)
- SPXUSD (S&P 500 - Risk regime indicator)
- USDXUSD (US Dollar Index - USD strength)

Usage:
    poetry run python scripts/download_histdata_historical.py --start 2022-01 --end 2024-10
    
    # Single symbol
    poetry run python scripts/download_histdata_historical.py --symbol EURUSD --year 2024
    
    # All 6 symbols
    poetry run python scripts/download_histdata_historical.py --all

Business Goal:
    - Robust backfill for DRL training
    - 3 years of M1 data (~1M bars/symbol)
    - Populate ohlcv table in TimescaleDB
    - Enable macro regime detection (SPX, DXY)
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
from typing import List, Dict
import yaml
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import zipfile
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF
from histdata.api import Platform as P, TimeFrame as TF

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from underdog.database.timescale.timescale_connector import TimescaleDBConnector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class HistDataDownloader:
    """
    Download historical data from HistData.com using histdata library
    
    HistData.com provides free M1 tick data via the histdata Python library.
    Format: CSV files (Date,Time,Open,High,Low,Close,Volume)
    Timezone: GMT+0 (UTC after library processing)
    """
    
    # Asset mapping: Our symbol → HistData library symbol
    SYMBOL_MAP = {
        # Trading pairs (4 symbols) - Available in HistData.com
        'EURUSD': 'eurusd',
        'GBPUSD': 'gbpusd',
        'USDJPY': 'usdjpy',
        'XAUUSD': 'xauusd',
        
        # Macro indicators NOT available in HistData.com
        # Will use FRED instead: VIX (SPX proxy), DXY data from other sources
        # 'SPXUSD': 'spxusd',  # NOT AVAILABLE
        # 'USDXUSD': 'usdx',   # NOT AVAILABLE
    }
    
    def __init__(self, cache_dir: str = "data/histdata"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_year_csv(self, symbol: str, year: int) -> pd.DataFrame:
        """
        Download M1 data for an entire year using histdata library
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            year: Year (e.g., 2022)
        
        Returns:
            DataFrame with OHLCV data (UTC timezone)
        """
        histdata_symbol = self.SYMBOL_MAP.get(symbol, symbol.lower())
        
        cache_file = self.cache_dir / f"{symbol}_{year}_full.csv"
        
        # Check cache first
        if cache_file.exists():
            logger.debug(f"Loading from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            return df
        
        try:
            logger.info(f"Downloading {symbol} {year} (full year) via histdata library...")
            
            # Download using histdata library (full year)
            # For past years, use month=None to download entire year
            dl(
                pair=histdata_symbol,
                year=year,
                month=None,  # Download entire year
                platform=P.GENERIC_ASCII,
                time_frame=TF.ONE_MINUTE,
                output_directory=str(self.cache_dir)
            )
            
            # Define ZIP file name (format: DAT_ASCII_EURUSD_M1_2022.zip)
            zip_file_name = f"DAT_ASCII_{histdata_symbol.upper()}_M1_{year}.zip"
            zip_file_path = self.cache_dir / zip_file_name
            
            if not zip_file_path.exists():
                logger.warning(f"ZIP file not found after download: {zip_file_path}")
                return pd.DataFrame()
            
            # Extract CSV from ZIP
            logger.info(f"Extracting ZIP: {zip_file_name}")
            with zipfile.ZipFile(zip_file_path, 'r') as zf:
                # Find CSV file inside ZIP
                csv_in_zip_list = [name for name in zf.namelist() if name.endswith('.csv')]
                
                if not csv_in_zip_list:
                    logger.error(f"No CSV file found inside ZIP: {zip_file_path}")
                    zip_file_path.unlink(missing_ok=True)
                    return pd.DataFrame()
                
                csv_file_name = csv_in_zip_list[0]
                # Extract to cache directory
                zf.extract(csv_file_name, path=self.cache_dir)
                logger.debug(f"Extracted {csv_file_name} from ZIP")
            
            # Delete ZIP after extraction to save space
            zip_file_path.unlink(missing_ok=True)
            
            # Path to extracted CSV
            downloaded_file = self.cache_dir / csv_file_name
            
            # Read CSV
            # HistData format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
            # First column contains datetime as "20220102 180000"
            df = pd.read_csv(downloaded_file, sep=';', header=None,
                           names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'])
            
            # Process data
            df = self._process_histdata_csv(df, symbol)
            
            # Cache processed data
            df.to_csv(cache_file, index=False)
            logger.info(f"✓ Downloaded {len(df)} bars for {symbol} {year}")
            
            # Cleanup: Remove original downloaded file
            downloaded_file.unlink(missing_ok=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error downloading {symbol} {year}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def download_month_csv(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        """
        Download M1 data for a specific month using histdata library
        (Only for current year)
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            year: Year (e.g., 2024)
            month: Month (1-12)
        
        Returns:
            DataFrame with OHLCV data (UTC timezone)
        """
        histdata_symbol = self.SYMBOL_MAP.get(symbol, symbol.lower())
        
        cache_file = self.cache_dir / f"{symbol}_{year}_{month:02d}.csv"
        
        # Check cache first
        if cache_file.exists():
            logger.debug(f"Loading from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            return df
        
        try:
            logger.info(f"Downloading {symbol} {year}-{month:02d} via histdata library...")
            
            # Download using histdata library
            # For current year, specify month
            dl(
                pair=histdata_symbol,
                year=year,
                month=month,
                platform=P.GENERIC_ASCII,
                time_frame=TF.ONE_MINUTE,
                output_directory=str(self.cache_dir)
            )
            
            # Define ZIP file name (format: DAT_ASCII_EURUSD_M1_202410.zip)
            zip_file_name = f"DAT_ASCII_{histdata_symbol.upper()}_M1_{year}{month:02d}.zip"
            zip_file_path = self.cache_dir / zip_file_name
            
            if not zip_file_path.exists():
                logger.warning(f"ZIP file not found after download: {zip_file_path}")
                return pd.DataFrame()
            
            # Extract CSV from ZIP
            logger.info(f"Extracting ZIP: {zip_file_name}")
            with zipfile.ZipFile(zip_file_path, 'r') as zf:
                # Find CSV file inside ZIP
                csv_in_zip_list = [name for name in zf.namelist() if name.endswith('.csv')]
                
                if not csv_in_zip_list:
                    logger.error(f"No CSV file found inside ZIP: {zip_file_path}")
                    zip_file_path.unlink(missing_ok=True)
                    return pd.DataFrame()
                
                csv_file_name = csv_in_zip_list[0]
                # Extract to cache directory
                zf.extract(csv_file_name, path=self.cache_dir)
                logger.debug(f"Extracted {csv_file_name} from ZIP")
            
            # Delete ZIP after extraction
            zip_file_path.unlink(missing_ok=True)
            
            # Path to extracted CSV
            downloaded_file = self.cache_dir / csv_file_name
            
            # Read CSV
            # HistData format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
            # First column contains datetime as "20220102 180000"
            df = pd.read_csv(downloaded_file, sep=';', header=None,
                           names=['datetime_str', 'open', 'high', 'low', 'close', 'volume'])
            
            # Process data
            df = self._process_histdata_csv(df, symbol)
            
            # Cache processed data
            df.to_csv(cache_file, index=False)
            logger.info(f"✓ Downloaded {len(df)} bars for {symbol} {year}-{month:02d}")
            
            # Cleanup: Remove original downloaded file
            downloaded_file.unlink(missing_ok=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error downloading {symbol} {year}-{month:02d}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _process_histdata_csv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process HistData.com CSV format to our OHLCV format
        
        HistData format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
        Example: 20220102 180000;1828.604000;1829.628000;1828.544000;1829.504000;0
        
        First column contains datetime as "20220102 180000" (space-separated)
        Timezone: EST (no DST adjustment)
        
        Our format: timestamp (UTC), open, high, low, close, volume, spread
        """
        if df.empty:
            return pd.DataFrame()
        
        try:
            # Parse datetime from first column
            # Format: "YYYYMMDD HHMMSS" -> "20220102 180000"
            df['timestamp'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H%M%S')
            
            # Convert from EST to UTC
            # EST is UTC-5 (HistData does not adjust for DST)
            df['timestamp'] = df['timestamp'].dt.tz_localize('EST', ambiguous='infer', nonexistent='shift_forward')
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
            
            # Remove any NaT rows
            df = df.dropna(subset=['timestamp'])
            
            # Calculate spread (estimate as high-low)
            # Real spread would require bid/ask data
            df['spread'] = (df['high'] - df['low']).abs()
            
            # Select and rename columns
            result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']].copy()
            
            # Sort by timestamp
            result = result.sort_values('timestamp').reset_index(drop=True)
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing CSV for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def download_date_range(
        self,
        symbol: str,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int
    ) -> pd.DataFrame:
        """
        Download data for a date range
        
        Strategy:
        - Past years: Download entire year at once (more efficient)
        - Current year: Download month by month
        
        Args:
            symbol: Trading symbol
            start_year: Start year
            start_month: Start month (1-12)
            end_year: End year
            end_month: End month (1-12)
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Downloading {symbol} from {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        
        all_data = []
        current_year = datetime.now().year
        
        # Download complete past years
        for year in range(start_year, end_year + 1):
            if year < current_year or (year == current_year and year < end_year):
                # Download entire year
                df_year = self.download_year_csv(symbol, year)
                
                if not df_year.empty:
                    # Filter by date range if needed
                    if year == start_year:
                        # Filter start month
                        start_date = datetime(start_year, start_month, 1, tzinfo=ZoneInfo('UTC'))
                        df_year = df_year[df_year['timestamp'] >= start_date]
                    
                    if year == end_year and year != current_year:
                        # Filter end month (only if not current year)
                        end_date = datetime(end_year, end_month + 1, 1, tzinfo=ZoneInfo('UTC'))
                        df_year = df_year[df_year['timestamp'] < end_date]
                    
                    all_data.append(df_year)
        
        # Download current year months if needed
        if end_year == current_year:
            for month in range(1, end_month + 1):
                # Skip if already covered by full year download
                if current_year != start_year or month >= start_month:
                    df_month = self.download_month_csv(symbol, current_year, month)
                    if not df_month.empty:
                        all_data.append(df_month)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # Remove duplicates and sort
            result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            logger.info(f"✓ Total: {len(result)} bars for {symbol}")
            return result
        else:
            logger.warning(f"No data downloaded for {symbol}")
            return pd.DataFrame()


async def main():
    parser = argparse.ArgumentParser(description='Download HistData.com historical data')
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],  # 4 trading pairs (SPXUSD/USDXUSD not available in HistData)
        help='Symbols to download (default: 4 core trading pairs)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        help='Single symbol to download (overrides --symbols)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all 4 available symbols'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default='2022-01',
        help='Start date YYYY-MM (default: 2022-01)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m'),
        help='End date YYYY-MM (default: current month)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify data after download'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_year, start_month = map(int, args.start.split('-'))
    end_year, end_month = map(int, args.end.split('-'))
    
    # Use single symbol if provided, else use all 4
    if args.symbol:
        symbols = [args.symbol]
    elif args.all:
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']  # Only 4 available in HistData
    else:
        symbols = args.symbols
    
    logger.info("="*80)
    logger.info("HISTDATA.COM HISTORICAL DATA DOWNLOAD")
    logger.info("="*80)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Timeframe: M1 (1-minute bars)")
    logger.info("="*80 + "\n")
    
    # Load config
    with open('config/data_providers.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize DB
    db = TimescaleDBConnector(
        host=config['timescaledb']['host'],
        port=config['timescaledb']['port'],
        database=config['timescaledb']['database'],
        user=config['timescaledb']['user'],
        password=config['timescaledb']['password']
    )
    
    await db.connect()
    
    # Initialize downloader
    downloader = HistDataDownloader()
    
    # Download each symbol
    for symbol in symbols:
        try:
            # Download data
            df = downloader.download_date_range(
                symbol,
                start_year,
                start_month,
                end_year,
                end_month
            )
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue
            
            # Insert into TimescaleDB
            logger.info(f"Inserting {len(df)} rows into TimescaleDB...")
            
            # M1 timeframe for all symbols
            timeframe = 'M1'
            
            records = [
                (
                    row['timestamp'],
                    symbol,
                    timeframe,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']) if not pd.isna(row['volume']) else 0,
                    float(row['spread']) if not pd.isna(row['spread']) else 0.0
                )
                for _, row in df.iterrows()
            ]
            
            # Batch insert
            batch_size = 10000
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                await db.pool.executemany(
                    """
                    INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume, spread)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (time, symbol, timeframe) DO NOTHING
                    """,
                    batch
                )
                logger.info(f"  Batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1} inserted")
            
            logger.info(f"✓ {symbol} complete\n")
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Verify if requested
    if args.verify:
        for symbol in symbols:
            query = """
            SELECT timeframe, COUNT(*) as bars, MIN(time) as first, MAX(time) as last
            FROM ohlcv
            WHERE symbol = $1
            GROUP BY timeframe
            ORDER BY timeframe
            """
            results = await db.pool.fetch(query, symbol)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"VERIFICATION: {symbol}")
            logger.info(f"{'='*80}")
            for row in results:
                logger.info(f"{row['timeframe']:5} | {row['bars']:8,} bars | {row['first']} to {row['last']}")
    
    await db.disconnect()
    
    logger.info("\n" + "="*80)
    logger.info("BACKFILL COMPLETE")
    logger.info("="*80)
    logger.info("Next steps:")
    logger.info("  1. Calculate features: poetry run python scripts/calculate_historical_features.py --all")
    logger.info("  2. Backfill FRED: poetry run python scripts/backfill_fred_macro.py")
    logger.info("  3. Backfill Reddit: poetry run python scripts/backfill_reddit_sentiment.py")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())
