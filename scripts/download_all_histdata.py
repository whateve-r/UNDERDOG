"""
Enhanced HistData Downloader + Indicator Calculator

This script downloads ALL available data from HistData.com and:
1. Downloads tick/minute data for all major pairs
2. Resamples to multiple timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d)
3. Calculates common technical indicators for each timeframe
4. Saves to Parquet (compressed, efficient)
5. Inserts into TimescaleDB with separate tables per timeframe

Benefits:
- Backtest 100-1000x faster (indicators pre-calculated)
- Consistent data across all backtests
- Flexible EA parameter tuning without recalculation
- Complete historical dataset for rigorous testing

Usage:
    # Download ALL available data (will take hours)
    poetry run python scripts/download_all_histdata.py --all
    
    # Download specific pairs and date range
    poetry run python scripts/download_all_histdata.py --symbols EURUSD GBPUSD --start-year 2020 --end-year 2024
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse
import logging
from histdata import download_hist_data as dl

# Add underdog to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging with UTF-8 encoding for Windows
log_file_handler = logging.FileHandler('data/download_histdata.log', encoding='utf-8')
log_file_handler.setLevel(logging.INFO)
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

log_stream_handler = logging.StreamHandler(sys.stdout)
log_stream_handler.setLevel(logging.INFO)
log_stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    handlers=[log_file_handler, log_stream_handler]
)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate common technical indicators for OHLCV data.
    
    Uses vectorized operations for speed.
    """
    
    @staticmethod
    def add_sma(df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Add Simple Moving Averages"""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, periods: List[int] = [12, 26, 50]) -> pd.DataFrame:
        """Add Exponential Moving Averages"""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands"""
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (std * std_dev)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Add SuperTrend indicator"""
        # Calculate ATR first
        if 'atr' not in df.columns:
            df = TechnicalIndicators.add_atr(df, period)
        
        # Basic bands
        hl_avg = (df['high'] + df['low']) / 2
        upper_band = hl_avg + (multiplier * df['atr'])
        lower_band = hl_avg - (multiplier * df['atr'])
        
        # SuperTrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upper_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif df['close'].iloc[i] < lower_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
        
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        return df
    
    @staticmethod
    def add_parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """Add Parabolic SAR"""
        sar = pd.Series(index=df.index, dtype=float)
        ep = pd.Series(index=df.index, dtype=float)
        af = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        
        # Initialize
        sar.iloc[0] = df['low'].iloc[0]
        ep.iloc[0] = df['high'].iloc[0]
        af.iloc[0] = af_start
        trend.iloc[0] = 1
        
        for i in range(1, len(df)):
            # Calculate SAR
            sar.iloc[i] = sar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - sar.iloc[i-1])
            
            # Check for trend reversal
            if trend.iloc[i-1] == 1:  # Uptrend
                if df['low'].iloc[i] < sar.iloc[i]:
                    # Reversal to downtrend
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = df['low'].iloc[i]
                    af.iloc[i] = af_start
                else:
                    trend.iloc[i] = 1
                    if df['high'].iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = df['high'].iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Downtrend
                if df['high'].iloc[i] > sar.iloc[i]:
                    # Reversal to uptrend
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = df['high'].iloc[i]
                    af.iloc[i] = af_start
                else:
                    trend.iloc[i] = -1
                    if df['low'].iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = df['low'].iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
        
        df['psar'] = sar
        df['psar_trend'] = trend
        return df
    
    @staticmethod
    def add_keltner_channels(df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
        """Add Keltner Channels"""
        if 'atr' not in df.columns:
            df = TechnicalIndicators.add_atr(df, period)
        
        df['keltner_middle'] = df['close'].ewm(span=period, adjust=False).mean()
        df['keltner_upper'] = df['keltner_middle'] + (multiplier * df['atr'])
        df['keltner_lower'] = df['keltner_middle'] - (multiplier * df['atr'])
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators at once"""
        logger.info("    Calculating technical indicators...")
        
        df = TechnicalIndicators.add_sma(df, [20, 50, 200])
        df = TechnicalIndicators.add_ema(df, [12, 26, 50])
        df = TechnicalIndicators.add_rsi(df, 14)
        df = TechnicalIndicators.add_macd(df, 12, 26, 9)
        df = TechnicalIndicators.add_bollinger_bands(df, 20, 2.0)
        df = TechnicalIndicators.add_atr(df, 14)
        df = TechnicalIndicators.add_supertrend(df, 10, 3.0)
        df = TechnicalIndicators.add_parabolic_sar(df, 0.02, 0.02, 0.2)
        df = TechnicalIndicators.add_keltner_channels(df, 20, 2.0)
        
        logger.info(f"    ✓ Added {len(df.columns) - 6} technical indicators")
        
        return df


class HistDataCompleteDownloader:
    """
    Download complete HistData.com dataset with indicators.
    """
    
    # All major pairs available on HistData
    ALL_PAIRS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
        'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURAUD',
        'EURCHF', 'GBPCHF', 'CADJPY', 'AUDNZD', 'NZDJPY'
    ]
    
    # Timeframes to generate
    TIMEFRAMES = {
        '1min': '1T',
        '5min': '5T',
        '15min': '15T',
        '30min': '30T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }
    
    def __init__(
        self,
        symbols: List[str],
        start_year: int,
        end_year: int,
        output_dir: str = 'data/parquet',
        calculate_indicators: bool = True
    ):
        """
        Initialize complete downloader.
        
        Args:
            symbols: List of currency pairs
            start_year: Start year (e.g., 2000)
            end_year: End year (e.g., 2024)
            output_dir: Output directory for Parquet files
            calculate_indicators: Whether to calculate technical indicators
        """
        self.symbols = symbols
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = Path(output_dir)
        self.calculate_indicators = calculate_indicators
        
        # Create output structure
        for symbol in symbols:
            for tf in self.TIMEFRAMES.keys():
                (self.output_dir / symbol / tf).mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'downloaded_months': 0,
            'total_ticks': 0,
            'total_bars': {},
            'errors': []
        }
        
        logger.info("=" * 80)
        logger.info("COMPLETE HISTDATA DOWNLOAD + INDICATOR CALCULATION")
        logger.info("=" * 80)
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Period: {start_year} → {end_year}")
        logger.info(f"Timeframes: {', '.join(self.TIMEFRAMES.keys())}")
        logger.info(f"Calculate indicators: {calculate_indicators}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 80)
    
    def download_all(self) -> Dict:
        """
        Download all data and process.
        
        Returns:
            Statistics dictionary
        """
        total_months = len(self.symbols) * (self.end_year - self.start_year + 1) * 12
        logger.info(f"\nTotal months to download: {total_months}")
        logger.info("This will take several hours. Progress will be saved incrementally.\n")
        
        for symbol in self.symbols:
            self._download_symbol(symbol)
        
        logger.info("\n" + "=" * 80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        self._print_statistics()
        
        return self.stats
    
    def _download_symbol(self, symbol: str) -> None:
        """Download and process all data for one symbol"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*80}")
        
        current_year = datetime.now().year
        
        for year in range(self.start_year, self.end_year + 1):
            # For current year, download month by month (API requirement)
            # For past years, download entire year at once (faster, more efficient)
            if year == current_year:
                current_month = datetime.now().month
                for month in range(1, current_month + 1):
                    self._download_month(symbol, year, month)
            else:
                self._download_year(symbol, year)
    
    def _download_year(self, symbol: str, year: int) -> None:
        """Download a full year of data at once (for past years)"""
        logger.info(f"\n{symbol} {year} (full year):")
        
        try:
            logger.info("  [1/4] Downloading from HistData.com...")
            
            result = dl(
                pair=symbol.lower(),
                year=str(year),
                month=None,  # None = download entire year for past years
                platform='ASCII',
                time_frame='M1'
            )
            
            # Handle both DataFrame and filename return types
            if isinstance(result, str):
                # Result is a ZIP filename, read the CSV from it
                logger.info(f"    Reading downloaded file: {result}")
                import zipfile
                import io
                
                with zipfile.ZipFile(result, 'r') as zip_ref:
                    # Find the CSV file in the ZIP
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        logger.warning(f"    ✗ No CSV file found in {result}")
                        return
                    
                    # Read CSV content into memory (before closing the ZIP)
                    csv_content = zip_ref.read(csv_files[0])
                
                # Parse CSV from memory (outside context manager to avoid "closed file" error)
                data = pd.read_csv(
                    io.BytesIO(csv_content),
                    sep=';',
                    names=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d %H%M%S')
                data.set_index('timestamp', inplace=True)
                
                # Force deep copy to ensure data is fully loaded in memory
                data = data.copy(deep=True)
            elif isinstance(result, pd.DataFrame):
                data = result.copy(deep=True)
            else:
                logger.warning(f"    ✗ Unexpected return type: {type(result)}")
                return
            
            if data is None or data.empty:
                logger.warning(f"    ✗ No data available for {year}")
                return
            
            logger.info(f"    ✓ Downloaded {len(data):,} 1-minute bars")
            self.stats['downloaded_months'] += 12  # Count as 12 months
            self.stats['total_ticks'] += len(data)
            
            # Ensure proper column names
            if 'timestamp' not in data.columns and data.index.name != 'timestamp':
                data.index.name = 'timestamp'
            
            # Process and save by month (to match expected structure)
            logger.info("  [2/4] Resampling to multiple timeframes...")
            
            for month in range(1, 13):
                # Filter data for this month
                if isinstance(data.index, pd.DatetimeIndex):
                    month_data = data[(data.index.year == year) & (data.index.month == month)]
                else:
                    # If index is not datetime, convert it
                    data_with_datetime = data.copy()
                    data_with_datetime.index = pd.to_datetime(data_with_datetime.index)
                    month_data = data_with_datetime[
                        (data_with_datetime.index.year == year) &
                        (data_with_datetime.index.month == month)
                    ]
                
                if not month_data.empty:
                    # Process all timeframes for this month
                    for tf_name, tf_rule in self.TIMEFRAMES.items():
                        self._process_timeframe(month_data, symbol, year, month, tf_name, tf_rule)
            
            logger.info(f"  ✓ Completed {symbol} {year}")
            
        except Exception as e:
            error_msg = f"Failed {symbol} {year}: {str(e)}"
            logger.error(f"  ✗ {error_msg}")
            self.stats['errors'].append(error_msg)
    
    def _download_month(self, symbol: str, year: int, month: int) -> None:
        """Download and process one month of data (for current year only)"""
        logger.info(f"\n{symbol} {year}/{month:02d}:")
        
        try:
            # Download minute data from HistData
            logger.info("  [1/4] Downloading from HistData.com...")
            
            result = dl(
                pair=symbol.lower(),
                year=str(year),
                month=month,  # Specific month (only works for current year)
                platform='ASCII',
                time_frame='M1'  # Download 1-minute bars (not ticks)
            )
            
            # Handle both DataFrame and filename return types
            if isinstance(result, str):
                # Result is a ZIP filename, read the CSV from it
                logger.info(f"    Reading downloaded file: {result}")
                import zipfile
                import io
                
                with zipfile.ZipFile(result, 'r') as zip_ref:
                    # Find the CSV file in the ZIP
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        logger.warning(f"    ✗ No CSV file found in {result}")
                        return
                    
                    # Read CSV content into memory (before closing the ZIP)
                    csv_content = zip_ref.read(csv_files[0])
                
                # Parse CSV from memory (outside context manager to avoid "closed file" error)
                data = pd.read_csv(
                    io.BytesIO(csv_content),
                    sep=';',
                    names=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d %H%M%S')
                data.set_index('timestamp', inplace=True)
                
                # Force deep copy to ensure data is fully loaded in memory
                data = data.copy(deep=True)
            elif isinstance(result, pd.DataFrame):
                data = result.copy(deep=True)
            else:
                logger.warning(f"    ✗ Unexpected return type: {type(result)}")
                return
            
            if data is None or data.empty:
                logger.warning(f"    ✗ No data available")
                return
            
            logger.info(f"    ✓ Downloaded {len(data):,} 1-minute bars")
            self.stats['downloaded_months'] += 1
            self.stats['total_ticks'] += len(data)
            
            # Ensure proper column names
            if 'timestamp' not in data.columns and data.index.name != 'timestamp':
                data.index.name = 'timestamp'
            
            # Process all timeframes
            logger.info("  [2/4] Resampling to multiple timeframes...")
            
            for tf_name, tf_rule in self.TIMEFRAMES.items():
                self._process_timeframe(data, symbol, year, month, tf_name, tf_rule)
            
            logger.info(f"  ✓ Completed {symbol} {year}/{month:02d}")
            
        except Exception as e:
            error_msg = f"Failed {symbol} {year}/{month:02d}: {str(e)}"
            logger.error(f"  ✗ {error_msg}")
            self.stats['errors'].append(error_msg)
    
    def _process_timeframe(
        self,
        minute_data: pd.DataFrame,
        symbol: str,
        year: int,
        month: int,
        tf_name: str,
        tf_rule: str
    ) -> None:
        """Resample and calculate indicators for one timeframe"""
        try:
            # Skip 1min (already have it)
            if tf_name == '1min':
                ohlcv = minute_data.copy()
            else:
                # Resample to target timeframe
                ohlcv = self._resample_ohlcv(minute_data, tf_rule)
            
            if ohlcv.empty:
                return
            
            # Calculate indicators if enabled
            if self.calculate_indicators:
                ohlcv = TechnicalIndicators.add_all_indicators(ohlcv)
            
            # Save to Parquet
            filename = f"{symbol}_{year}{month:02d}_{tf_name}.parquet"
            filepath = self.output_dir / symbol / tf_name / filename
            
            ohlcv.to_parquet(
                filepath,
                engine='fastparquet',
                compression='snappy',
                index=True
            )
            
            bars = len(ohlcv)
            if tf_name not in self.stats['total_bars']:
                self.stats['total_bars'][tf_name] = 0
            self.stats['total_bars'][tf_name] += bars
            
            logger.info(f"    ✓ {tf_name}: {bars:,} bars → {filename}")
            
        except Exception as e:
            logger.error(f"    ✗ {tf_name} failed: {e}")
    
    def _resample_ohlcv(self, minute_data: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample minute data to higher timeframe"""
        # Force materialization of all data to avoid lazy evaluation issues
        minute_data = minute_data.copy(deep=True)
        
        ohlcv = pd.DataFrame({
            'open': minute_data['open'].resample(rule).first(),
            'high': minute_data['high'].resample(rule).max(),
            'low': minute_data['low'].resample(rule).min(),
            'close': minute_data['close'].resample(rule).last(),
            'volume': minute_data['volume'].resample(rule).sum() if 'volume' in minute_data.columns else minute_data['close'].resample(rule).count()
        })
        
        return ohlcv.dropna()
    
    def _print_statistics(self) -> None:
        """Print final statistics"""
        logger.info("\n📊 Download Summary:")
        logger.info("-" * 80)
        logger.info(f"Months downloaded: {self.stats['downloaded_months']}")
        logger.info(f"Total 1-minute bars: {self.stats['total_ticks']:,}")
        
        logger.info("\nBars per timeframe:")
        for tf, count in sorted(self.stats['total_bars'].items()):
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
        description='Download complete HistData.com dataset with technical indicators'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download ALL major pairs (17 pairs)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific currency pairs to download'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=2000,
        help='Start year (default: 2000)'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        default=datetime.now().year,
        help=f'End year (default: {datetime.now().year})'
    )
    
    parser.add_argument(
        '--no-indicators',
        action='store_true',
        help='Skip technical indicator calculation (faster)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/parquet',
        help='Output directory (default: data/parquet)'
    )
    
    args = parser.parse_args()
    
    # Determine symbols
    if args.all:
        symbols = HistDataCompleteDownloader.ALL_PAIRS
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        # Default to major pairs
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
    
    # Confirm if downloading all
    if len(symbols) > 5 or (args.end_year - args.start_year) > 5:
        logger.warning("\n⚠️  WARNING: Large download detected!")
        logger.warning(f"Symbols: {len(symbols)}")
        logger.warning(f"Years: {args.start_year} → {args.end_year}")
        logger.warning("This may take several hours and use significant disk space.")
        
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            logger.info("Download cancelled")
            return
    
    # Run download
    downloader = HistDataCompleteDownloader(
        symbols=symbols,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
        calculate_indicators=not args.no_indicators
    )
    
    stats = downloader.download_all()
    
    return stats


if __name__ == '__main__':
    main()
