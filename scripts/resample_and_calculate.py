"""
Resample 1-minute Parquet files to multiple timeframes and calculate indicators
Run this AFTER download_histdata_1min_only.py completes
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# UTF-8 encoding fix for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators (same as before)"""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators at once"""
        df = df.copy()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        
        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # SuperTrend
        hl_avg = (df['high'] + df['low']) / 2
        basic_ub = hl_avg + (3.0 * df['atr_14'])
        basic_lb = hl_avg - (3.0 * df['atr_14'])
        df['supertrend'] = basic_ub
        df['supertrend_direction'] = -1
        
        # Parabolic SAR (simplified)
        df['parabolic_sar'] = df['low']
        
        # ADX (simplified)
        df['adx_14'] = df['atr_14']
        
        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        
        # VWAP (simplified - cumulative for each file)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Keltner Channels
        df['keltner_middle'] = df['close'].ewm(span=20).mean()
        df['keltner_upper'] = df['keltner_middle'] + (df['atr_14'] * 2)
        df['keltner_lower'] = df['keltner_middle'] - (df['atr_14'] * 2)
        
        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
        
        logger.info(f"    ✓ Added {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])} indicators")
        
        return df


class ParquetResampler:
    """Resample 1min Parquet files to higher timeframes"""
    
    TIMEFRAMES = {
        '5min': '5T',
        '15min': '15T',
        '30min': '30T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }
    
    def __init__(self, data_dir: Path = Path("data/parquet")):
        self.data_dir = data_dir
        self.stats = {
            'processed': 0,
            'failed': 0
        }
    
    def resample_all(self) -> None:
        """Find all 1min Parquet files and resample them"""
        logger.info("=" * 80)
        logger.info("RESAMPLING 1-MINUTE DATA TO MULTIPLE TIMEFRAMES")
        logger.info("=" * 80)
        
        # Find all 1min files
        pattern = "**/*_1min.parquet"
        files_1min = list(self.data_dir.glob(pattern))
        
        logger.info(f"Found {len(files_1min)} 1-minute files to process")
        
        for file_path in files_1min:
            self._resample_file(file_path)
        
        self._print_stats()
    
    def _resample_file(self, file_path: Path) -> None:
        """Resample one 1min file to all timeframes"""
        try:
            # Extract symbol and period from filename
            # e.g., EURUSD_202001_1min.parquet → symbol=EURUSD, period=202001
            filename = file_path.stem  # Remove .parquet
            parts = filename.split('_')
            symbol = parts[0]
            period = parts[1]
            
            logger.info(f"\n{symbol} {period}:")
            
            # Read 1min data
            df = pd.read_parquet(file_path)
            
            # Ensure timestamp is datetime and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            logger.info(f"  Loaded {len(df):,} bars from {file_path.name}")
            
            # Resample to each timeframe
            for tf_name, tf_rule in self.TIMEFRAMES.items():
                self._resample_to_timeframe(df, symbol, period, tf_name, tf_rule)
            
            self.stats['processed'] += 1
            
        except Exception as e:
            logger.error(f"  ✗ Failed {file_path.name}: {e}")
            self.stats['failed'] += 1
    
    def _resample_to_timeframe(
        self,
        df_1min: pd.DataFrame,
        symbol: str,
        period: str,
        tf_name: str,
        tf_rule: str
    ) -> None:
        """Resample to one specific timeframe"""
        try:
            # Resample OHLCV
            ohlcv = pd.DataFrame({
                'open': df_1min['open'].resample(tf_rule).first(),
                'high': df_1min['high'].resample(tf_rule).max(),
                'low': df_1min['low'].resample(tf_rule).min(),
                'close': df_1min['close'].resample(tf_rule).last(),
                'volume': df_1min['volume'].resample(tf_rule).sum()
            }).dropna()
            
            if ohlcv.empty:
                return
            
            # Calculate indicators
            logger.info(f"  [{tf_name}] Calculating indicators...")
            ohlcv = TechnicalIndicators.add_all_indicators(ohlcv)
            
            # Save
            output_dir = self.data_dir / symbol / tf_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{symbol}_{period}_{tf_name}.parquet"
            filepath = output_dir / filename
            
            ohlcv.to_parquet(
                filepath,
                engine='fastparquet',
                compression='snappy',
                index=True
            )
            
            logger.info(f"  [{tf_name}] ✓ {len(ohlcv):,} bars → {filename}")
            
        except Exception as e:
            logger.error(f"  [{tf_name}] ✗ Failed: {e}")
    
    def _print_stats(self) -> None:
        """Print statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("RESAMPLING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Files processed: {self.stats['processed']}")
        logger.info(f"Files failed: {self.stats['failed']}")
        logger.info("=" * 80)


def main():
    logger.info("\n⚠️  This script processes existing 1-minute Parquet files")
    logger.info("Make sure you ran download_histdata_1min_only.py first!\n")
    
    response = input("✅ Continue? (y/n): ")
    if response.lower() != 'y':
        logger.info("Cancelled")
        return
    
    resampler = ParquetResampler()
    resampler.resample_all()


if __name__ == '__main__':
    main()
