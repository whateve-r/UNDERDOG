"""
Download Stock Indices from Yahoo Finance

HistData.com only provides forex pairs. This script downloads major indices
from Yahoo Finance (free, no API key required).

Indices downloaded:
- US30 (Dow Jones Industrial Average) - ^DJI
- US500 (S&P 500) - ^GSPC
- NAS100 (NASDAQ 100) - ^NDX
- GER40 (DAX 40) - ^GDAXI
- UK100 (FTSE 100) - ^FTSE
- JPN225 (Nikkei 225) - ^N225

Features:
- Downloads daily OHLCV data
- Resamples to multiple timeframes (1d, 1h, 4h - limited by Yahoo data)
- Calculates same technical indicators as forex data
- Saves to Parquet format (compatible with main system)
- Inserts into TimescaleDB

Requirements:
    poetry add yfinance

Usage:
    poetry run python scripts/download_indices.py --start-date 2020-01-01 --end-date 2024-12-31
"""
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta
import pandas as pd
import argparse
import logging

# Add underdog to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Major stock indices (Yahoo Finance tickers)
INDICES = {
    'US30': '^DJI',       # Dow Jones Industrial Average
    'US500': '^GSPC',     # S&P 500
    'NAS100': '^NDX',     # NASDAQ 100
    'GER40': '^GDAXI',    # DAX 40 (Germany)
    'UK100': '^FTSE',     # FTSE 100 (UK)
    'JPN225': '^N225',    # Nikkei 225 (Japan)
    'FRA40': '^FCHI',     # CAC 40 (France)
    'AUS200': '^AXJO',    # ASX 200 (Australia)
}


class IndicesDownloader:
    """
    Download stock indices from Yahoo Finance.
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = 'data/parquet',
        calculate_indicators: bool = True
    ):
        """
        Initialize downloader.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Output directory for Parquet files
            calculate_indicators: Calculate technical indicators
        """
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.calculate_indicators = calculate_indicators
        
        # Create output directories
        for symbol in INDICES.keys():
            for tf in ['1d', '4h', '1h']:
                (self.output_dir / symbol / tf).mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'downloaded': 0,
            'saved': 0,
            'errors': []
        }
        
        logger.info("=" * 80)
        logger.info("STOCK INDICES DOWNLOADER (Yahoo Finance)")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} → {end_date}")
        logger.info(f"Indices: {', '.join(INDICES.keys())}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 80)
    
    def download_all(self) -> Dict:
        """Download all indices"""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("❌ yfinance not installed!")
            logger.error("Install with: poetry add yfinance")
            logger.error("Or: pip install yfinance")
            return self.stats
        
        for symbol, ticker in INDICES.items():
            self._download_index(symbol, ticker, yf)
        
        logger.info("\n" + "=" * 80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        self._print_statistics()
        
        return self.stats
    
    def _download_index(self, symbol: str, ticker: str, yf) -> None:
        """Download one index"""
        logger.info(f"\n{symbol} ({ticker}):")
        
        try:
            # Download from Yahoo Finance
            logger.info(f"  Downloading from Yahoo Finance...")
            
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                interval='1h',  # Hourly data (Yahoo provides intraday for recent periods)
                progress=False
            )
            
            if data.empty:
                logger.warning(f"  ✗ No data available")
                return
            
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add volume if missing
            if 'volume' not in data.columns:
                data['volume'] = 0
            
            logger.info(f"  ✓ Downloaded {len(data):,} hourly bars")
            self.stats['downloaded'] += len(data)
            
            # Calculate indicators if enabled
            if self.calculate_indicators:
                from download_all_histdata import TechnicalIndicators
                data = TechnicalIndicators.add_all_indicators(data)
            
            # Save 1h data
            self._save_parquet(data, symbol, '1h')
            
            # Resample to 4h and 1d
            data_4h = self._resample_data(data, '4h')
            if not data_4h.empty:
                if self.calculate_indicators:
                    data_4h = TechnicalIndicators.add_all_indicators(data_4h)
                self._save_parquet(data_4h, symbol, '4h')
            
            data_1d = self._resample_data(data, '1d')
            if not data_1d.empty:
                if self.calculate_indicators:
                    data_1d = TechnicalIndicators.add_all_indicators(data_1d)
                self._save_parquet(data_1d, symbol, '1d')
            
            logger.info(f"  ✓ Completed {symbol}")
            
        except Exception as e:
            error_msg = f"Failed {symbol}: {str(e)}"
            logger.error(f"  ✗ {error_msg}")
            self.stats['errors'].append(error_msg)
    
    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample to higher timeframe"""
        rule_map = {
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        rule = rule_map.get(timeframe, '1D')
        
        resampled = pd.DataFrame({
            'open': data['open'].resample(rule).first(),
            'high': data['high'].resample(rule).max(),
            'low': data['low'].resample(rule).min(),
            'close': data['close'].resample(rule).last(),
            'volume': data['volume'].resample(rule).sum()
        })
        
        return resampled.dropna()
    
    def _save_parquet(self, data: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Save to Parquet"""
        filename = f"{symbol}_yfinance_{timeframe}.parquet"
        filepath = self.output_dir / symbol / timeframe / filename
        
        data.to_parquet(
            filepath,
            engine='fastparquet',
            compression='snappy',
            index=True
        )
        
        logger.info(f"    ✓ {timeframe}: {len(data):,} bars → {filename}")
        self.stats['saved'] += 1
    
    def _print_statistics(self) -> None:
        """Print statistics"""
        logger.info("\n📊 Download Summary:")
        logger.info("-" * 80)
        logger.info(f"Total bars downloaded: {self.stats['downloaded']:,}")
        logger.info(f"Files saved: {self.stats['saved']}")
        
        if self.stats['errors']:
            logger.warning(f"\n⚠️  Errors ({len(self.stats['errors'])}):")
            for error in self.stats['errors']:
                logger.warning(f"  - {error}")
        else:
            logger.info("\n✓ No errors")
        
        logger.info("-" * 80)


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description='Download stock indices from Yahoo Finance'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date (YYYY-MM-DD, default: 2020-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help=f'End date (YYYY-MM-DD, default: today)'
    )
    
    parser.add_argument(
        '--no-indicators',
        action='store_true',
        help='Skip indicator calculation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/parquet',
        help='Output directory (default: data/parquet)'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = IndicesDownloader(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        calculate_indicators=not args.no_indicators
    )
    
    # Run download
    stats = downloader.download_all()
    
    # Next steps
    logger.info("\n" + "=" * 80)
    logger.info("📋 NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("\n1️⃣  Insert into TimescaleDB:")
    logger.info("    poetry run python scripts/insert_indicators_to_db.py --all")
    logger.info("\n2️⃣  Verify data:")
    logger.info("    poetry run python -c \"from underdog.database.db_loader import get_loader; print(get_loader().get_available_symbols())\"")
    logger.info("\n" + "=" * 80)
    
    return stats


if __name__ == '__main__':
    main()
