"""
Download HistData 1-minute bars only (no resampling, no indicators)
Much simpler and faster - resampling will be done later from these files
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from histdata import download_hist_data as dl

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


class HistData1MinDownloader:
    """Simple downloader - 1min data only, no processing"""
    
    def __init__(
        self,
        symbols: list[str],
        start_year: int,
        end_year: int,
        output_dir: Path = Path("data/parquet")
    ):
        self.symbols = symbols
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = output_dir
        
        # Stats
        self.stats = {
            'downloaded': 0,
            'failed': 0,
            'total_bars': 0
        }
    
    def download_all(self) -> None:
        """Download all symbols"""
        logger.info("=" * 80)
        logger.info("HISTDATA 1-MINUTE DOWNLOAD (SIMPLIFIED)")
        logger.info("=" * 80)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Period: {self.start_year} → {self.end_year}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 80)
        
        for symbol in self.symbols:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Processing {symbol}")
            logger.info('=' * 80)
            self._download_symbol(symbol)
        
        self._print_stats()
    
    def _download_symbol(self, symbol: str) -> None:
        """Download one symbol for all years"""
        current_year = datetime.now().year
        
        for year in range(self.start_year, self.end_year + 1):
            if year == current_year:
                # Current year: download month by month
                current_month = datetime.now().month
                for month in range(1, current_month + 1):
                    self._download_and_save(symbol, year, month)
            else:
                # Past years: download full year
                self._download_and_save(symbol, year, None)
    
    def _download_and_save(self, symbol: str, year: int, month: int | None) -> None:
        """Download one file and save as Parquet"""
        try:
            # Download
            if month is None:
                logger.info(f"\n{symbol} {year} (full year):")
                result = dl(
                    pair=symbol.lower(),
                    year=str(year),
                    month=None,
                    platform='ASCII',
                    time_frame='M1'
                )
                period_str = str(year)
            else:
                logger.info(f"\n{symbol} {year}/{month:02d}:")
                result = dl(
                    pair=symbol.lower(),
                    year=str(year),
                    month=month,
                    platform='ASCII',
                    time_frame='M1'
                )
                period_str = f"{year}{month:02d}"
            
            # Read ZIP file
            if not isinstance(result, str):
                logger.warning(f"  ✗ Unexpected return type: {type(result)}")
                self.stats['failed'] += 1
                return
            
            logger.info(f"  Reading: {result}")
            
            import zipfile
            import io
            
            with zipfile.ZipFile(result, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if not csv_files:
                    logger.warning(f"  ✗ No CSV in ZIP")
                    self.stats['failed'] += 1
                    return
                
                # Read CSV content into memory
                csv_content = zip_ref.read(csv_files[0])
            
            # Parse CSV (simple, no processing)
            df = pd.read_csv(
                io.BytesIO(csv_content),
                sep=';',
                names=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
            
            # Save to Parquet (one file per period)
            output_path = self.output_dir / symbol / "1min"
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"{symbol}_{period_str}_1min.parquet"
            filepath = output_path / filename
            
            df.to_parquet(
                filepath,
                engine='fastparquet',
                compression='snappy',
                index=False
            )
            
            bars = len(df)
            self.stats['downloaded'] += 1
            self.stats['total_bars'] += bars
            
            logger.info(f"  ✓ Saved {bars:,} bars → {filename}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            self.stats['failed'] += 1
    
    def _print_stats(self) -> None:
        """Print statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Files downloaded: {self.stats['downloaded']}")
        logger.info(f"Files failed: {self.stats['failed']}")
        logger.info(f"Total bars: {self.stats['total_bars']:,}")
        logger.info("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download HistData 1-minute bars only")
    parser.add_argument('--start-year', type=int, default=2020, help='Start year')
    parser.add_argument('--end-year', type=int, default=2025, help='End year')
    
    args = parser.parse_args()
    
    # Instruments
    LIQUID_FOREX = [
        'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD',
        'USDCHF', 'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY'
    ]
    
    COMMODITIES = ['XAUUSD', 'XAGUSD']
    
    INDICES = [
        'SPXUSD', 'NSXUSD', 'UDXUSD',  # US
        'GRXEUR', 'FRXEUR', 'UKXGBP', 'ETXEUR',  # Europe
        'JPXJPY', 'HKXHKD', 'AUXAUD'  # Asia-Pacific
    ]
    
    all_symbols = LIQUID_FOREX + COMMODITIES + INDICES
    
    # Confirm
    logger.info(f"\n📊 Will download {len(all_symbols)} symbols (1-minute bars only)")
    logger.info(f"📅 Period: {args.start_year} → {args.end_year}")
    logger.info(f"⚠️  No resampling, no indicators (will be done later)")
    
    response = input("\n✅ Continue? (y/n): ")
    if response.lower() != 'y':
        logger.info("Cancelled")
        return
    
    # Download
    downloader = HistData1MinDownloader(
        symbols=all_symbols,
        start_year=args.start_year,
        end_year=args.end_year
    )
    
    downloader.download_all()


if __name__ == '__main__':
    main()
