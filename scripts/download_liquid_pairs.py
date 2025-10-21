"""
Download Most Liquid Forex Pairs + Gold/Silver + Indices

This script downloads the most liquid and relevant assets for professional trading:
- 10 most liquid forex pairs (tight spreads, high volume)
- Commodities: Gold (XAUUSD), Silver (XAGUSD)
- Major indices: US30, US500, NAS100, GER40, UK100

Total: ~15 instruments covering all major markets

Estimated download time: 3-5 hours (2020-2024)
Estimated disk space: ~60GB compressed Parquet

Usage:
    poetry run python scripts/download_liquid_pairs.py --start-year 2020 --end-year 2024
"""
import sys
from pathlib import Path
import logging
import argparse
from datetime import datetime

# Add underdog to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main downloader
from download_all_histdata import HistDataCompleteDownloader

# Setup logging with UTF-8 encoding for Windows
log_file_handler = logging.FileHandler('data/download_liquid_pairs.log', encoding='utf-8')
log_file_handler.setLevel(logging.INFO)
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Stream handler with UTF-8 for console (Windows compatibility)
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


# ============================================================================
# MOST LIQUID INSTRUMENTS FOR TRADING (All available on HistData.com!)
# ============================================================================

LIQUID_FOREX_PAIRS = [
    # Major Pairs (Highest Volume, Tightest Spreads)
    'EURUSD',    # #1 Most liquid pair (28% of forex volume)
    'USDJPY',    # #2 Most liquid (14% of volume)
    'GBPUSD',    # #3 Cable (11% of volume)
    'AUDUSD',    # #4 Aussie (6% of volume)
    'USDCAD',    # #5 Loonie (5% of volume)
    'USDCHF',    # #6 Swissie (4% of volume)
    
    # Major Crosses (High Volume, Good for Diversification)
    'EURJPY',    # Euro-Yen cross (very liquid)
    'GBPJPY',    # Pound-Yen cross (volatile, good volume)
    'EURGBP',    # Euro-Pound cross (tight spread)
    'AUDJPY',    # Aussie-Yen cross (carry trade favorite)
]

# ============================================================================
# COMMODITIES (Available on HistData.com as XXX/USD format)
# ============================================================================

COMMODITIES = [
    # Precious Metals
    'XAUUSD',    # Gold vs USD (safe haven, high liquidity)
    'XAGUSD',    # Silver vs USD (industrial metal)
]

# ============================================================================
# STOCK INDICES (Available on HistData.com as XXX/YYY format)
# ============================================================================

INDICES = [
    # US Indices
    'SPXUSD',    # S&P 500 in USD
    'NSXUSD',    # NASDAQ 100 in USD
    'UDXUSD',    # US Dollar Index in USD
    
    # European Indices
    'GRXEUR',    # DAX 30 (Germany) in EUR
    'FRXEUR',    # CAC 40 (France) in EUR
    'UKXGBP',    # FTSE 100 (UK) in GBP
    'ETXEUR',    # EUROSTOXX 50 in EUR
    
    # Asia-Pacific Indices
    'JPXJPY',    # Nikkei 225 (Japan) in JPY
    'HKXHKD',    # Hang Seng (Hong Kong) in HKD
    'AUXAUD',    # ASX 200 (Australia) in AUD
]

INSTRUMENTS_INFO = """
✅ ALL INSTRUMENTS AVAILABLE ON HISTDATA.COM

Total instruments: 27
- Forex Pairs: 10 (most liquid)
- Commodities: 8 (precious metals + oil)
- Stock Indices: 9 (US, Europe, Asia-Pacific)

Format in HistData.com:
- Forex: EURUSD, GBPUSD, etc.
- Commodities: XAUUSD (Gold), WTIUSD (Oil), etc.
- Indices: SPXUSD (S&P 500), GRXEUR (DAX), etc.

All data available as:
- Tick data (highest resolution)
- M1 data (1-minute bars)
- Organized by pair/year/month
"""


def main():
    """Download most liquid instruments"""
    parser = argparse.ArgumentParser(
        description='Download most liquid forex pairs + commodities'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=2020,
        help='Start year (default: 2020)'
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
        help='Skip indicator calculation (faster, but less useful)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/parquet',
        help='Output directory (default: data/parquet)'
    )
    
    args = parser.parse_args()
    
    # Combine all instruments
    all_instruments = LIQUID_FOREX_PAIRS + COMMODITIES + INDICES
    
    # Display summary
    logger.info("=" * 80)
    logger.info("LIQUID INSTRUMENTS DOWNLOAD - COMPLETE PACKAGE")
    logger.info("=" * 80)
    logger.info("\n📊 INSTRUMENTS TO DOWNLOAD (27 total):")
    logger.info("\n  🔵 Forex Pairs (10):")
    for i, pair in enumerate(LIQUID_FOREX_PAIRS, 1):
        logger.info(f"    {i:2d}. {pair}")
    
    logger.info("\n  🟡 Commodities (8):")
    logger.info("    Precious Metals:")
    for commodity in COMMODITIES[:6]:
        logger.info(f"      • {commodity}")
    
    
    logger.info("\n  📈 Stock Indices (9):")
    logger.info("    US Markets:")
    for index in INDICES[:3]:
        logger.info(f"      • {index}")
    logger.info("    European Markets:")
    for index in INDICES[3:7]:
        logger.info(f"      • {index}")
    logger.info("    Asia-Pacific:")
    for index in INDICES[7:]:
        logger.info(f"      • {index}")
    
    logger.info(f"\n📅 Date Range: {args.start_year} → {args.end_year}")
    logger.info(f"⏱️  Estimated Time: {3 + (args.end_year - args.start_year) * 0.8:.1f} hours")
    logger.info(f"💾 Estimated Space: {15 + (args.end_year - args.start_year) * 18:.0f} GB")
    
    logger.info("\n" + "=" * 80)
    
    # Confirm
    response = input("\n✅ Continue with download? (y/n): ")
    if response.lower() != 'y':
        logger.info("Download cancelled")
        return
    
    # Create downloader
    downloader = HistDataCompleteDownloader(
        symbols=all_instruments,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
        calculate_indicators=not args.no_indicators
    )
    
    # Run download
    logger.info("\n🚀 Starting download...")
    logger.info("Progress will be saved incrementally (safe to interrupt)\n")
    
    stats = downloader.download_all()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ DOWNLOAD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nMonths downloaded: {stats['downloaded_months']}")
    logger.info(f"Total bars (1min): {stats['total_ticks']:,}")
    
    if stats['total_bars']:
        logger.info("\nBars per timeframe:")
        for tf, count in sorted(stats['total_bars'].items()):
            logger.info(f"  {tf:6s}: {count:,}")
    
    if stats['errors']:
        logger.warning(f"\n⚠️  Errors: {len(stats['errors'])}")
        logger.warning("Check data/download_liquid_pairs.log for details")
    
    # Next steps
    logger.info("\n" + "=" * 80)
    logger.info("📋 NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("\n1️⃣  Insert into TimescaleDB:")
    logger.info("    poetry run python scripts/insert_indicators_to_db.py --all")
    
    logger.info("\n2️⃣  Verify data availability:")
    logger.info("    poetry run python -c \"from underdog.database.db_loader import get_loader; print(get_loader().get_available_symbols())\"")
    
    logger.info("\n3️⃣  Run backtest with any instrument:")
    logger.info("    poetry run python underdog/backtesting/simple_runner.py")
    
    logger.info("\n" + "=" * 80)
    
    # Print instruments info
    print("\n" + INSTRUMENTS_INFO)
    
    return stats


if __name__ == '__main__':
    main()
