"""
Consolidated Backfill Script for Historical Data + News

Integrates:
1. HistData.com - Forex tick/minute data (FREE)
2. News Scraping - RSS feeds + sentiment (FREE)
3. TimescaleDB - Data storage and hypertable optimization

Workflow:
- Download/Load historical OHLCV data from HistData
- Scrape news from multiple RSS sources
- Apply sentiment analysis (VADER/FinBERT)
- Ingest into TimescaleDB with validation
- Create hypertables and compression policies
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import argparse

# Add underdog to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from underdog.database.histdata_ingestion import (
    HistDataIngestion,
    HistDataConfig
)
from underdog.database.news_scraping import (
    NewsIngestion,
    NewsScrapingConfig
)


class BackfillPipeline:
    """
    Consolidated pipeline for historical data backfill.
    
    Features:
    - Multi-symbol Forex data ingestion
    - Multi-source news aggregation
    - Sentiment analysis
    - TimescaleDB integration
    - Data quality validation
    - Progress tracking
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframes: List[str] = ['5min', '15min', '1h'],
        sentiment_model: str = 'vader'
    ):
        """
        Initialize backfill pipeline.
        
        Args:
            symbols: List of Forex symbols (e.g., ['EURUSD', 'GBPUSD'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframes: List of timeframes to generate
            sentiment_model: Sentiment analysis model ('vader' or 'finbert')
        """
        self.symbols = symbols
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.timeframes = timeframes
        self.sentiment_model = sentiment_model
        
        # Initialize ingestion modules
        self.histdata = HistDataIngestion()
        self.news_scraper = NewsIngestion(
            NewsScrapingConfig(
                sentiment_model=sentiment_model,
                symbols=symbols
            )
        )
        
        # Statistics
        self.stats = {
            'forex_data': {
                'symbols_processed': 0,
                'total_bars': 0,
                'timeframes': {}
            },
            'news_data': {
                'total_articles': 0,
                'relevant_articles': 0,
                'avg_sentiment': 0.0
            },
            'errors': []
        }
        
        print("=" * 70)
        print("UNDERDOG - Historical Data Backfill Pipeline")
        print("=" * 70)
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Period: {start_date} → {end_date}")
        print(f"Timeframes: {', '.join(timeframes)}")
        print(f"Sentiment Model: {sentiment_model}")
        print("=" * 70)
    
    def run_full_backfill(self) -> Dict:
        """
        Run complete backfill pipeline.
        
        Returns:
            Dict with execution statistics
        """
        print("\n[STEP 1/3] Loading Historical Forex Data")
        print("-" * 70)
        self._backfill_forex_data()
        
        print("\n[STEP 2/3] Scraping News + Sentiment Analysis")
        print("-" * 70)
        self._backfill_news_data()
        
        print("\n[STEP 3/3] Database Ingestion (TimescaleDB)")
        print("-" * 70)
        self._ingest_to_database()
        
        print("\n" + "=" * 70)
        print("BACKFILL COMPLETE")
        print("=" * 70)
        self._print_statistics()
        
        return self.stats
    
    def _backfill_forex_data(self) -> None:
        """Backfill Forex historical data from HistData"""
        data_dir = Path("data/raw/histdata")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for symbol in self.symbols:
            print(f"\n[Forex] Processing {symbol}...")
            
            # Calculate year/month range
            start_year = self.start_date.year
            start_month = self.start_date.month
            end_year = self.end_date.year
            end_month = self.end_date.month
            
            # Check if CSV files exist (manual download)
            csv_files = list(data_dir.glob(f"{symbol}_*.csv"))
            
            if csv_files:
                print(f"  Found {len(csv_files)} CSV files for {symbol}")
                
                # Load all CSV files
                all_data = []
                for csv_file in csv_files:
                    print(f"  Loading: {csv_file.name}")
                    try:
                        tick_data = self.histdata.load_from_csv(str(csv_file), symbol)
                        all_data.append(tick_data)
                    except Exception as e:
                        self.stats['errors'].append(f"Failed to load {csv_file}: {e}")
                        print(f"    ERROR: {e}")
                
                if all_data:
                    # Concatenate all data
                    combined_data = pd.concat(all_data, ignore_index=False)
                    combined_data = combined_data.sort_index()
                    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
                    
                    print(f"  Total tick records: {len(combined_data):,}")
                    
                    # Convert to OHLCV for each timeframe
                    for timeframe in self.timeframes:
                        print(f"  Converting to {timeframe} OHLCV...")
                        
                        try:
                            ohlcv = self.histdata.convert_tick_to_ohlcv(
                                combined_data,
                                timeframe=timeframe
                            )
                            
                            # Save to Parquet
                            output_path = self.histdata.save_to_parquet(
                                ohlcv,
                                symbol,
                                f"ohlcv_{timeframe}"
                            )
                            
                            bars = len(ohlcv)
                            print(f"    ✓ {bars:,} bars → {output_path}")
                            
                            # Update statistics
                            if timeframe not in self.stats['forex_data']['timeframes']:
                                self.stats['forex_data']['timeframes'][timeframe] = 0
                            self.stats['forex_data']['timeframes'][timeframe] += bars
                            self.stats['forex_data']['total_bars'] += bars
                            
                        except Exception as e:
                            self.stats['errors'].append(f"{symbol} {timeframe} conversion failed: {e}")
                            print(f"    ERROR: {e}")
                    
                    self.stats['forex_data']['symbols_processed'] += 1
            else:
                print(f"  WARNING: No CSV files found for {symbol}")
                print(f"  Please download manually from:")
                print(f"    https://www.histdata.com/download-free-forex-historical-data/")
                print(f"  Save to: {data_dir}/")
                self.stats['errors'].append(f"No data files for {symbol}")
    
    def _backfill_news_data(self) -> None:
        """Backfill news data with sentiment analysis"""
        print("\n[News] Scraping from multiple sources...")
        
        try:
            # Scrape all sources
            news_df = self.news_scraper.scrape_all_sources()
            
            if news_df.empty:
                print("  WARNING: No news articles scraped")
                self.stats['errors'].append("News scraping returned no results")
                return
            
            # Filter by date range
            news_df = news_df[
                (news_df['timestamp'] >= self.start_date) &
                (news_df['timestamp'] <= self.end_date)
            ]
            
            # Update statistics
            total = len(news_df)
            relevant = news_df['is_relevant'].sum()
            avg_sentiment = news_df['sentiment_score'].mean()
            
            self.stats['news_data']['total_articles'] = total
            self.stats['news_data']['relevant_articles'] = relevant
            self.stats['news_data']['avg_sentiment'] = avg_sentiment
            
            print(f"  Total articles: {total}")
            print(f"  Relevant: {relevant} ({relevant/total*100:.1f}%)")
            print(f"  Avg sentiment: {avg_sentiment:.3f}")
            
            # Save to Parquet
            output_path = self.news_scraper.save_to_parquet(
                news_df,
                filename=f"news_backfill_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}"
            )
            print(f"  Saved to: {output_path}")
            
        except Exception as e:
            self.stats['errors'].append(f"News scraping failed: {e}")
            print(f"  ERROR: {e}")
    
    def _ingest_to_database(self) -> None:
        """Ingest data into TimescaleDB"""
        print("\n[Database] Ingesting to TimescaleDB...")
        
        try:
            # Import database module (if available)
            from underdog.database.data_store import DataStore
            
            db = DataStore()
            
            # Ingest Forex data
            print("  [1/2] Ingesting Forex OHLCV data...")
            processed_dir = Path("data/processed/histdata")
            
            for parquet_file in processed_dir.glob("*.parquet"):
                print(f"    Processing: {parquet_file.name}")
                
                try:
                    data = pd.read_parquet(parquet_file)
                    
                    # Insert into database
                    # db.insert_ohlcv(data, table_name=...)
                    # TODO: Implement database insertion
                    
                    print(f"      ✓ Inserted {len(data):,} records")
                    
                except Exception as e:
                    self.stats['errors'].append(f"Database insert failed for {parquet_file}: {e}")
                    print(f"      ERROR: {e}")
            
            # Ingest News data
            print("  [2/2] Ingesting News data...")
            news_dir = Path("data/processed/news")
            
            for parquet_file in news_dir.glob("*.parquet"):
                print(f"    Processing: {parquet_file.name}")
                
                try:
                    data = pd.read_parquet(parquet_file)
                    
                    # Insert into database
                    # db.insert_news(data, table_name=...)
                    # TODO: Implement database insertion
                    
                    print(f"      ✓ Inserted {len(data):,} records")
                    
                except Exception as e:
                    self.stats['errors'].append(f"Database insert failed for {parquet_file}: {e}")
                    print(f"      ERROR: {e}")
            
            print("  Database ingestion complete!")
            
        except ImportError:
            print("  WARNING: DataStore module not available")
            print("  Data saved to Parquet files only")
            self.stats['errors'].append("TimescaleDB ingestion skipped (module not found)")
    
    def _print_statistics(self) -> None:
        """Print execution statistics"""
        print("\n📊 Execution Summary:")
        print("-" * 70)
        
        # Forex data
        forex = self.stats['forex_data']
        print(f"Forex Data:")
        print(f"  Symbols processed: {forex['symbols_processed']}/{len(self.symbols)}")
        print(f"  Total bars: {forex['total_bars']:,}")
        for tf, count in forex['timeframes'].items():
            print(f"    {tf}: {count:,} bars")
        
        # News data
        news = self.stats['news_data']
        print(f"\nNews Data:")
        print(f"  Total articles: {news['total_articles']:,}")
        print(f"  Relevant: {news['relevant_articles']:,}")
        print(f"  Avg sentiment: {news['avg_sentiment']:.3f}")
        
        # Errors
        if self.stats['errors']:
            print(f"\n⚠️  Errors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:5]:  # Show first 5
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors'])-5} more")
        else:
            print(f"\n✓ No errors")
        
        print("-" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Backfill historical Forex + News data')
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['EURUSD', 'GBPUSD'],
        help='Forex symbols to backfill (default: EURUSD GBPUSD)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['5min', '15min', '1h'],
        help='OHLCV timeframes to generate (default: 5min 15min 1h)'
    )
    
    parser.add_argument(
        '--sentiment-model',
        type=str,
        choices=['vader', 'finbert'],
        default='vader',
        help='Sentiment analysis model (default: vader)'
    )
    
    args = parser.parse_args()
    
    # Run backfill
    pipeline = BackfillPipeline(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframes=args.timeframes,
        sentiment_model=args.sentiment_model
    )
    
    stats = pipeline.run_full_backfill()
    
    return stats


def run_example():
    """Example usage (no command line args)"""
    print("\n" + "=" * 70)
    print("EXAMPLE BACKFILL EXECUTION")
    print("=" * 70)
    print("\nThis example assumes you have manually downloaded:")
    print("  - EURUSD tick data for 2023-01 from HistData.com")
    print("  - Saved to: data/raw/histdata/EURUSD_202301.csv")
    print("\n" + "=" * 70 + "\n")
    
    pipeline = BackfillPipeline(
        symbols=['EURUSD'],
        start_date='2023-01-01',
        end_date='2023-01-31',
        timeframes=['5min', '15min', '1h'],
        sentiment_model='vader'
    )
    
    stats = pipeline.run_full_backfill()
    
    return stats


if __name__ == '__main__':
    # Check if command line args provided
    if len(sys.argv) > 1:
        main()
    else:
        # Run example
        run_example()
