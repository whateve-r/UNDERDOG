"""
FRED Macro Indicators Historical Backfill

Downloads historical macroeconomic data from FRED API and populates TimescaleDB.

Usage:
    poetry run python scripts/backfill_fred_macro.py --start 2022-01-01 --end 2024-10-31 --verify

Features:
- Downloads 3 years of historical data for 8 macro indicators
- Inserts into macro_indicators table in TimescaleDB
- Deduplication (ON CONFLICT DO NOTHING)
- Batch insertion (1000 rows/batch)
- Progress tracking with statistics

Indicators:
- DFF: Federal Funds Rate (daily)
- VIXCLS: VIX Volatility Index (daily)
- T10Y2Y: Treasury Yield Curve 10Y-2Y (daily)
- DGS10: 10-Year Treasury Yield (daily)
- DGS2: 2-Year Treasury Yield (daily)
- UNRATE: Unemployment Rate (monthly)
- CPIAUCSL: Consumer Price Index (monthly)
- GDP: Gross Domestic Product (quarterly)
"""

import os
import sys
import logging
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yaml

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from underdog.data.fred_collector import FREDCollector, MacroIndicator
from underdog.database.timescale.timescale_connector import TimescaleDBConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FREDBackfiller:
    """
    Historical backfill for FRED macro indicators
    """
    
    # Series to download (aligned with config/data_providers.yaml)
    SERIES_CONFIG = {
        'DFF': {
            'name': 'Federal Funds Rate',
            'frequency': 'daily',
            'priority': 'high'
        },
        'VIXCLS': {
            'name': 'VIX Volatility Index',
            'frequency': 'daily',
            'priority': 'high'
        },
        'T10Y2Y': {
            'name': 'Treasury Yield Curve (10Y-2Y)',
            'frequency': 'daily',
            'priority': 'high'
        },
        'DGS10': {
            'name': '10-Year Treasury Yield',
            'frequency': 'daily',
            'priority': 'medium'
        },
        'DGS2': {
            'name': '2-Year Treasury Yield',
            'frequency': 'daily',
            'priority': 'medium'
        },
        'UNRATE': {
            'name': 'Unemployment Rate',
            'frequency': 'monthly',
            'priority': 'medium'
        },
        'CPIAUCSL': {
            'name': 'Consumer Price Index',
            'frequency': 'monthly',
            'priority': 'low'
        },
        'GDP': {
            'name': 'Gross Domestic Product',
            'frequency': 'quarterly',
            'priority': 'low'
        }
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize FRED backfiller
        
        Args:
            api_key: FRED API key (or from config)
        """
        # Load config
        config_path = PROJECT_ROOT / 'config' / 'data_providers.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get API key from config or parameter
        self.api_key = api_key or config.get('fred', {}).get('api_key')
        
        if not self.api_key:
            raise ValueError("FRED API key not found in config/data_providers.yaml")
        
        # Initialize collectors
        self.fred = FREDCollector(api_key=self.api_key)
        
        # Initialize database with credentials from config
        db_config = config.get('timescaledb', {})
        self.db = TimescaleDBConnector(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'underdog_trading'),
            user=db_config.get('user', 'underdog'),
            password=db_config.get('password', 'underdog_trading_2024_secure')
        )
        
        logger.info(f"FRED Backfiller initialized - {len(self.SERIES_CONFIG)} series")
    
    async def download_series(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[MacroIndicator]:
        """
        Download historical data for specific series
        
        Args:
            series_id: FRED series ID (e.g., 'DFF', 'VIXCLS')
            start_date: Start date
            end_date: End date
        
        Returns:
            List of MacroIndicator objects
        """
        series_info = self.SERIES_CONFIG.get(series_id, {})
        series_name = series_info.get('name', series_id)
        
        logger.info(f"Downloading {series_name} ({series_id})...")
        
        indicators = await self.fred.fetch_historical(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if indicators:
            first = indicators[0].date.strftime('%Y-%m-%d')
            last = indicators[-1].date.strftime('%Y-%m-%d')
            logger.info(
                f"✓ {series_name}: {len(indicators)} observations "
                f"({first} to {last})"
            )
        else:
            logger.warning(f"✗ {series_name}: No data retrieved")
        
        return indicators
    
    async def insert_to_db(self, indicators: List[MacroIndicator]) -> int:
        """
        Insert indicators into TimescaleDB macro_indicators table
        
        Args:
            indicators: List of MacroIndicator objects
        
        Returns:
            Number of rows inserted
        """
        if not indicators:
            return 0
        
        # Build INSERT query with ON CONFLICT DO NOTHING (deduplication)
        insert_query = """
            INSERT INTO macro_indicators (
                time, series_id, name, value, units
            ) VALUES (
                %s, %s, %s, %s, %s
            )
            ON CONFLICT (time, series_id) DO NOTHING
        """
        
        # Prepare batch data
        batch_data = [
            (
                ind.date,
                ind.series_id,
                ind.name,
                ind.value,
                ind.units
            )
            for ind in indicators
        ]
        
        # Batch insert (1000 rows at a time)
        batch_size = 1000
        total_inserted = 0
        
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i+batch_size]
            
            try:
                with self.db.get_connection() as conn:
                    with conn.cursor() as cur:
                        # Execute batch
                        cur.executemany(insert_query, batch)
                        inserted = cur.rowcount
                        conn.commit()
                        total_inserted += inserted
                        
                        logger.debug(f"Batch {i//batch_size + 1}: Inserted {inserted} rows")
            
            except Exception as e:
                logger.error(f"Error inserting batch: {e}")
                continue
        
        return total_inserted
    
    async def backfill_all(
        self,
        start_date: datetime,
        end_date: datetime,
        verify: bool = False
    ):
        """
        Backfill all FRED series
        
        Args:
            start_date: Start date
            end_date: End date
            verify: Print verification stats after insertion
        """
        logger.info("="*80)
        logger.info("FRED MACRO INDICATORS HISTORICAL BACKFILL")
        logger.info("="*80)
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Series: {len(self.SERIES_CONFIG)}")
        logger.info("="*80)
        logger.info("")
        
        # Connect to database
        self.db.connect()
        
        # Track statistics
        total_downloaded = 0
        total_inserted = 0
        
        # Download and insert each series
        for series_id, series_info in self.SERIES_CONFIG.items():
            try:
                # Download historical data
                indicators = await self.download_series(
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not indicators:
                    logger.warning(f"Skipping {series_id} - no data")
                    continue
                
                total_downloaded += len(indicators)
                
                # Insert into database
                inserted = await self.insert_to_db(indicators)
                total_inserted += inserted
                
                logger.info(f"  Inserted {inserted}/{len(indicators)} rows for {series_id}")
                logger.info("")
            
            except Exception as e:
                logger.error(f"Error processing {series_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary
        logger.info("="*80)
        logger.info("BACKFILL COMPLETE")
        logger.info("="*80)
        logger.info(f"Total downloaded: {total_downloaded} observations")
        logger.info(f"Total inserted: {total_inserted} rows")
        logger.info(f"Duplicates skipped: {total_downloaded - total_inserted}")
        logger.info("="*80)
        
        # Verification query
        if verify:
            logger.info("")
            logger.info("VERIFICATION:")
            logger.info("-"*80)
            
            verify_query = """
                SELECT 
                    series_id,
                    name,
                    COUNT(*) as observations,
                    MIN(time) as first_date,
                    MAX(time) as last_date,
                    ROUND(AVG(value)::numeric, 2) as avg_value
                FROM macro_indicators
                GROUP BY series_id, name
                ORDER BY series_id
            """
            
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(verify_query)
                    results = cur.fetchall()
                    
                    logger.info(f"{'Series':<12} {'Name':<35} {'Obs':<8} {'First Date':<12} {'Last Date':<12} {'Avg Value':<10}")
                    logger.info("-"*80)
                    
                    for row in results:
                        series_id, name, count, first, last, avg = row
                        logger.info(
                            f"{series_id:<12} {name:<35} {count:<8} "
                            f"{first.strftime('%Y-%m-%d'):<12} {last.strftime('%Y-%m-%d'):<12} "
                            f"{avg:<10}"
                        )
            
            logger.info("="*80)
        
        # Disconnect
        self.db.disconnect()


async def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='FRED Macro Indicators Historical Backfill'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD), e.g., 2022-01-01'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD), default: today'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Print verification statistics after backfill'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='FRED API key (or use config/data_providers.yaml)'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = datetime.utcnow()
    
    # Initialize backfiller
    try:
        backfiller = FREDBackfiller(api_key=args.api_key)
        
        # Execute backfill
        await backfiller.backfill_all(
            start_date=start_date,
            end_date=end_date,
            verify=args.verify
        )
    
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
