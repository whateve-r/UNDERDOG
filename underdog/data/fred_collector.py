"""
FRED Macro Indicators Collector

Collects macroeconomic indicators from Federal Reserve Economic Data (FRED) API.
Free API: https://fred.stlouisfed.org/docs/api/api_key.html

Indicators for Forex Trading:
- DFF: Federal Funds Rate (interest rates)
- T10Y2Y: 10-Year vs 2-Year Treasury Yield Curve (recession predictor)
- VIXCLS: VIX Volatility Index (market fear gauge)
- DEXUSEU: EUR/USD Exchange Rate
- UNRATE: Unemployment Rate
- CPIAUCSL: Consumer Price Index (inflation)

Setup:
1. Get free API key: https://fred.stlouisfed.org/docs/api/api_key.html
2. Set environment variable:
   export FRED_API_KEY="your_api_key"
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class MacroIndicator:
    """Macro indicator data point"""
    series_id: str
    name: str
    value: float
    date: datetime
    units: str


class FREDCollector:
    """
    FRED API collector for macroeconomic indicators
    
    Usage:
        collector = FREDCollector()
        dff = await collector.fetch_latest('DFF')  # Fed Funds Rate
        
        # Batch fetch
        indicators = await collector.fetch_all_indicators()
        
        # Start polling loop
        await collector.start_polling(interval_hours=24)
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    # Core indicators for Forex trading
    INDICATORS = {
        'DFF': {
            'name': 'Federal Funds Rate',
            'units': 'Percent',
            'impact': 'High - Direct USD impact'
        },
        'T10Y2Y': {
            'name': 'Treasury Yield Curve (10Y-2Y)',
            'units': 'Percent',
            'impact': 'High - Recession predictor'
        },
        'VIXCLS': {
            'name': 'VIX Volatility Index',
            'units': 'Index',
            'impact': 'High - Risk appetite gauge'
        },
        'DEXUSEU': {
            'name': 'EUR/USD Exchange Rate',
            'units': 'USD per EUR',
            'impact': 'Direct - EUR/USD price'
        },
        'UNRATE': {
            'name': 'Unemployment Rate',
            'units': 'Percent',
            'impact': 'Medium - Economic health'
        },
        'CPIAUCSL': {
            'name': 'Consumer Price Index',
            'units': 'Index 1982-84=100',
            'impact': 'Medium - Inflation gauge'
        },
        'GDP': {
            'name': 'Gross Domestic Product',
            'units': 'Billions of Dollars',
            'impact': 'Medium - Economic growth'
        },
        'MORTGAGE30US': {
            'name': '30-Year Mortgage Rate',
            'units': 'Percent',
            'impact': 'Low - Housing market'
        }
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize FRED API client
        
        Args:
            api_key: FRED API key (or env FRED_API_KEY)
                    Get free key: https://fred.stlouisfed.org/docs/api/api_key.html
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass as argument. Get free key: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        
        logger.info(f"FRED API initialized - Tracking {len(self.INDICATORS)} indicators")
    
    async def fetch_latest(self, series_id: str) -> Optional[MacroIndicator]:
        """
        Fetch latest value for specific indicator
        
        Args:
            series_id: FRED series ID (e.g., 'DFF', 'VIXCLS')
        
        Returns:
            MacroIndicator object or None if error
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'limit': 1,
            'sort_order': 'desc'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"FRED API error: {resp.status} for {series_id}")
                        return None
                    
                    data = await resp.json()
                    
                    if 'observations' not in data or not data['observations']:
                        logger.warning(f"No data for {series_id}")
                        return None
                    
                    latest = data['observations'][0]
                    
                    # Skip if value is '.'
                    if latest['value'] == '.':
                        logger.warning(f"Missing value for {series_id} on {latest['date']}")
                        return None
                    
                    indicator = MacroIndicator(
                        series_id=series_id,
                        name=self.INDICATORS.get(series_id, {}).get('name', series_id),
                        value=float(latest['value']),
                        date=datetime.strptime(latest['date'], '%Y-%m-%d'),
                        units=self.INDICATORS.get(series_id, {}).get('units', 'Unknown')
                    )
                    
                    logger.debug(
                        f"{indicator.name}: {indicator.value} {indicator.units} "
                        f"(as of {indicator.date.strftime('%Y-%m-%d')})"
                    )
                    
                    return indicator
        
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return None
    
    async def fetch_all_indicators(self) -> Dict[str, MacroIndicator]:
        """
        Fetch all tracked indicators in parallel
        
        Returns:
            Dict mapping series_id to MacroIndicator
        """
        tasks = [self.fetch_latest(series_id) for series_id in self.INDICATORS.keys()]
        results = await asyncio.gather(*tasks)
        
        indicators = {}
        for indicator in results:
            if indicator:
                indicators[indicator.series_id] = indicator
        
        logger.info(f"Fetched {len(indicators)}/{len(self.INDICATORS)} indicators successfully")
        return indicators
    
    async def fetch_historical(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime = None
    ) -> List[MacroIndicator]:
        """
        Fetch historical data for indicator
        
        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date (default: today)
        
        Returns:
            List of MacroIndicator objects (chronological)
        """
        end_date = end_date or datetime.utcnow()
        
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date.strftime('%Y-%m-%d'),
            'observation_end': end_date.strftime('%Y-%m-%d'),
            'sort_order': 'asc'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"FRED API error: {resp.status}")
                        return []
                    
                    data = await resp.json()
                    observations = data.get('observations', [])
                    
                    indicators = []
                    for obs in observations:
                        if obs['value'] != '.':
                            indicator = MacroIndicator(
                                series_id=series_id,
                                name=self.INDICATORS.get(series_id, {}).get('name', series_id),
                                value=float(obs['value']),
                                date=datetime.strptime(obs['date'], '%Y-%m-%d'),
                                units=self.INDICATORS.get(series_id, {}).get('units', 'Unknown')
                            )
                            indicators.append(indicator)
                    
                    logger.info(f"Fetched {len(indicators)} historical points for {series_id}")
                    return indicators
        
        except Exception as e:
            logger.error(f"Error fetching historical {series_id}: {e}")
            return []
    
    async def start_polling(
        self,
        interval_hours: int = 24,
        callback = None
    ):
        """
        Start async polling loop (FRED updates once per day)
        
        Args:
            interval_hours: Poll frequency (default: 24h, FRED updates daily)
            callback: Optional callback(indicators_dict) when new data collected
        """
        logger.info(f"Starting FRED polling every {interval_hours}h")
        
        while True:
            try:
                indicators = await self.fetch_all_indicators()
                
                if callback:
                    await callback(indicators)
                
                logger.info(f"FRED poll complete. Next poll in {interval_hours}h.")
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(3600)  # Retry after 1h on error
    
    def get_indicator_info(self, series_id: str) -> Dict:
        """Get metadata for indicator"""
        return self.INDICATORS.get(series_id, {})


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Set environment variable first:
    # export FRED_API_KEY="your_api_key"
    
    async def main():
        collector = FREDCollector()
        
        print(f"\n{'='*60}")
        print(f"FRED Macro Indicators - Latest Values")
        print(f"{'='*60}\n")
        
        # Fetch all current indicators
        indicators = await collector.fetch_all_indicators()
        
        # Print in order of importance
        priority_order = ['DFF', 'VIXCLS', 'T10Y2Y', 'DEXUSEU', 'UNRATE', 'CPIAUCSL']
        
        for series_id in priority_order:
            if series_id in indicators:
                ind = indicators[series_id]
                info = collector.get_indicator_info(series_id)
                
                print(f"{ind.name}")
                print(f"  Value: {ind.value:.2f} {ind.units}")
                print(f"  Date: {ind.date.strftime('%Y-%m-%d')}")
                print(f"  Impact: {info.get('impact', 'Unknown')}")
                print()
        
        print(f"\n{'='*60}")
        print(f"Historical Data Example: VIX (last 30 days)")
        print(f"{'='*60}\n")
        
        # Fetch VIX historical
        start = datetime.utcnow() - timedelta(days=30)
        vix_history = await collector.fetch_historical('VIXCLS', start)
        
        if vix_history:
            print(f"Date                Value")
            print(f"{'-'*30}")
            for ind in vix_history[-10:]:  # Last 10 days
                print(f"{ind.date.strftime('%Y-%m-%d')}      {ind.value:.2f}")
            
            # Calculate volatility
            values = [ind.value for ind in vix_history]
            avg_vix = sum(values) / len(values)
            max_vix = max(values)
            min_vix = min(values)
            
            print(f"\nStatistics (30 days):")
            print(f"  Average: {avg_vix:.2f}")
            print(f"  Max: {max_vix:.2f}")
            print(f"  Min: {min_vix:.2f}")
    
    # Run async example
    asyncio.run(main())
