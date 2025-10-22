"""
Data Ingestion Orchestrator - Feature Store Coordinator

Coordina 3 categorías de datos según frecuencia y latencia:

1. 📈 Microestructura (Alta Frecuencia - Streaming):
   - MT5 Tick/OHLCV (ZeroMQ) → TimescaleDB
   - Bid/Ask Spread → TimescaleDB
   - Latencia: <100ms

2. 📰 Sentiment (Media Frecuencia - Pooling 15min):
   - News API → FinGPT (inference local) → TimescaleDB
   - Reddit API (PRAW) → FinGPT → TimescaleDB
   - Output: Sentiment score [-1, 1]

3. 🌍 Macro (Baja Frecuencia - EOD):
   - FRED API → TimescaleDB (daily)
   - Alpha Vantage Fundamentals → TimescaleDB (daily)

Papers:
- AlphaQuanter.pdf: Feature Store + LLM Orchestrator pattern
- arXiv:2510.10526v1: LLM sentiment integration (TD3)

Architecture:
    MT5 (Streaming) ──────┐
    News API (15min) ─────┤
    Reddit (15min) ───────┼──→ DataOrchestrator ──→ TimescaleDB ──→ DRL Agent
    FRED (EOD) ───────────┤                              ↓
    Alpha Vantage (EOD) ──┘                         Redis Cache
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from underdog.database.timescale.timescale_connector import (
    TimescaleDBConnector,
    OHLCVBar,
    SentimentScore,
    MacroIndicator
)
from underdog.sentiment.llm_connector import FinGPTConnector
from underdog.data.reddit_collector import RedditSentimentCollector
from underdog.data.fred_collector import FREDCollector

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    # Microestructura (Streaming)
    mt5_enabled: bool = True
    mt5_symbols: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY'])
    
    # Sentiment (Pooling 15min)
    news_enabled: bool = True
    news_api_key: Optional[str] = None
    reddit_enabled: bool = True
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    sentiment_interval_minutes: int = 15
    
    # Macro (EOD)
    fred_enabled: bool = True
    fred_api_key: Optional[str] = None
    alphavantage_enabled: bool = True
    alphavantage_api_key: Optional[str] = None
    macro_interval_hours: int = 24


class DataIngestionOrchestrator:
    """
    Central coordinator for all data ingestion pipelines
    
    Responsibilities:
    1. Schedule data collection based on update frequency
    2. Route data to appropriate processors (FinGPT for sentiment)
    3. Write processed features to TimescaleDB
    4. Handle errors and retries
    5. Monitor ingestion health
    
    Usage:
        orchestrator = DataIngestionOrchestrator(config)
        await orchestrator.start()
        # Runs indefinitely, polling/streaming data
    """
    
    def __init__(
        self,
        config: DataSourceConfig,
        db_connector: Optional[TimescaleDBConnector] = None
    ):
        """
        Initialize orchestrator
        
        Args:
            config: Data source configuration
            db_connector: TimescaleDB connector (creates new if None)
        """
        self.config = config
        self.db = db_connector or TimescaleDBConnector()
        
        # Components (lazy-loaded)
        self.sentiment_analyzer: Optional[FinGPTConnector] = None
        self.reddit_collector: Optional[RedditSentimentCollector] = None
        self.fred_collector: Optional[FREDCollector] = None
        
        # State tracking
        self.is_running = False
        self._tasks: List[asyncio.Task] = []
        
        # Metrics
        self.stats = {
            'ohlcv_bars_ingested': 0,
            'sentiment_scores_ingested': 0,
            'macro_indicators_ingested': 0,
            'errors': 0,
            'last_update': {
                'ohlcv': None,
                'sentiment': None,
                'macro': None,
            }
        }
        
        logger.info("DataIngestionOrchestrator initialized")
    
    async def start(self):
        """
        Start all data ingestion pipelines
        
        Launches async tasks for:
        - MT5 streaming (if enabled)
        - Sentiment pooling (if enabled)
        - Macro pooling (if enabled)
        """
        if self.is_running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info("Starting DataIngestionOrchestrator...")
        
        # Connect to TimescaleDB
        await self.db.connect()
        
        # Initialize components
        self._initialize_components()
        
        # Launch pipelines
        self.is_running = True
        
        if self.config.mt5_enabled:
            task = asyncio.create_task(self._mt5_streaming_pipeline())
            self._tasks.append(task)
            logger.info("✓ MT5 streaming pipeline started")
        
        if self.config.news_enabled or self.config.reddit_enabled:
            task = asyncio.create_task(self._sentiment_pooling_pipeline())
            self._tasks.append(task)
            logger.info("✓ Sentiment pooling pipeline started")
        
        if self.config.fred_enabled or self.config.alphavantage_enabled:
            task = asyncio.create_task(self._macro_pooling_pipeline())
            self._tasks.append(task)
            logger.info("✓ Macro pooling pipeline started")
        
        # Health monitor
        task = asyncio.create_task(self._health_monitor())
        self._tasks.append(task)
        logger.info("✓ Health monitor started")
        
        logger.info(f"Orchestrator started with {len(self._tasks)} pipelines")
    
    async def stop(self):
        """Stop all pipelines gracefully"""
        logger.info("Stopping DataIngestionOrchestrator...")
        
        self.is_running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Disconnect DB
        await self.db.disconnect()
        
        logger.info("Orchestrator stopped")
    
    def _initialize_components(self):
        """Lazy-load data collectors"""
        # Sentiment analyzer (FinGPT)
        if self.config.news_enabled or self.config.reddit_enabled:
            self.sentiment_analyzer = FinGPTConnector()
            logger.info("✓ FinGPT sentiment analyzer loaded")
        
        # Reddit collector
        if self.config.reddit_enabled:
            if not self.config.reddit_client_id or not self.config.reddit_client_secret:
                logger.error("Reddit enabled but credentials missing (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)")
            else:
                self.reddit_collector = RedditSentimentCollector(
                    client_id=self.config.reddit_client_id,
                    client_secret=self.config.reddit_client_secret
                )
                logger.info("✓ Reddit collector initialized")
        
        # FRED collector
        if self.config.fred_enabled:
            if not self.config.fred_api_key:
                logger.error("FRED enabled but API key missing (FRED_API_KEY)")
            else:
                self.fred_collector = FREDCollector(api_key=self.config.fred_api_key)
                logger.info("✓ FRED collector initialized")
    
    # ========================================================================
    # PIPELINE 1: MT5 Streaming (Microestructura)
    # ========================================================================
    
    async def _mt5_streaming_pipeline(self):
        """
        Stream OHLCV data from MT5 (ZeroMQ)
        
        Frequency: Real-time (tick-by-tick or 1-min bars)
        Latency: <100ms
        """
        logger.info("MT5 streaming pipeline started")
        
        # TODO: Integrate with Mt5Connector (ZeroMQ)
        # For now, placeholder implementation
        
        while self.is_running:
            try:
                # PLACEHOLDER: Replace with actual Mt5Connector.stream_bars()
                # bars = await self.mt5_connector.stream_bars(symbols=self.config.mt5_symbols)
                
                # Simulate receiving bars
                await asyncio.sleep(60)  # 1-min bars
                
                # Example bar
                bar = OHLCVBar(
                    timestamp=datetime.now(),
                    symbol='EURUSD',
                    open=1.1000,
                    high=1.1010,
                    low=1.0990,
                    close=1.1005,
                    volume=1000.0,
                    spread=0.0001
                )
                
                # Insert to TimescaleDB
                await self.db.insert_ohlcv_batch([bar])
                
                self.stats['ohlcv_bars_ingested'] += 1
                self.stats['last_update']['ohlcv'] = datetime.now()
                
                logger.debug(f"Ingested OHLCV bar: {bar.symbol} @ {bar.close}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MT5 streaming error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(5)  # Retry delay
    
    # ========================================================================
    # PIPELINE 2: Sentiment Pooling (News + Reddit → FinGPT)
    # ========================================================================
    
    async def _sentiment_pooling_pipeline(self):
        """
        Poll news/Reddit, analyze with FinGPT, store sentiment scores
        
        Frequency: Every 15 minutes (configurable)
        Latency: ~5-10 seconds (FinGPT inference)
        """
        logger.info(f"Sentiment pooling pipeline started (interval: {self.config.sentiment_interval_minutes}min)")
        
        while self.is_running:
            try:
                sentiment_scores = []
                
                # 1. Collect Reddit posts
                if self.config.reddit_enabled and self.reddit_collector:
                    for symbol in self.config.mt5_symbols:
                        posts = self.reddit_collector.collect_symbol_posts(symbol, hours=1)
                        
                        if posts:
                            # Combine post texts
                            texts = [f"{p.title} {p.text}" for p in posts]
                            
                            # Analyze with FinGPT (batch)
                            sentiment = self.sentiment_analyzer.analyze_batch(texts, aggregate=True)
                            
                            score = SentimentScore(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                sentiment=sentiment,
                                source='reddit',
                                confidence=0.8  # TODO: Get confidence from FinGPT
                            )
                            sentiment_scores.append(score)
                            
                            logger.info(f"Reddit sentiment for {symbol}: {sentiment:.3f} ({len(posts)} posts)")
                
                # 2. Collect news (PLACEHOLDER - integrate News API)
                if self.config.news_enabled:
                    # TODO: Integrate News API (NewsAPI, GNews, Alpha Vantage News)
                    pass
                
                # 3. Insert to TimescaleDB
                if sentiment_scores:
                    await self.db.insert_sentiment_batch(sentiment_scores)
                    self.stats['sentiment_scores_ingested'] += len(sentiment_scores)
                    self.stats['last_update']['sentiment'] = datetime.now()
                
                # Wait for next interval
                await asyncio.sleep(self.config.sentiment_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sentiment pooling error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(60)  # Retry delay
    
    # ========================================================================
    # PIPELINE 3: Macro Pooling (FRED + Alpha Vantage)
    # ========================================================================
    
    async def _macro_pooling_pipeline(self):
        """
        Poll macroeconomic data (FRED, Alpha Vantage)
        
        Frequency: End-of-day (24h)
        Latency: ~1-2 seconds
        """
        logger.info(f"Macro pooling pipeline started (interval: {self.config.macro_interval_hours}h)")
        
        while self.is_running:
            try:
                # 1. Fetch FRED indicators
                if self.config.fred_enabled and self.fred_collector:
                    indicators_dict = await self.fred_collector.fetch_all_indicators()
                    
                    # Convert to MacroIndicator format
                    macro_indicators = [
                        MacroIndicator(
                            timestamp=ind.date,
                            series_id=ind.series_id,
                            value=ind.value,
                            name=ind.name
                        )
                        for ind in indicators_dict.values()
                        if ind
                    ]
                    
                    # Insert to TimescaleDB
                    if macro_indicators:
                        # TODO: Add insert_macro_batch() to TimescaleDBConnector
                        logger.info(f"Fetched {len(macro_indicators)} FRED indicators")
                        self.stats['macro_indicators_ingested'] += len(macro_indicators)
                        self.stats['last_update']['macro'] = datetime.now()
                
                # 2. Fetch Alpha Vantage fundamentals (PLACEHOLDER)
                if self.config.alphavantage_enabled:
                    # TODO: Integrate Alpha Vantage API
                    pass
                
                # Wait for next interval (24h)
                await asyncio.sleep(self.config.macro_interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Macro pooling error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(3600)  # Retry in 1h
    
    # ========================================================================
    # Health Monitor
    # ========================================================================
    
    async def _health_monitor(self):
        """
        Monitor pipeline health and log statistics
        
        Frequency: Every 5 minutes
        """
        while self.is_running:
            try:
                await asyncio.sleep(300)  # 5 min
                
                logger.info("=== Data Ingestion Health ===")
                logger.info(f"OHLCV bars ingested: {self.stats['ohlcv_bars_ingested']}")
                logger.info(f"Sentiment scores ingested: {self.stats['sentiment_scores_ingested']}")
                logger.info(f"Macro indicators ingested: {self.stats['macro_indicators_ingested']}")
                logger.info(f"Errors: {self.stats['errors']}")
                logger.info(f"Last OHLCV update: {self.stats['last_update']['ohlcv']}")
                logger.info(f"Last sentiment update: {self.stats['last_update']['sentiment']}")
                logger.info(f"Last macro update: {self.stats['last_update']['macro']}")
                logger.info("============================")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    import os
    
    async def test():
        print("Testing DataIngestionOrchestrator...")
        
        # Load API keys from environment
        config = DataSourceConfig(
            mt5_enabled=True,
            mt5_symbols=['EURUSD', 'GBPUSD'],
            news_enabled=False,  # Disabled for test
            reddit_enabled=True,
            reddit_client_id=os.getenv('REDDIT_CLIENT_ID'),
            reddit_client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            sentiment_interval_minutes=1,  # Fast test
            fred_enabled=True,
            fred_api_key=os.getenv('FRED_API_KEY'),
            macro_interval_hours=1,  # Fast test
        )
        
        orchestrator = DataIngestionOrchestrator(config)
        
        try:
            await orchestrator.start()
            
            # Run for 2 minutes
            await asyncio.sleep(120)
            
        finally:
            await orchestrator.stop()
        
        print("\n=== Final Stats ===")
        print(f"OHLCV: {orchestrator.stats['ohlcv_bars_ingested']}")
        print(f"Sentiment: {orchestrator.stats['sentiment_scores_ingested']}")
        print(f"Macro: {orchestrator.stats['macro_indicators_ingested']}")
        print(f"Errors: {orchestrator.stats['errors']}")
    
    asyncio.run(test())
