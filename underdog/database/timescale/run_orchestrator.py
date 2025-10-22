"""
Data Ingestion Orchestrator - Production Runner

Runs 3 async pipelines 24/7:
1. MT5 Streaming (tick-by-tick, <100ms latency)
2. Reddit Sentiment Pooling (every 15 min → FinGPT analysis)
3. FRED Macro Indicators (EOD daily update)

Usage:
    poetry run python underdog/database/timescale/run_orchestrator.py
    
    # Custom config
    poetry run python underdog/database/timescale/run_orchestrator.py --config config/custom.yaml
    
    # Debug mode
    poetry run python underdog/database/timescale/run_orchestrator.py --debug

Business Goal:
    - Real-time data ingestion for DRL StateVectorBuilder
    - Alternative data (sentiment + macro) for alpha generation
    - Health monitoring and auto-recovery
"""

import asyncio
import argparse
import signal
import sys
from pathlib import Path
from datetime import datetime, time as datetime_time
import logging
from typing import Optional
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from underdog.database.timescale.data_orchestrator import (
    DataIngestionOrchestrator,
    DataSourceConfig
)
from underdog.data.reddit_collector import RedditSentimentCollector
from underdog.sentiment.llm_connector import FinGPTConnector
from underdog.database.timescale.timescale_connector import TimescaleDBConnector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/orchestrator.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class ProductionOrchestrator:
    """
    Production wrapper for DataIngestionOrchestrator
    
    Handles:
    - Configuration loading
    - Graceful shutdown
    - Auto-recovery on errors
    - Health monitoring
    """
    
    def __init__(self, config_path: str = "config/data_providers.yaml"):
        self.config = self._load_config(config_path)
        self.orchestrator: Optional[DataIngestionOrchestrator] = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML"""
        logger.info(f"Loading configuration from {path}")
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        logger.info(f"  MT5 Streaming: {'ENABLED' if config['ingestion']['mt5_streaming']['enabled'] else 'DISABLED'}")
        logger.info(f"  Sentiment Pooling: {'ENABLED' if config['ingestion']['sentiment_pooling']['enabled'] else 'DISABLED'}")
        logger.info(f"  Macro Indicators: {'ENABLED' if config['ingestion']['macro_indicators']['enabled'] else 'DISABLED'}")
        
        return config
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self.running = False
    
    async def initialize(self):
        """Initialize orchestrator and all data sources"""
        logger.info("="*80)
        logger.info("INITIALIZING DATA INGESTION ORCHESTRATOR")
        logger.info("="*80)
        
        # Initialize TimescaleDB
        db = TimescaleDBConnector(
            host=self.config['timescaledb']['host'],
            port=self.config['timescaledb']['port'],
            database=self.config['timescaledb']['database'],
            user=self.config['timescaledb']['user'],
            password=self.config['timescaledb']['password']
        )
        
        await db.connect()
        logger.info("✓ TimescaleDB connected")
        
        # Initialize data sources
        sources = []
        
        # 1. MT5 Streaming (if enabled)
        if self.config['ingestion']['mt5_streaming']['enabled']:
            mt5_config = DataSourceConfig(
                source_id='mt5_streaming',
                api_key=None,  # ZMQ connector doesn't need API key
                endpoint=f"tcp://{self.config['mt5']['zmq_host']}:{self.config['mt5']['zmq_stream_port']}",
                interval_seconds=0,  # Real-time streaming
                symbols=self.config['ingestion']['mt5_streaming']['symbols']
            )
            sources.append(mt5_config)
            logger.info(f"✓ MT5 Streaming configured for {len(mt5_config.symbols)} symbols")
        
        # 2. Reddit Sentiment (if enabled)
        if self.config['ingestion']['sentiment_pooling']['enabled']:
            reddit_config = DataSourceConfig(
                source_id='reddit_sentiment',
                api_key=self.config['reddit']['client_secret'],  # Used as credentials
                endpoint='https://oauth.reddit.com',
                interval_seconds=self.config['reddit']['polling_interval_minutes'] * 60,
                symbols=self.config['mt5']['symbols']
            )
            sources.append(reddit_config)
            logger.info(f"✓ Reddit Sentiment configured (polling every {self.config['reddit']['polling_interval_minutes']} min)")
        
        # 3. FRED Macro (if enabled)
        if self.config['ingestion']['macro_indicators']['enabled']:
            fred_config = DataSourceConfig(
                source_id='fred_macro',
                api_key=self.config['fred']['api_key'],
                endpoint='https://api.stlouisfed.org/fred/series/observations',
                interval_seconds=86400,  # Daily (EOD)
                symbols=list(self.config['fred']['series'].values())
            )
            sources.append(fred_config)
            logger.info(f"✓ FRED Macro configured for {len(self.config['fred']['series'])} series")
        
        # Initialize orchestrator
        self.orchestrator = DataIngestionOrchestrator(
            db_connector=db,
            data_sources=sources
        )
        
        logger.info("="*80)
        logger.info(f"Orchestrator initialized with {len(sources)} data sources")
        logger.info("="*80 + "\n")
    
    async def run(self):
        """Run orchestrator with health monitoring"""
        self.running = True
        
        try:
            await self.initialize()
            
            logger.info("🚀 Starting Data Ingestion Orchestrator...")
            logger.info("   Press Ctrl+C to stop\n")
            
            # Start orchestrator
            orchestrator_task = asyncio.create_task(self.orchestrator.start())
            
            # Health monitoring loop
            while self.running:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.running:
                    break
                
                # Log status
                logger.info(f"[Health Check] Orchestrator running - {datetime.now().isoformat()}")
            
            # Graceful shutdown
            logger.info("\n🛑 Stopping Data Ingestion Orchestrator...")
            await self.orchestrator.stop()
            orchestrator_task.cancel()
            
            try:
                await orchestrator_task
            except asyncio.CancelledError:
                pass
            
            logger.info("✓ Orchestrator stopped gracefully")
        
        except Exception as e:
            logger.error(f"❌ Fatal error in orchestrator: {e}", exc_info=True)
            raise
        
        finally:
            if self.orchestrator and self.orchestrator.db:
                await self.orchestrator.db.disconnect()
                logger.info("✓ Database disconnected")


async def main():
    parser = argparse.ArgumentParser(description='Run Data Ingestion Orchestrator')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/data_providers.yaml',
        help='Path to configuration file (default: config/data_providers.yaml)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Run orchestrator
    runner = ProductionOrchestrator(config_path=args.config)
    
    try:
        await runner.run()
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    # Run
    asyncio.run(main())
