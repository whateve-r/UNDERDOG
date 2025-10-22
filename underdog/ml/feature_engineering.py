"""
State Vector Builder - 14-Dimensional Feature Vector for DRL

Construye el state vector para el TD3 Agent combinando:
1. Price features (price, returns, ATR)
2. Technical indicators (MACD, RSI, Bollinger Bands)
3. Sentiment score (from FinGPT/Redis)
4. Regime classification (trend/range/transition one-hot)
5. Macro indicators (VIX, Fed Funds Rate, Yield Curve)

State Vector (14 dimensions):
[
    0. Close price (normalized)
    1. Log returns
    2. ATR (volatility)
    3. MACD
    4. RSI (overbought/oversold)
    5. Bollinger Band width
    6. Sentiment score [-1, 1]
    7. Regime: Trend (one-hot)
    8. Regime: Range (one-hot)
    9. Regime: Transition (one-hot)
    10. VIX (market fear)
    11. Fed Funds Rate (interest rate)
    12. Yield Curve (T10Y2Y, recession indicator)
    13. Volume ratio
]

Papers:
- arXiv:2510.10526v1: LLM + RL Integration (sentiment feature)
- arXiv:2510.03236v1: Regime-Switching (regime one-hot)
- AlphaQuanter.pdf: Multi-source feature engineering

Architecture:
    TimescaleDB ──┐
    Redis Cache ───┼──→ StateVectorBuilder ──→ np.array(14,) ──→ TD3 Agent
    RegimeModel ───┘
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from underdog.database.timescale.timescale_connector import TimescaleDBConnector

logger = logging.getLogger(__name__)


@dataclass
class StateVectorConfig:
    """Configuration for state vector construction - EXPANDED for CMDP"""
    # Technical indicators (expanded from 6 to 15+)
    atr_period: int = 14
    atr_long_period: int = 50  # For market turbulence index
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14
    cci_period: int = 20
    stoch_period: int = 14
    
    # Normalization
    price_window: int = 100  # Lookback for price normalization
    returns_clip: float = 0.05  # Clip returns to [-5%, +5%]
    
    # Risk/Position tracking (CRITICAL for CMDP)
    max_daily_dd_pct: float = 0.05  # 5% max daily drawdown
    max_total_dd_pct: float = 0.10  # 10% max total drawdown
    
    # Cache settings
    use_redis: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Fallback values (if data missing)
    default_sentiment: float = 0.0
    default_vix: float = 20.0
    default_fed_rate: float = 5.0
    default_yield_curve: float = 0.5


class StateVectorBuilder:
    """
    Builds EXPANDED state vectors for DRL Agent (20+ dimensions)
    
    **CRITICAL UPDATES FOR CMDP (Constrained MDP)**:
    - Added position/risk awareness (current_position, cash_balance, DD_used)
    - Added market turbulence index (volatility regime detection)
    - Expanded technical indicators (15+ features)
    
    Responsibilities:
    1. Query TimescaleDB for OHLCV, sentiment, macro data
    2. Calculate expanded technical indicators (15+)
    3. Query regime classifier prediction
    4. **NEW**: Track position & risk metrics for CMDP
    5. Normalize features to [-1, 1] or [0, 1] range
    6. Cache results in Redis for low latency
    
    State Vector Dimensions (24 total):
    [0-2]   Price features (normalized price, returns, volatility)
    [3-8]   Technical indicators (RSI, MACD, ATR, BB, ADX, CCI)
    [9]     Sentiment score
    [10-12] Regime features (trend, range, transition confidence)
    [13-15] Macro features (VIX, Fed Rate, Yield Curve)
    [16]    Volume ratio
    [17]    **NEW** Current position size [-1, 1]
    [18]    **NEW** Current cash balance [0, 1] (normalized)
    [19]    **NEW** Daily drawdown used [0, 1] (% of limit)
    [20]    **NEW** Total drawdown used [0, 1] (% of limit)
    [21]    **NEW** Market turbulence index [0, 1]
    [22-23] **NEW** Additional technical indicators (Stochastic, momentum)
    
    Usage:
        builder = StateVectorBuilder(db_connector)
        
        # Pass position/account state for CMDP
        state = await builder.build_state(
            symbol='EURUSD',
            timestamp=datetime.now(),
            current_position=0.5,  # 50% long
            cash_balance=95000,
            daily_dd=2500,  # $2.5k DD today
            total_dd=7500,  # $7.5k total DD
            initial_balance=100000
        )
        # Returns: np.array([...], shape=(24,))
    """
    
    def __init__(
        self,
        db_connector: TimescaleDBConnector,
        config: Optional[StateVectorConfig] = None
    ):
        """
        Initialize builder
        
        Args:
            db_connector: TimescaleDB connector
            config: Configuration (uses defaults if None)
        """
        self.db = db_connector
        self.config = config or StateVectorConfig()
        
        # Redis cache (optional)
        self.redis_client = None
        if self.config.use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
                self.redis_client.ping()
                logger.info("Redis cache enabled")
            except Exception as e:
                logger.warning(f"Redis connection failed, caching disabled: {e}")
                self.redis_client = None
        
        logger.info("StateVectorBuilder initialized (24-dimensional CMDP mode)")
    
    async def build_state(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_bars: int = 200,
        # **NEW PARAMETERS FOR CMDP**
        current_position: float = 0.0,
        cash_balance: float = 100000.0,
        daily_dd: float = 0.0,
        total_dd: float = 0.0,
        initial_balance: float = 100000.0
    ) -> np.ndarray:
        """
        Build complete state vector for given symbol and timestamp
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timestamp: Current timestamp
            lookback_bars: Historical bars to fetch (for indicators)
            current_position: Current position size [-1, 1] (**CRITICAL for CMDP**)
            cash_balance: Current cash/equity balance
            daily_dd: Current daily drawdown value (absolute, e.g., $2500)
            total_dd: Current total drawdown value (absolute, e.g., $7500)
            initial_balance: Initial account balance (for normalization)
        
        Returns:
            State vector: np.array(shape=(24,), dtype=float32)
        """
        # 1. Check Redis cache (disabled for CMDP due to position dependency)
        # Note: Position state changes per step, so caching may be counterproductive
        
        # 2. Fetch OHLCV data
        start_date = timestamp - timedelta(minutes=lookback_bars)
        df_ohlcv = await self.db.query_ohlcv(symbol, start_date, timestamp)
        
        if df_ohlcv.empty or len(df_ohlcv) < 50:
            logger.error(f"Insufficient OHLCV data for {symbol} (need 50 bars, got {len(df_ohlcv)})")
            return self._build_fallback_state()
        
        # 3. Calculate price features (0, 1, 2)
        price_features = self._calculate_price_features(df_ohlcv)
        
        # 4. Calculate EXPANDED technical indicators (3-8)
        technical_features = self._calculate_technical_features_extended(df_ohlcv)
        
        # 5. Fetch sentiment score (9)
        sentiment = await self._fetch_sentiment(symbol, timestamp)
        
        # 6. Fetch regime classification (10, 11, 12)
        regime_features = await self._fetch_regime(symbol, timestamp)
        
        # 7. Fetch macro indicators (13, 14, 15)
        macro_features = await self._fetch_macro(timestamp)
        
        # 8. Calculate volume ratio (16)
        volume_ratio = self._calculate_volume_ratio(df_ohlcv)
        
        # **NEW 9. Calculate position/risk features (17-20) - CMDP CRITICAL**
        position_features = self._calculate_position_features(
            current_position=current_position,
            cash_balance=cash_balance,
            daily_dd=daily_dd,
            total_dd=total_dd,
            initial_balance=initial_balance
        )
        
        # **NEW 10. Calculate market turbulence index (21)**
        turbulence = self._calculate_market_turbulence(df_ohlcv)
        
        # **NEW 11. Calculate additional technical indicators (22-23)**
        stoch_momentum = self._calculate_stochastic_momentum(df_ohlcv)
        
        # 12. Assemble EXPANDED state vector (24 dimensions)
        state = np.array([
            # Price features (0-2)
            price_features['price_norm'],      # 0: Normalized price [0, 1]
            price_features['returns'],          # 1: Log returns [-1, 1]
            price_features['volatility'],       # 2: Short-term volatility [0, 1]
            
            # Technical indicators (3-8)
            technical_features['rsi_norm'],     # 3: RSI [0, 1]
            technical_features['macd_norm'],    # 4: MACD [-1, 1]
            technical_features['atr_norm'],     # 5: ATR [0, 1]
            technical_features['bb_width'],     # 6: Bollinger width [0, 1]
            technical_features['adx_norm'],     # 7: ADX [0, 1]
            technical_features['cci_norm'],     # 8: CCI [-1, 1]
            
            # Sentiment (9)
            sentiment,                          # 9: Sentiment score [-1, 1]
            
            # Regime features (10-12)
            regime_features['trend'],           # 10: Trend confidence [0, 1]
            regime_features['range'],           # 11: Range confidence [0, 1]
            regime_features['transition'],      # 12: Transition confidence [0, 1]
            
            # Macro features (13-15)
            macro_features['vix_norm'],         # 13: VIX [0, 1]
            macro_features['fed_rate_norm'],    # 14: Fed Rate [0, 1]
            macro_features['yield_curve_norm'], # 15: Yield Curve [-1, 1]
            
            # Volume (16)
            volume_ratio,                       # 16: Volume ratio [0, 1]
            
            # **NEW: Position/Risk features - CMDP CRITICAL (17-20)**
            position_features['position_norm'], # 17: Current position [-1, 1]
            position_features['cash_norm'],     # 18: Cash balance [0, 1]
            position_features['daily_dd_pct'],  # 19: Daily DD used [0, 1]
            position_features['total_dd_pct'],  # 20: Total DD used [0, 1]
            
            # **NEW: Market turbulence (21)**
            turbulence,                         # 21: Turbulence index [0, 1]
            
            # **NEW: Additional technicals (22-23)**
            stoch_momentum['stoch_k'],          # 22: Stochastic %K [0, 1]
            stoch_momentum['momentum_norm'],    # 23: Momentum [-1, 1]
        ], dtype=np.float32)
        
        # 13. Sanity check
        assert state.shape == (24,), f"State shape mismatch: {state.shape} != (24,)"
        
        logger.debug(f"Built CMDP state vector for {symbol}: position={current_position:.2f}, DD={daily_dd:.0f}/{total_dd:.0f}")
        return state
    
    def _calculate_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate price-based features (dimensions 0, 1, 2)
        
        Returns:
            {
                'price_norm': normalized close price [0, 1],
                'returns': log returns [-1, 1],
                'atr_norm': normalized ATR [0, 1]
            }
        """
        close = df['close'].values
        
        # 0. Normalized price (min-max over lookback window)
        price_min = close[-self.config.price_window:].min()
        price_max = close[-self.config.price_window:].max()
        price_norm = (close[-1] - price_min) / (price_max - price_min + 1e-8)
        
        # 1. Log returns (clipped to [-5%, +5%])
        if len(close) >= 2:
            returns = np.log(close[-1] / close[-2])
            returns = np.clip(returns, -self.config.returns_clip, self.config.returns_clip)
            returns_norm = returns / self.config.returns_clip  # Normalize to [-1, 1]
        else:
            returns_norm = 0.0
        
        return {
            'price_norm': float(price_norm),
            'returns': float(returns_norm)
        }
    
    def _calculate_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate technical indicators (dimensions 3, 4, 5)
        
        Returns:
            {
                'atr_norm': ATR normalized [0, 1],
                'macd_norm': MACD normalized [-1, 1],
                'rsi_norm': RSI normalized [0, 1],
                'bb_width': Bollinger Band width [0, 1]
            }
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 2. ATR (Average True Range)
        tr = np.maximum(high[1:] - low[1:], 
                       np.abs(high[1:] - close[:-1]))
        atr = np.mean(tr[-self.config.atr_period:]) if len(tr) >= self.config.atr_period else 0.0
        atr_norm = atr / (close[-1] + 1e-8)  # Normalize by price
        
        # 3. MACD
        ema_fast = self._ema(close, self.config.macd_fast)
        ema_slow = self._ema(close, self.config.macd_slow)
        macd = ema_fast - ema_slow
        macd_norm = np.tanh(macd / (close[-1] + 1e-8) * 100)  # Normalize to [-1, 1]
        
        # 4. RSI
        rsi = self._calculate_rsi(close, self.config.rsi_period)
        rsi_norm = (rsi - 50) / 50  # Normalize to [-1, 1] (50 is neutral)
        
        # 5. Bollinger Band width
        sma = np.mean(close[-self.config.bb_period:])
        std = np.std(close[-self.config.bb_period:])
        bb_width = (std * self.config.bb_std * 2) / (sma + 1e-8)
        bb_width_norm = np.clip(bb_width, 0, 0.1) / 0.1  # Normalize to [0, 1]
        
        return {
            'atr_norm': float(atr_norm),
            'macd_norm': float(macd_norm),
            'rsi_norm': float(rsi_norm),
            'bb_width': float(bb_width_norm)
        }
    
    async def _fetch_sentiment(self, symbol: str, timestamp: datetime) -> float:
        """
        Fetch latest sentiment score from TimescaleDB (dimension 6)
        
        Returns:
            Sentiment score in [-1, 1] range
        """
        try:
            sentiment = await self.db.get_latest_sentiment(symbol, source='reddit')
            if sentiment is None:
                logger.warning(f"No sentiment data for {symbol}, using default")
                return self.config.default_sentiment
            return float(np.clip(sentiment, -1, 1))
        except Exception as e:
            logger.error(f"Failed to fetch sentiment: {e}")
            return self.config.default_sentiment
    
    async def _fetch_regime(self, symbol: str, timestamp: datetime) -> Dict[str, float]:
        """
        Fetch regime classification (dimensions 7, 8, 9)
        
        Returns:
            One-hot encoded regime: {'trend': 0/1, 'range': 0/1, 'transition': 0/1}
        """
        try:
            # TODO: Query regime_predictions table
            # For now, placeholder implementation
            
            # Simulate regime prediction
            regime = 'trend'  # Placeholder
            
            return {
                'trend': 1.0 if regime == 'trend' else 0.0,
                'range': 1.0 if regime == 'range' else 0.0,
                'transition': 1.0 if regime == 'transition' else 0.0,
            }
        except Exception as e:
            logger.error(f"Failed to fetch regime: {e}")
            return {'trend': 0.33, 'range': 0.33, 'transition': 0.33}  # Uniform fallback
    
    async def _fetch_macro(self, timestamp: datetime) -> Dict[str, float]:
        """
        Fetch macroeconomic indicators (dimensions 10, 11, 12)
        
        Returns:
            {
                'vix_norm': VIX normalized [0, 1],
                'fed_rate_norm': Fed Funds Rate normalized [0, 1],
                'yield_curve_norm': Yield Curve (T10Y2Y) normalized [-1, 1]
            }
        """
        try:
            # TODO: Query macro_indicators table
            # For now, use defaults
            
            vix = self.config.default_vix
            fed_rate = self.config.default_fed_rate
            yield_curve = self.config.default_yield_curve
            
            return {
                'vix_norm': float(np.clip(vix / 80.0, 0, 1)),  # VIX typically 10-80
                'fed_rate_norm': float(np.clip(fed_rate / 10.0, 0, 1)),  # Fed rate 0-10%
                'yield_curve_norm': float(np.tanh(yield_curve)),  # Yield curve -2% to +2%
            }
        except Exception as e:
            logger.error(f"Failed to fetch macro: {e}")
            return {
                'vix_norm': 0.25,
                'fed_rate_norm': 0.5,
                'yield_curve_norm': 0.0
            }
    
    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate volume ratio (dimension 13)
        
        Returns:
            Current volume / 20-period average volume [0, 1]
        """
        volume = df['volume'].values
        if len(volume) < 20:
            return 0.5  # Neutral
        
        avg_volume = np.mean(volume[-20:])
        volume_ratio = volume[-1] / (avg_volume + 1e-8)
        
        # Normalize to [0, 1] (assuming volume ratio typically 0-3x)
        return float(np.clip(volume_ratio / 3.0, 0, 1))
    
    def _build_fallback_state(self) -> np.ndarray:
        """
        Build fallback state with neutral values (when data is missing)
        """
        logger.warning("Building fallback state (data missing)")
        return np.array([
            0.5,  # price_norm
            0.0,  # returns
            0.02, # atr_norm
            0.0,  # macd_norm
            0.0,  # rsi_norm
            0.05, # bb_width
            0.0,  # sentiment
            0.33, # regime_trend
            0.33, # regime_range
            0.33, # regime_transition
            0.25, # vix_norm
            0.5,  # fed_rate_norm
            0.0,  # yield_curve_norm
            0.5,  # volume_ratio
        ], dtype=np.float32)
    
    # ========================================================================
    # Helper Functions
    # ========================================================================
    
    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return data[-1]
        
        alpha = 2 / (period + 1)
        ema = data[-period]
        for price in data[-period+1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    @staticmethod
    def _calculate_rsi(close: np.ndarray, period: int) -> float:
        """Calculate Relative Strength Index"""
        if len(close) < period + 1:
            return 50.0  # Neutral
        
        deltas = np.diff(close[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _get_from_cache(self, key: str) -> Optional[np.ndarray]:
        """Get state vector from Redis cache"""
        try:
            data = self.redis_client.get(key)
            if data:
                return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Cache GET failed: {e}")
        return None
    
    def _save_to_cache(self, key: str, state: np.ndarray):
        """Save state vector to Redis cache"""
        try:
            self.redis_client.setex(
                key,
                self.config.cache_ttl_seconds,
                state.tobytes()
            )
        except Exception as e:
            logger.debug(f"Cache SET failed: {e}")


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from datetime import datetime
    
    async def test():
        print("Testing StateVectorBuilder...")
        
        # Create DB connector
        db = TimescaleDBConnector()
        await db.connect()
        
        # Create builder
        builder = StateVectorBuilder(db)
        
        # Build state
        symbol = 'EURUSD'
        timestamp = datetime.now()
        
        print(f"\nBuilding state for {symbol} at {timestamp}...")
        state = await builder.build_state(symbol, timestamp, lookback_bars=200)
        
        print(f"\nState Vector (14 dimensions):")
        print(f"  Shape: {state.shape}")
        print(f"  Dtype: {state.dtype}")
        print(f"  Min: {state.min():.4f}, Max: {state.max():.4f}")
        print(f"\n  Values:")
        feature_names = [
            '0. Price (norm)',
            '1. Returns',
            '2. ATR (norm)',
            '3. MACD (norm)',
            '4. RSI (norm)',
            '5. BB Width',
            '6. Sentiment',
            '7. Regime: Trend',
            '8. Regime: Range',
            '9. Regime: Transition',
            '10. VIX (norm)',
            '11. Fed Rate (norm)',
            '12. Yield Curve (norm)',
            '13. Volume Ratio',
        ]
        
        for i, (name, value) in enumerate(zip(feature_names, state)):
            print(f"    {name:25} = {value:7.4f}")
        
        await db.disconnect()
        print("\n✓ Test complete")
    
    asyncio.run(test())
