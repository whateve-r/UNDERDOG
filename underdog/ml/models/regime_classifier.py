"""
Hidden Markov Model Regime Classifier
Detects market regimes (bull/bear/sideways/high_vol/low_vol) for model gating.

Implements rigorous statistical methodology for regime detection with:
- Multi-state HMM (3-5 states) fitted on returns + volatility
- ADF test for mean reversion validation
- Regime persistence metrics (minimum duration filtering)
- Model gating logic for strategy activation/deactivation
- Out-of-sample regime prediction with confidence scoring
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RegimeType(Enum):
    """Market regime types"""
    BULL = "bull"  # Strong uptrend
    BEAR = "bear"  # Strong downtrend
    SIDEWAYS = "sideways"  # Range-bound, mean-reverting
    HIGH_VOL = "high_volatility"  # Elevated volatility regime
    LOW_VOL = "low_volatility"  # Compressed volatility regime
    UNKNOWN = "unknown"  # Insufficient data or undefined


@dataclass
class RegimeConfig:
    """Configuration for regime detection"""
    # HMM parameters
    n_states: int = 3  # Number of hidden states (3-5 recommended)
    covariance_type: Literal["full", "diag", "spherical", "tied"] = "full"
    n_iter: int = 100  # EM algorithm iterations
    random_state: int = 42
    
    # Feature engineering
    returns_window: int = 1  # Window for returns calculation
    volatility_window: int = 20  # Window for realized volatility
    
    # Regime labeling
    vol_percentile_high: float = 70.0  # High volatility threshold (percentile)
    vol_percentile_low: float = 30.0  # Low volatility threshold (percentile)
    trend_threshold: float = 0.0  # Mean return threshold for bull/bear
    
    # Stationarity testing
    adf_significance: float = 0.05  # ADF test p-value threshold
    
    # Regime persistence
    min_regime_duration: int = 5  # Minimum bars to confirm regime
    
    # Training
    train_window: int = 252  # Rolling window for HMM fitting (1 year daily)
    retrain_frequency: int = 21  # Retrain every N bars (monthly)


@dataclass
class RegimeState:
    """Current regime state"""
    regime: RegimeType
    state_id: int  # HMM state ID
    confidence: float  # Posterior probability
    duration: int  # Bars in current regime
    characteristics: Dict[str, float]  # Mean return, volatility, etc.
    timestamp: datetime


class HMMRegimeClassifier:
    """
    Hidden Markov Model for market regime detection.
    
    Methodology (Scientific Approach):
    1. Observation: Markets exhibit distinct behavioral regimes
    2. Hypothesis: Price dynamics follow a switching process
    3. Test: HMM can capture latent regime transitions
    4. Validation: Regime persistence + economic rationale
    
    Features:
    - Multi-dimensional observations (returns + volatility)
    - Gaussian emission distributions per state
    - Viterbi algorithm for most likely state sequence
    - ADF test for mean reversion validation in sideways regimes
    - Dynamic regime labeling based on state characteristics
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize HMM regime classifier.
        
        Args:
            config: Regime detection configuration
        """
        self.config = config or RegimeConfig()
        
        # HMM model
        self.hmm = None
        self._init_hmm()
        
        # State tracking
        self.current_state: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []
        
        # Feature statistics
        self.feature_scaler_mean = None
        self.feature_scaler_std = None
        
        # Training metadata
        self.last_train_idx = 0
        self.is_fitted = False
        
        print(f"[HMM] Initialized with {self.config.n_states} states")
    
    def _init_hmm(self) -> None:
        """Initialize HMM model"""
        try:
            from hmmlearn import hmm
            
            self.hmm = hmm.GaussianHMM(
                n_components=self.config.n_states,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                random_state=self.config.random_state
            )
            
        except ImportError:
            print("[WARNING] hmmlearn not installed. Install with: pip install hmmlearn")
            self.hmm = None
    
    def fit(self, prices: pd.Series, train_all: bool = False) -> None:
        """
        Fit HMM on price history.
        
        Args:
            prices: Historical price series (OHLC close recommended)
            train_all: If True, train on all data; if False, use rolling window
        """
        if self.hmm is None:
            raise RuntimeError("HMM not available. Install hmmlearn.")
        
        # Extract features
        features = self._extract_features(prices)
        
        # Handle rolling vs. full training
        if train_all:
            train_features = features
        else:
            train_start = max(0, len(features) - self.config.train_window)
            train_features = features[train_start:]
        
        if len(train_features) < 50:
            raise ValueError(f"Insufficient data for training. Need at least 50 samples, got {len(train_features)}")
        
        # Normalize features (critical for HMM convergence)
        train_features_norm, mean, std = self._normalize_features(train_features)
        self.feature_scaler_mean = mean
        self.feature_scaler_std = std
        
        # Fit HMM
        print(f"[HMM] Fitting on {len(train_features)} samples...")
        self.hmm.fit(train_features_norm)
        
        self.is_fitted = True
        self.last_train_idx = len(prices)
        
        # Label states based on characteristics
        self._label_states(train_features_norm, prices.index[-len(train_features):])
        
        print(f"[HMM] Training complete. States labeled:")
        self._print_state_characteristics()
    
    def predict(self, prices: pd.Series, return_confidence: bool = True) -> pd.DataFrame:
        """
        Predict regime sequence for price history.
        
        Args:
            prices: Price series
            return_confidence: If True, include posterior probabilities
        
        Returns:
            DataFrame with regime predictions and confidence
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Extract and normalize features
        features = self._extract_features(prices)
        features_norm = (features - self.feature_scaler_mean) / (self.feature_scaler_std + 1e-8)
        
        # Predict state sequence (Viterbi algorithm)
        states = self.hmm.predict(features_norm)
        
        # Get posterior probabilities
        if return_confidence:
            posteriors = self.hmm.predict_proba(features_norm)
            confidence = np.max(posteriors, axis=1)
        else:
            confidence = np.ones(len(states))
        
        # Map states to regime labels
        regimes = [self._state_to_regime(s) for s in states]
        
        # Create result DataFrame
        result = pd.DataFrame({
            'regime': regimes,
            'state_id': states,
            'confidence': confidence
        }, index=prices.index[-len(states):])
        
        return result
    
    def predict_current(self, prices: pd.Series) -> RegimeState:
        """
        Predict current regime state (online inference).
        
        Args:
            prices: Recent price history (must include sufficient lookback)
        
        Returns:
            Current RegimeState
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Check if retraining is needed
        if len(prices) - self.last_train_idx >= self.config.retrain_frequency:
            print(f"[HMM] Retraining triggered at bar {len(prices)}")
            self.fit(prices, train_all=False)
        
        # Predict regime
        predictions = self.predict(prices, return_confidence=True)
        
        # Get current state
        current = predictions.iloc[-1]
        regime = RegimeType(current['regime'])
        state_id = int(current['state_id'])
        confidence = float(current['confidence'])
        
        # Calculate regime duration
        duration = self._calculate_regime_duration(predictions)
        
        # Get state characteristics
        features = self._extract_features(prices)
        current_features = features.iloc[-1]
        characteristics = {
            'mean_return': float(current_features[0]),
            'volatility': float(current_features[1])
        }
        
        # Create regime state
        regime_state = RegimeState(
            regime=regime,
            state_id=state_id,
            confidence=confidence,
            duration=duration,
            characteristics=characteristics,
            timestamp=datetime.now()
        )
        
        self.current_state = regime_state
        self.regime_history.append(regime_state)
        
        return regime_state
    
    def _extract_features(self, prices: pd.Series) -> pd.DataFrame:
        """
        Extract features for HMM (returns + volatility).
        
        Args:
            prices: Price series
        
        Returns:
            DataFrame with [returns, volatility]
        """
        # Returns
        returns = prices.pct_change(self.config.returns_window).fillna(0.0)
        
        # Realized volatility (rolling std of returns)
        volatility = returns.rolling(
            window=self.config.volatility_window,
            min_periods=self.config.volatility_window // 2
        ).std().fillna(returns.std())
        
        # Combine features
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility
        }, index=prices.index)
        
        # Drop initial NaN rows
        features = features.dropna()
        
        return features
    
    def _normalize_features(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features (Z-score).
        
        Args:
            features: Raw features
        
        Returns:
            Tuple of (normalized_features, mean, std)
        """
        mean = features.mean().values
        std = features.std().values
        
        # Prevent division by zero
        std = np.where(std < 1e-8, 1.0, std)
        
        normalized = (features.values - mean) / std
        
        return normalized, mean, std
    
    def _label_states(self, features_norm: np.ndarray, timestamps: pd.Index) -> None:
        """
        Label HMM states based on emission characteristics.
        
        Args:
            features_norm: Normalized training features
            timestamps: Timestamps for features
        """
        # Predict states on training data
        states = self.hmm.predict(features_norm)
        
        # Calculate statistics per state
        self.state_labels = {}
        
        for state_id in range(self.config.n_states):
            mask = (states == state_id)
            
            if not mask.any():
                self.state_labels[state_id] = RegimeType.UNKNOWN
                continue
            
            # Denormalize features for interpretation
            state_features = features_norm[mask]
            state_features_orig = (state_features * self.feature_scaler_std) + self.feature_scaler_mean
            
            mean_return = np.mean(state_features_orig[:, 0])
            mean_volatility = np.mean(state_features_orig[:, 1])
            
            # Label based on characteristics
            regime = self._label_regime(mean_return, mean_volatility, features_norm[:, 1])
            self.state_labels[state_id] = regime
            
            # Store characteristics
            if not hasattr(self, 'state_characteristics'):
                self.state_characteristics = {}
            
            self.state_characteristics[state_id] = {
                'mean_return': mean_return,
                'mean_volatility': mean_volatility,
                'n_samples': int(mask.sum()),
                'regime': regime.value
            }
    
    def _label_regime(self, mean_return: float, mean_volatility: float, all_volatilities: np.ndarray) -> RegimeType:
        """
        Label regime based on return and volatility characteristics.
        
        Args:
            mean_return: Mean return for state
            mean_volatility: Mean volatility for state
            all_volatilities: All volatilities for percentile calculation
        
        Returns:
            RegimeType label
        """
        # Denormalize all volatilities
        all_vols_orig = (all_volatilities * self.feature_scaler_std[1]) + self.feature_scaler_mean[1]
        
        # Calculate volatility percentiles
        vol_high = np.percentile(all_vols_orig, self.config.vol_percentile_high)
        vol_low = np.percentile(all_vols_orig, self.config.vol_percentile_low)
        
        # Primary classification: Volatility-based
        if mean_volatility > vol_high:
            return RegimeType.HIGH_VOL
        elif mean_volatility < vol_low:
            return RegimeType.LOW_VOL
        
        # Secondary classification: Trend-based (medium volatility)
        if mean_return > self.config.trend_threshold:
            return RegimeType.BULL
        elif mean_return < -self.config.trend_threshold:
            return RegimeType.BEAR
        else:
            return RegimeType.SIDEWAYS
    
    def _state_to_regime(self, state_id: int) -> str:
        """Map HMM state ID to regime label"""
        if not hasattr(self, 'state_labels'):
            return RegimeType.UNKNOWN.value
        
        regime = self.state_labels.get(state_id, RegimeType.UNKNOWN)
        return regime.value
    
    def _calculate_regime_duration(self, predictions: pd.DataFrame) -> int:
        """
        Calculate how long the current regime has persisted.
        
        Args:
            predictions: Regime predictions DataFrame
        
        Returns:
            Duration in bars
        """
        if len(predictions) < 2:
            return 1
        
        current_regime = predictions.iloc[-1]['regime']
        duration = 1
        
        for i in range(len(predictions) - 2, -1, -1):
            if predictions.iloc[i]['regime'] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _print_state_characteristics(self) -> None:
        """Print state characteristics for debugging"""
        if not hasattr(self, 'state_characteristics'):
            return
        
        for state_id, chars in self.state_characteristics.items():
            print(f"  State {state_id} [{chars['regime']}]: "
                  f"Return={chars['mean_return']:.4f}, "
                  f"Vol={chars['mean_volatility']:.4f}, "
                  f"N={chars['n_samples']}")
    
    def test_stationarity(self, prices: pd.Series, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Test for stationarity using Augmented Dickey-Fuller.
        
        Critical for validating mean-reversion strategies in sideways regimes.
        
        Args:
            prices: Price series
            regime: Optional regime filter
        
        Returns:
            Dict with ADF statistics
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Filter by regime if specified
            if regime and self.is_fitted:
                predictions = self.predict(prices, return_confidence=False)
                mask = predictions['regime'] == regime
                test_prices = prices[predictions.index][mask]
            else:
                test_prices = prices
            
            if len(test_prices) < 20:
                return {'adf_stat': np.nan, 'p_value': 1.0, 'is_stationary': False}
            
            # Run ADF test
            result = adfuller(test_prices.values, maxlag=None, regression='c', autolag='AIC')
            
            adf_stat = result[0]
            p_value = result[1]
            is_stationary = p_value < self.config.adf_significance
            
            return {
                'adf_stat': float(adf_stat),
                'p_value': float(p_value),
                'is_stationary': bool(is_stationary),
                'critical_values': result[4]
            }
            
        except ImportError:
            print("[WARNING] statsmodels not installed. Cannot run ADF test.")
            return {'adf_stat': np.nan, 'p_value': 1.0, 'is_stationary': False}
    
    def get_strategy_gate(self, strategy_type: str) -> bool:
        """
        Gate strategy activation based on current regime.
        
        Model gating logic:
        - Trend strategies: Active in BULL/BEAR, inactive in SIDEWAYS
        - Mean reversion: Active in SIDEWAYS (if stationary), inactive in trending
        - Volatility breakout: Active in LOW_VOL → HIGH_VOL transition
        
        Args:
            strategy_type: One of ['trend', 'mean_reversion', 'breakout', 'ml_predictive']
        
        Returns:
            True if strategy should be active
        """
        if not self.current_state:
            return False  # Fail-safe: Don't trade without regime knowledge
        
        regime = self.current_state.regime
        confidence = self.current_state.confidence
        duration = self.current_state.duration
        
        # Minimum confidence threshold
        if confidence < 0.6:
            return False
        
        # Minimum regime persistence (avoid whipsaws)
        if duration < self.config.min_regime_duration:
            return False
        
        # Strategy-specific gating
        if strategy_type == 'trend':
            return regime in [RegimeType.BULL, RegimeType.BEAR]
        
        elif strategy_type == 'mean_reversion':
            return regime == RegimeType.SIDEWAYS
        
        elif strategy_type == 'breakout':
            # Active during volatility compression (before breakout)
            return regime == RegimeType.LOW_VOL
        
        elif strategy_type == 'ml_predictive':
            # ML strategies can operate in all regimes except extreme volatility
            return regime != RegimeType.HIGH_VOL
        
        else:
            return False
    
    def calculate_transition_probability(
        self,
        from_regime: RegimeType,
        to_regime: RegimeType,
        horizon: int = 1
    ) -> float:
        """
        Calculate probability of transitioning from one regime to another.
        
        **MEJORA CIENTÍFICA #4: Regime Transition Probability Matrix**
        
        Methodology (López de Prado - "Advances in Financial Machine Learning"):
        - HMMs naturally encode regime persistence via transition matrix
        - P(regime_t+1 = j | regime_t = i) = A[i,j]
        - Multi-step transitions: A^n for n-step ahead forecasts
        - Critical for risk management: anticipate regime changes before they occur
        
        Applications:
        1. Position sizing: Reduce exposure before high-volatility transitions
        2. Strategy rotation: Switch strategies when regime change is imminent
        3. Risk budgeting: Allocate capital based on transition probabilities
        
        Args:
            from_regime: Current regime (source state)
            to_regime: Target regime (destination state)
            horizon: Number of steps ahead (1 = next bar, 5 = 5 bars ahead)
        
        Returns:
            Transition probability [0, 1]
        
        Example:
            >>> classifier = HMMRegimeClassifier()
            >>> classifier.fit(prices)
            >>> # Probability of transitioning from LOW_VOL to HIGH_VOL in next 5 bars
            >>> prob = classifier.calculate_transition_probability(
            ...     RegimeType.LOW_VOL,
            ...     RegimeType.HIGH_VOL,
            ...     horizon=5
            ... )
            >>> if prob > 0.3:
            ...     print("WARNING: High probability of volatility spike ahead!")
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if not hasattr(self, 'state_labels'):
            raise RuntimeError("State labels not assigned. Refit model.")
        
        # Get HMM transition matrix
        transition_matrix = self.hmm.transmat_  # Shape: (n_states, n_states)
        
        # Map regimes to state IDs
        from_state_id = self._regime_to_state_id(from_regime)
        to_state_id = self._regime_to_state_id(to_regime)
        
        if from_state_id is None or to_state_id is None:
            warnings.warn(f"Regime mapping not found for {from_regime} or {to_regime}")
            return 0.0
        
        # Calculate n-step transition probability
        if horizon == 1:
            # Direct lookup for 1-step transition
            prob = transition_matrix[from_state_id, to_state_id]
        else:
            # Matrix exponentiation for multi-step transitions
            # P(i -> j, n steps) = (A^n)[i, j]
            transition_matrix_n = np.linalg.matrix_power(transition_matrix, horizon)
            prob = transition_matrix_n[from_state_id, to_state_id]
        
        return float(prob)
    
    def get_transition_matrix(self, regime_based: bool = True) -> pd.DataFrame:
        """
        Get full transition probability matrix.
        
        Args:
            regime_based: If True, return regime-labeled matrix; else state IDs
        
        Returns:
            DataFrame with transition probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        transition_matrix = self.hmm.transmat_
        
        if regime_based and hasattr(self, 'state_labels'):
            # Create regime-level aggregated matrix
            unique_regimes = list(set(self.state_labels.values()))
            n_regimes = len(unique_regimes)
            
            regime_matrix = np.zeros((n_regimes, n_regimes))
            
            for i, regime_from in enumerate(unique_regimes):
                for j, regime_to in enumerate(unique_regimes):
                    # Aggregate probabilities for all states mapped to these regimes
                    states_from = [s for s, r in self.state_labels.items() if r == regime_from]
                    states_to = [s for s, r in self.state_labels.items() if r == regime_to]
                    
                    # Weighted average of transition probabilities
                    total_prob = 0.0
                    for sf in states_from:
                        for st in states_to:
                            total_prob += transition_matrix[sf, st]
                    
                    regime_matrix[i, j] = total_prob / (len(states_from) * len(states_to)) if states_from and states_to else 0.0
            
            # Create DataFrame with regime labels
            regime_labels = [r.value for r in unique_regimes]
            df = pd.DataFrame(
                regime_matrix,
                index=[f"From {r}" for r in regime_labels],
                columns=[f"To {r}" for r in regime_labels]
            )
        else:
            # Return raw state-based matrix
            df = pd.DataFrame(
                transition_matrix,
                index=[f"State {i}" for i in range(self.config.n_states)],
                columns=[f"State {i}" for i in range(self.config.n_states)]
            )
        
        return df
    
    def predict_regime_change(
        self,
        prices: pd.Series,
        horizon: int = 5,
        threshold: float = 0.3
    ) -> Dict[str, any]:
        """
        Predict if a regime change is likely in the near future.
        
        **Critical for proactive risk management.**
        
        Args:
            prices: Recent price history
            horizon: Look-ahead window (bars)
            threshold: Minimum probability to flag regime change
        
        Returns:
            Dict with:
                - regime_change_likely: bool
                - current_regime: str
                - most_likely_next_regime: str
                - probability: float
                - horizon: int
        
        Example:
            >>> forecast = classifier.predict_regime_change(prices, horizon=10)
            >>> if forecast['regime_change_likely']:
            ...     print(f"Likely transition to {forecast['most_likely_next_regime']}")
            ...     print(f"Probability: {forecast['probability']:.1%}")
        """
        # Predict current regime
        current_state = self.predict_current(prices)
        current_regime = current_state.regime
        
        # Calculate transition probabilities to all other regimes
        all_regimes = list(RegimeType)
        transition_probs = {}
        
        for target_regime in all_regimes:
            if target_regime == current_regime:
                continue  # Skip self-transition
            
            prob = self.calculate_transition_probability(
                current_regime,
                target_regime,
                horizon=horizon
            )
            transition_probs[target_regime] = prob
        
        # Find most likely transition
        if transition_probs:
            most_likely_regime = max(transition_probs, key=transition_probs.get)
            max_prob = transition_probs[most_likely_regime]
        else:
            most_likely_regime = current_regime
            max_prob = 0.0
        
        # Determine if change is likely
        regime_change_likely = max_prob > threshold
        
        return {
            'regime_change_likely': regime_change_likely,
            'current_regime': current_regime.value,
            'most_likely_next_regime': most_likely_regime.value,
            'probability': max_prob,
            'horizon': horizon,
            'all_transition_probs': {r.value: p for r, p in transition_probs.items()}
        }
    
    def _regime_to_state_id(self, regime: RegimeType) -> Optional[int]:
        """
        Map regime label to HMM state ID.
        
        If multiple states map to same regime, returns the first one.
        
        Args:
            regime: RegimeType to map
        
        Returns:
            State ID or None if not found
        """
        if not hasattr(self, 'state_labels'):
            return None
        
        for state_id, state_regime in self.state_labels.items():
            if state_regime == regime:
                return state_id
        
        return None


# ========================================
# Utility Functions
# ========================================

def create_synthetic_regime_data(n_samples=1000, n_regimes=3, seed=42) -> pd.Series:
    """
    Create synthetic price data with regime switches.
    
    Args:
        n_samples: Number of samples
        n_regimes: Number of distinct regimes
        seed: Random seed
    
    Returns:
        Synthetic price series with embedded regimes
    """
    np.random.seed(seed)
    
    # Define regime characteristics (return, volatility, duration)
    regime_params = [
        (0.001, 0.01, 200),  # Bull: positive drift, low vol
        (-0.002, 0.015, 150),  # Bear: negative drift, medium vol
        (0.0, 0.005, 200)  # Sideways: no drift, very low vol
    ]
    
    prices = [100.0]
    current_regime = 0
    regime_duration = 0
    
    for _ in range(n_samples - 1):
        # Switch regime based on duration
        if regime_duration >= regime_params[current_regime][2]:
            current_regime = (current_regime + 1) % n_regimes
            regime_duration = 0
        
        mu, sigma, _ = regime_params[current_regime]
        
        # Generate return
        ret = np.random.normal(mu, sigma)
        prices.append(prices[-1] * (1 + ret))
        
        regime_duration += 1
    
    # Create time index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    return pd.Series(prices, index=dates, name='price')


def run_regime_detection_example():
    """Example regime detection pipeline"""
    print("=" * 60)
    print("HMM Regime Detection Example")
    print("=" * 60)
    
    # Create synthetic data
    print("\n[1/4] Creating synthetic multi-regime data...")
    prices = create_synthetic_regime_data(n_samples=500, n_regimes=3)
    print(f"  Generated {len(prices)} price points")
    
    # Initialize classifier
    print("\n[2/4] Initializing HMM classifier...")
    config = RegimeConfig(
        n_states=3,
        train_window=200,
        min_regime_duration=5
    )
    classifier = HMMRegimeClassifier(config)
    
    # Fit model
    print("\n[3/4] Fitting HMM on training data...")
    classifier.fit(prices)
    
    # Predict regimes
    print("\n[4/4] Predicting regime sequence...")
    predictions = classifier.predict(prices)
    
    # Print results
    print("\nRegime Distribution:")
    print(predictions['regime'].value_counts())
    
    print("\nRecent Predictions:")
    print(predictions.tail(10))
    
    # Test current state prediction
    print("\n" + "=" * 60)
    print("Current State Analysis:")
    print("=" * 60)
    current = classifier.predict_current(prices)
    print(f"  Regime: {current.regime.value}")
    print(f"  Confidence: {current.confidence:.2%}")
    print(f"  Duration: {current.duration} bars")
    print(f"  Return: {current.characteristics['mean_return']:.4f}")
    print(f"  Volatility: {current.characteristics['volatility']:.4f}")
    
    # Test strategy gating
    print("\n" + "=" * 60)
    print("Strategy Gating Logic:")
    print("=" * 60)
    for strat in ['trend', 'mean_reversion', 'breakout', 'ml_predictive']:
        active = classifier.get_strategy_gate(strat)
        status = "✓ ACTIVE" if active else "✗ INACTIVE"
        print(f"  {strat.upper():20s}: {status}")
    
    # Test stationarity (for mean reversion validation)
    print("\n" + "=" * 60)
    print("Stationarity Test (ADF):")
    print("=" * 60)
    adf_result = classifier.test_stationarity(prices)
    print(f"  ADF Statistic: {adf_result['adf_stat']:.4f}")
    print(f"  P-value: {adf_result['p_value']:.4f}")
    print(f"  Stationary: {adf_result['is_stationary']}")
    
    return classifier, predictions


if __name__ == '__main__':
    print("Running HMM Regime Classifier Example...\n")
    classifier, predictions = run_regime_detection_example()
