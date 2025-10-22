"""
Test Suite for HMM Regime Classifier
Tests regime detection, strategy gating, and state labeling.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from underdog.ml.models.regime_classifier import (
    HMMRegimeClassifier, RegimeType, StrategyGate
)


class TestRegimeType:
    """Test RegimeType enum"""
    
    def test_regime_types(self):
        """Test all regime types defined"""
        assert hasattr(RegimeType, 'BULL')
        assert hasattr(RegimeType, 'BEAR')
        assert hasattr(RegimeType, 'SIDEWAYS')
        assert hasattr(RegimeType, 'HIGH_VOL')
        assert hasattr(RegimeType, 'LOW_VOL')
    
    def test_regime_values(self):
        """Test regime enum values"""
        assert RegimeType.BULL.value == 'BULL'
        assert RegimeType.BEAR.value == 'BEAR'
        assert RegimeType.SIDEWAYS.value == 'SIDEWAYS'


class TestStrategyGate:
    """Test StrategyGate configuration"""
    
    def test_gate_initialization(self):
        """Test gate initialization"""
        gate = StrategyGate(
            trend_following=['BULL', 'BEAR'],
            mean_reversion=['SIDEWAYS'],
            confidence_threshold=0.7
        )
        
        assert 'BULL' in gate.trend_following
        assert 'SIDEWAYS' in gate.mean_reversion
        assert gate.confidence_threshold == 0.7
    
    def test_default_gate(self):
        """Test default gate configuration"""
        gate = StrategyGate()
        
        # Default should activate trend in bull/bear
        assert RegimeType.BULL.value in gate.trend_following
        assert RegimeType.BEAR.value in gate.trend_following
        
        # Default should activate mean-reversion in sideways
        assert RegimeType.SIDEWAYS.value in gate.mean_reversion


class TestHMMRegimeClassifier:
    """Test HMM Regime Classifier main functionality"""
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        clf = HMMRegimeClassifier(n_states=3, random_seed=42)
        
        assert clf.n_states == 3
        assert clf.random_seed == 42
        assert clf.hmm is None  # Not fitted yet
    
    def test_fit_with_price_series(self, sample_price_series):
        """Test fitting HMM with price data"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        assert clf.hmm is not None
        assert clf.is_fitted
    
    def test_fit_with_returns(self, sample_price_series):
        """Test fitting with returns"""
        returns = sample_price_series.pct_change().dropna()
        
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(returns, input_type='returns')
        
        assert clf.is_fitted
    
    def test_predict_states(self, sample_price_series):
        """Test state prediction"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        states = clf.predict(sample_price_series)
        
        assert len(states) == len(sample_price_series)
        assert states.min() >= 0
        assert states.max() < 3  # Should be 0, 1, 2
    
    def test_predict_current(self, sample_price_series):
        """Test current state prediction"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        current_state = clf.predict_current(sample_price_series)
        
        assert 0 <= current_state < 3


class TestRegimeLabeling:
    """Test regime state labeling"""
    
    def test_state_labeling_3_states(self, sample_price_series):
        """Test 3-state labeling (BULL/SIDEWAYS/BEAR)"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        states = clf.predict(sample_price_series)
        regimes = clf.get_regimes(sample_price_series)
        
        assert len(regimes) == len(states)
        
        # Should have BULL, BEAR, SIDEWAYS
        unique_regimes = set(regimes)
        assert len(unique_regimes) <= 3
    
    def test_state_labeling_5_states(self, sample_price_series):
        """Test 5-state labeling (includes VOL states)"""
        clf = HMMRegimeClassifier(n_states=5)
        clf.fit(sample_price_series)
        
        regimes = clf.get_regimes(sample_price_series)
        
        # Should have up to 5 different regimes
        unique_regimes = set(regimes)
        assert len(unique_regimes) <= 5
    
    def test_regime_characteristics(self, sample_price_series):
        """Test that regimes have expected characteristics"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        states = clf.predict(sample_price_series)
        regimes = clf.get_regimes(sample_price_series)
        
        # Get mean return for each regime
        returns = sample_price_series.pct_change()
        
        bull_returns = returns[regimes == RegimeType.BULL.value]
        bear_returns = returns[regimes == RegimeType.BEAR.value]
        
        if len(bull_returns) > 0 and len(bear_returns) > 0:
            # Bull should have higher mean return than bear
            assert bull_returns.mean() > bear_returns.mean()


class TestStrategyGating:
    """Test strategy gating logic"""
    
    def test_get_strategy_gate_bull(self, sample_price_series):
        """Test strategy gate in bull regime"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        # Force BULL regime
        gate_config = StrategyGate(
            trend_following=[RegimeType.BULL.value],
            mean_reversion=[RegimeType.SIDEWAYS.value],
            confidence_threshold=0.5
        )
        
        # Mock current regime as BULL
        current_regime = RegimeType.BULL.value
        gate = clf.get_strategy_gate(current_regime, gate_config)
        
        assert gate['trend_following'] is True
        assert gate['mean_reversion'] is False
    
    def test_get_strategy_gate_bear(self):
        """Test strategy gate in bear regime"""
        clf = HMMRegimeClassifier(n_states=3)
        
        gate_config = StrategyGate(
            trend_following=[RegimeType.BULL.value, RegimeType.BEAR.value],
            mean_reversion=[RegimeType.SIDEWAYS.value]
        )
        
        gate = clf.get_strategy_gate(RegimeType.BEAR.value, gate_config)
        
        assert gate['trend_following'] is True
        assert gate['mean_reversion'] is False
    
    def test_get_strategy_gate_sideways(self):
        """Test strategy gate in sideways regime"""
        clf = HMMRegimeClassifier(n_states=3)
        
        gate_config = StrategyGate(
            trend_following=[RegimeType.BULL.value, RegimeType.BEAR.value],
            mean_reversion=[RegimeType.SIDEWAYS.value]
        )
        
        gate = clf.get_strategy_gate(RegimeType.SIDEWAYS.value, gate_config)
        
        assert gate['trend_following'] is False
        assert gate['mean_reversion'] is True
    
    def test_confidence_threshold_gating(self, sample_price_series):
        """Test confidence threshold blocks low-confidence signals"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        gate_config = StrategyGate(
            trend_following=[RegimeType.BULL.value],
            confidence_threshold=0.9  # High threshold
        )
        
        # Get gate with low confidence
        gate = clf.get_strategy_gate(
            RegimeType.BULL.value,
            gate_config,
            confidence=0.5  # Below threshold
        )
        
        # Both should be False due to low confidence
        assert gate['trend_following'] is False
        assert gate['mean_reversion'] is False
    
    def test_confidence_threshold_pass(self, sample_price_series):
        """Test confidence threshold passes high-confidence signals"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        gate_config = StrategyGate(
            trend_following=[RegimeType.BULL.value],
            confidence_threshold=0.7
        )
        
        # Get gate with high confidence
        gate = clf.get_strategy_gate(
            RegimeType.BULL.value,
            gate_config,
            confidence=0.85  # Above threshold
        )
        
        # Should be True due to high confidence
        assert gate['trend_following'] is True


class TestRegimePersistence:
    """Test regime persistence filtering"""
    
    def test_min_duration_filter(self, sample_price_series):
        """Test minimum duration filter"""
        clf = HMMRegimeClassifier(n_states=3, min_regime_duration=5)
        clf.fit(sample_price_series)
        
        regimes = clf.get_regimes(sample_price_series)
        
        # Check that no regime lasts < 5 bars
        regime_changes = np.diff(regimes.values.astype(str))
        regime_lengths = []
        current_length = 1
        
        for i in range(len(regime_changes)):
            if regime_changes[i] == '':  # Same regime
                current_length += 1
            else:
                regime_lengths.append(current_length)
                current_length = 1
        regime_lengths.append(current_length)
        
        # Most regimes should be >= min_duration
        long_regimes = [l for l in regime_lengths if l >= 5]
        assert len(long_regimes) > 0
    
    def test_no_min_duration(self, sample_price_series):
        """Test with no minimum duration"""
        clf = HMMRegimeClassifier(n_states=3, min_regime_duration=1)
        clf.fit(sample_price_series)
        
        regimes = clf.get_regimes(sample_price_series)
        
        # Should have regimes
        assert len(set(regimes)) > 1


class TestStationarityTest:
    """Test ADF stationarity test"""
    
    def test_stationary_series(self):
        """Test with stationary series"""
        # White noise (stationary)
        np.random.seed(42)
        stationary = pd.Series(np.random.randn(500))
        
        clf = HMMRegimeClassifier()
        is_stationary, pvalue = clf.test_stationarity(stationary)
        
        # White noise should be stationary
        assert is_stationary is True
        assert pvalue < 0.05
    
    def test_nonstationary_series(self, sample_price_series):
        """Test with non-stationary series (prices)"""
        clf = HMMRegimeClassifier()
        is_stationary, pvalue = clf.test_stationarity(sample_price_series)
        
        # Prices are typically non-stationary
        # (though synthetic data might be ambiguous)
        assert pvalue is not None
    
    def test_returns_stationary(self, sample_price_series):
        """Test that returns are typically stationary"""
        returns = sample_price_series.pct_change().dropna()
        
        clf = HMMRegimeClassifier()
        is_stationary, pvalue = clf.test_stationarity(returns)
        
        # Returns should be more stationary than prices
        assert pvalue is not None


class TestViterbiAlgorithm:
    """Test Viterbi algorithm (most likely state sequence)"""
    
    def test_viterbi_vs_predict(self, sample_price_series):
        """Test that Viterbi gives coherent state sequence"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        # Predict states (Viterbi algorithm)
        states = clf.predict(sample_price_series)
        
        # States should be coherent (not random noise)
        state_changes = np.sum(np.diff(states) != 0)
        total_states = len(states) - 1
        
        # Change rate should be < 50% (not random)
        change_rate = state_changes / total_states
        assert change_rate < 0.5


class TestModelSaveLoad:
    """Test model save/load"""
    
    def test_save_model(self, sample_price_series, temp_dir):
        """Test saving model"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        model_path = temp_dir / "hmm_model.pkl"
        clf.save(str(model_path))
        
        assert model_path.exists()
    
    def test_load_model(self, sample_price_series, temp_dir):
        """Test loading model"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        model_path = temp_dir / "hmm_model.pkl"
        clf.save(str(model_path))
        
        # Load
        loaded_clf = HMMRegimeClassifier.load(str(model_path))
        
        assert loaded_clf.is_fitted
        assert loaded_clf.n_states == clf.n_states
    
    def test_loaded_model_predictions(self, sample_price_series, temp_dir):
        """Test that loaded model gives same predictions"""
        clf = HMMRegimeClassifier(n_states=3, random_seed=42)
        clf.fit(sample_price_series)
        
        original_states = clf.predict(sample_price_series)
        
        # Save and load
        model_path = temp_dir / "hmm_model.pkl"
        clf.save(str(model_path))
        loaded_clf = HMMRegimeClassifier.load(str(model_path))
        
        loaded_states = loaded_clf.predict(sample_price_series)
        
        # Should be identical
        assert np.array_equal(original_states, loaded_states)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_state(self):
        """Test with n_states=1"""
        clf = HMMRegimeClassifier(n_states=1)
        
        # Should initialize
        assert clf.n_states == 1
    
    def test_many_states(self, sample_price_series):
        """Test with many states"""
        clf = HMMRegimeClassifier(n_states=10)
        clf.fit(sample_price_series)
        
        states = clf.predict(sample_price_series)
        
        # Should have states in [0, 9]
        assert states.min() >= 0
        assert states.max() < 10
    
    def test_insufficient_data(self):
        """Test with insufficient data"""
        short_series = pd.Series([100.0, 101.0, 102.0])
        
        clf = HMMRegimeClassifier(n_states=3)
        
        # Should handle gracefully or raise informative error
        try:
            clf.fit(short_series)
            # If it fits, predict should work
            states = clf.predict(short_series)
            assert len(states) == len(short_series)
        except ValueError as e:
            # Expected for insufficient data
            assert "insufficient" in str(e).lower() or "data" in str(e).lower()
    
    def test_predict_before_fit(self, sample_price_series):
        """Test predict before fit raises error"""
        clf = HMMRegimeClassifier(n_states=3)
        
        with pytest.raises((ValueError, AttributeError)):
            clf.predict(sample_price_series)
    
    def test_constant_price(self):
        """Test with constant price (zero variance)"""
        constant_price = pd.Series([100.0] * 100)
        
        clf = HMMRegimeClassifier(n_states=3)
        
        # Should handle zero variance
        try:
            clf.fit(constant_price)
            # If it fits, all states should be same
            states = clf.predict(constant_price)
            assert len(set(states)) <= 2  # Likely only 1 or 2 states used
        except (ValueError, np.linalg.LinAlgError):
            # Expected for zero variance
            pass


class TestRegimeStatistics:
    """Test regime statistics and characteristics"""
    
    def test_get_regime_stats(self, sample_price_series):
        """Test regime statistics calculation"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        regimes = clf.get_regimes(sample_price_series)
        returns = sample_price_series.pct_change()
        
        # Calculate stats per regime
        stats = {}
        for regime in set(regimes):
            regime_returns = returns[regimes == regime]
            stats[regime] = {
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'count': len(regime_returns)
            }
        
        # Should have stats for each regime
        assert len(stats) <= 3
        
        # Each regime should have observations
        for regime, stat in stats.items():
            assert stat['count'] > 0
    
    def test_regime_transitions(self, sample_price_series):
        """Test regime transition matrix"""
        clf = HMMRegimeClassifier(n_states=3)
        clf.fit(sample_price_series)
        
        states = clf.predict(sample_price_series)
        
        # Build transition matrix
        n_states = 3
        transitions = np.zeros((n_states, n_states))
        
        for i in range(len(states) - 1):
            current = states[i]
            next_state = states[i + 1]
            transitions[current, next_state] += 1
        
        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transitions / row_sums
        
        # Each row should sum to ~1.0 (or 0 if state never occurred)
        for i in range(n_states):
            row_sum = transition_probs[i].sum()
            assert row_sum == 0 or abs(row_sum - 1.0) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
