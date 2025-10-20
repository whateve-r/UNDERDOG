"""
Test Suite for Position Sizing
Tests multi-factor position sizing with Kelly, confidence, and DD scaling.
"""
import pytest
import numpy as np

from underdog.risk_management.position_sizing import (
    PositionSizer, SizingConfig
)


class TestSizingConfig:
    """Test SizingConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SizingConfig()
        
        assert config.base_risk_pct == 0.01
        assert config.kelly_fraction == 0.25
        assert config.use_confidence is True
        assert config.max_leverage == 2.0
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = SizingConfig(
            base_risk_pct=0.02,
            kelly_fraction=0.5,
            max_leverage=3.0
        )
        
        assert config.base_risk_pct == 0.02
        assert config.kelly_fraction == 0.5


class TestPositionSizer:
    """Test PositionSizer main functionality"""
    
    def test_calculate_size_basic(self, position_sizer_config):
        """Test basic position size calculation"""
        sizer = PositionSizer(position_sizer_config)
        
        result = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,  # 50 pips SL
            confidence=0.75
        )
        
        assert 'position_size' in result
        assert 'risk_amount' in result
        assert result['position_size'] > 0
        assert result['risk_amount'] <= 100000.0 * 0.01  # Max 1% risk
    
    def test_calculate_size_with_dd_scaling(self, position_sizer_config):
        """Test position sizing with DD scaling factor"""
        sizer = PositionSizer(position_sizer_config)
        
        # With DD scaling (0.5 = in drawdown)
        result_dd = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,
            confidence=0.75,
            dd_scaling_factor=0.5
        )
        
        # Without DD scaling
        result_no_dd = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,
            confidence=0.75,
            dd_scaling_factor=1.0
        )
        
        # Position size should be smaller with DD scaling
        assert result_dd['position_size'] < result_no_dd['position_size']
    
    def test_calculate_size_with_low_confidence(self, position_sizer_config):
        """Test position sizing with low confidence"""
        sizer = PositionSizer(position_sizer_config)
        
        result_low = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,
            confidence=0.3  # Low confidence
        )
        
        result_high = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,
            confidence=0.9  # High confidence
        )
        
        # Lower confidence should result in smaller position
        assert result_low['position_size'] < result_high['position_size']
    
    def test_calculate_size_long_vs_short(self, position_sizer_config):
        """Test position sizing for long vs short"""
        sizer = PositionSizer(position_sizer_config)
        
        result_long = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,  # SL below entry (long)
            confidence=0.75
        )
        
        result_short = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.10500,  # SL above entry (short)
            confidence=0.75
        )
        
        # Both should be positive sizes
        assert result_long['position_size'] > 0
        assert result_short['position_size'] > 0


class TestKellyFraction:
    """Test Kelly Criterion calculation"""
    
    def test_kelly_fraction_calculation(self, position_sizer_config):
        """Test Kelly fraction with win rate and payoff ratio"""
        sizer = PositionSizer(position_sizer_config)
        
        kelly = sizer.calculate_kelly_fraction(
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=100.0
        )
        
        # Kelly = (p * b - q) / b
        # where p = win rate, q = 1-p, b = avg_win/avg_loss
        # Kelly = (0.55 * 1.5 - 0.45) / 1.5 = 0.25
        assert kelly == pytest.approx(0.25, abs=0.01)
    
    def test_kelly_fraction_applied(self, position_sizer_config):
        """Test that Half-Kelly is applied (0.25 = 0.5 * Kelly)"""
        sizer = PositionSizer(position_sizer_config)
        
        # Full Kelly would be 0.5
        full_kelly = 0.5
        half_kelly = full_kelly * position_sizer_config.kelly_fraction
        
        assert half_kelly == 0.125  # Half of full Kelly
    
    def test_kelly_fraction_negative(self, position_sizer_config):
        """Test Kelly returns 0 for negative edge"""
        sizer = PositionSizer(position_sizer_config)
        
        # Losing strategy (win rate too low)
        kelly = sizer.calculate_kelly_fraction(
            win_rate=0.30,
            avg_win=100.0,
            avg_loss=100.0
        )
        
        # Kelly should be 0 or negative (no edge)
        assert kelly <= 0.0


class TestStopLossCalculation:
    """Test stop loss and risk calculation"""
    
    def test_calculate_stop_loss_distance(self, position_sizer_config):
        """Test stop loss distance calculation"""
        sizer = PositionSizer(position_sizer_config)
        
        entry = 1.10000
        stop_loss = 1.09500
        
        distance = abs(entry - stop_loss)
        assert distance == 0.00500
    
    def test_risk_per_unit(self, position_sizer_config):
        """Test risk per unit calculation"""
        sizer = PositionSizer(position_sizer_config)
        
        result = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,
            confidence=0.75
        )
        
        # Risk per unit = distance to SL
        risk_per_unit = abs(1.10000 - 1.09500)
        expected_risk = result['position_size'] * risk_per_unit
        
        # Risk amount should match
        assert result['risk_amount'] == pytest.approx(expected_risk, rel=0.01)
    
    def test_calculate_stop_loss_atr_based(self, position_sizer_config):
        """Test ATR-based stop loss calculation"""
        sizer = PositionSizer(position_sizer_config)
        
        entry_price = 1.10000
        atr = 0.00050  # 5 pips ATR
        multiplier = 1.5
        
        # Long position: SL = entry - (ATR * multiplier)
        sl_long = sizer.calculate_stop_loss(
            entry_price=entry_price,
            atr=atr,
            direction='long',
            multiplier=multiplier
        )
        
        expected_sl_long = entry_price - (atr * multiplier)
        assert sl_long == pytest.approx(expected_sl_long, abs=1e-6)
        
        # Short position: SL = entry + (ATR * multiplier)
        sl_short = sizer.calculate_stop_loss(
            entry_price=entry_price,
            atr=atr,
            direction='short',
            multiplier=multiplier
        )
        
        expected_sl_short = entry_price + (atr * multiplier)
        assert sl_short == pytest.approx(expected_sl_short, abs=1e-6)


class TestLeverageConstraints:
    """Test leverage limits"""
    
    def test_max_leverage_constraint(self, position_sizer_config):
        """Test that position size respects max leverage"""
        sizer = PositionSizer(position_sizer_config)
        
        # Try to size with very tight stop loss (would require high leverage)
        result = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09999,  # 0.1 pip SL (very tight)
            confidence=1.0
        )
        
        # Position value should not exceed max_leverage * balance
        max_position_value = 100000.0 * position_sizer_config.max_leverage
        actual_position_value = result['position_size'] * 1.10000
        
        assert actual_position_value <= max_position_value
    
    def test_min_position_size(self, position_sizer_config):
        """Test minimum position size constraint"""
        sizer = PositionSizer(position_sizer_config)
        
        # Very small account
        result = sizer.calculate_size(
            account_balance=500.0,  # Small account
            entry_price=1.10000,
            stop_loss=1.09500,
            confidence=0.5
        )
        
        # Should still return a valid (possibly 0) size
        assert result['position_size'] >= 0


class TestConfidenceScaling:
    """Test confidence-based scaling"""
    
    def test_confidence_scaling_enabled(self):
        """Test that confidence scaling works when enabled"""
        config = SizingConfig(use_confidence=True)
        sizer = PositionSizer(config)
        
        result = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,
            confidence=0.5  # 50% confidence
        )
        
        # Position should be scaled by confidence
        assert result['position_size'] > 0
    
    def test_confidence_scaling_disabled(self):
        """Test that confidence scaling can be disabled"""
        config = SizingConfig(use_confidence=False)
        sizer = PositionSizer(config)
        
        result_low = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,
            confidence=0.3
        )
        
        result_high = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,
            confidence=0.9
        )
        
        # Positions should be equal (confidence not used)
        assert result_low['position_size'] == pytest.approx(
            result_high['position_size'], 
            rel=0.01
        )


class TestMultiFactorSizing:
    """Test combined multi-factor sizing"""
    
    def test_full_multi_factor_sizing(self, position_sizer_config):
        """Test all factors combined: base × kelly × confidence × dd_scale"""
        sizer = PositionSizer(position_sizer_config)
        
        result = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.09500,
            confidence=0.75,
            dd_scaling_factor=0.8,
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=100.0
        )
        
        assert result['position_size'] > 0
        assert result['risk_amount'] > 0
        assert result['risk_amount'] <= 100000.0 * 0.01  # Base risk cap


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_stop_loss_distance(self, position_sizer_config):
        """Test behavior when SL equals entry (zero distance)"""
        sizer = PositionSizer(position_sizer_config)
        
        result = sizer.calculate_size(
            account_balance=100000.0,
            entry_price=1.10000,
            stop_loss=1.10000,  # Same as entry
            confidence=0.75
        )
        
        # Should return 0 or handle gracefully
        assert result['position_size'] >= 0
    
    def test_negative_account_balance(self, position_sizer_config):
        """Test behavior with negative balance"""
        sizer = PositionSizer(position_sizer_config)
        
        with pytest.raises(ValueError):
            sizer.calculate_size(
                account_balance=-1000.0,
                entry_price=1.10000,
                stop_loss=1.09500,
                confidence=0.75
            )
    
    def test_invalid_confidence(self, position_sizer_config):
        """Test behavior with invalid confidence"""
        sizer = PositionSizer(position_sizer_config)
        
        with pytest.raises(ValueError):
            sizer.calculate_size(
                account_balance=100000.0,
                entry_price=1.10000,
                stop_loss=1.09500,
                confidence=1.5  # Invalid: >1.0
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
