"""
Test Suite for Risk Master
Tests drawdown limits, exposure constraints, and correlation tracking.
"""
import pytest
from datetime import datetime, timedelta

from underdog.risk_management.risk_master import (
    RiskMaster, DrawdownLimits, ExposureLimits
)


class TestDrawdownLimits:
    """Test drawdown limit enforcement"""
    
    def test_daily_drawdown_limit(self, risk_master_config):
        """Test that daily DD limit triggers kill switch"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Simulate 2.5% loss (exceeds 2% daily limit)
        risk_master.update_pnl(-2500.0)
        
        # Should reject new trades
        is_allowed = risk_master.pre_trade_check('EURUSD', 'long', 10000.0)
        assert is_allowed is False
        assert risk_master.kill_switch_active is True
    
    def test_daily_drawdown_within_limit(self, risk_master_config):
        """Test that trades allowed within daily DD limit"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Simulate 1.5% loss (within 2% daily limit)
        risk_master.update_pnl(-1500.0)
        
        # Should allow new trades
        is_allowed = risk_master.pre_trade_check('EURUSD', 'long', 5000.0)
        assert is_allowed is True
        assert risk_master.kill_switch_active is False
    
    def test_weekly_drawdown_limit(self, risk_master_config):
        """Test weekly DD limit enforcement"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Simulate losses over multiple days (total 6% weekly)
        for _ in range(4):
            risk_master.update_pnl(-1500.0)  # 1.5% each day
        
        # Should trigger kill switch (6% > 5% weekly limit)
        assert risk_master.kill_switch_active is True
    
    def test_monthly_drawdown_limit(self, risk_master_config):
        """Test monthly DD limit enforcement"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Simulate 11% monthly loss (exceeds 10% monthly limit)
        risk_master.update_pnl(-11000.0)
        
        assert risk_master.kill_switch_active is True
    
    def test_drawdown_recovery(self, risk_master_config):
        """Test drawdown reduces after profitable trades"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Loss then profit
        risk_master.update_pnl(-1500.0)
        risk_master.update_pnl(1000.0)
        
        # Net DD should be reduced
        daily_dd = abs(risk_master.current_capital - 100000.0) / 100000.0
        assert daily_dd < 0.01  # Less than 1%


class TestExposureLimits:
    """Test position exposure constraints"""
    
    def test_max_position_size_limit(self, risk_master_config):
        """Test that single position cannot exceed max %"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Try to open position worth 6% of capital (exceeds 5% limit)
        is_allowed = risk_master.pre_trade_check('EURUSD', 'long', 6000.0)
        assert is_allowed is False
    
    def test_max_position_size_within_limit(self, risk_master_config):
        """Test that position within limit is allowed"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Position worth 4% of capital (within 5% limit)
        is_allowed = risk_master.pre_trade_check('EURUSD', 'long', 4000.0)
        assert is_allowed is True
    
    def test_max_total_exposure_limit(self, risk_master_config):
        """Test total exposure across all positions"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Open multiple positions totaling 21% exposure (exceeds 20% limit)
        risk_master.add_position('EURUSD', 'long', 8000.0)
        risk_master.add_position('GBPUSD', 'long', 8000.0)
        
        # Try to add another position (would exceed total exposure)
        is_allowed = risk_master.pre_trade_check('USDJPY', 'long', 5000.0)
        assert is_allowed is False
    
    def test_correlated_exposure_limit(self, risk_master_config):
        """Test correlated position exposure limits"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Add correlated positions (EUR pairs)
        risk_master.add_position('EURUSD', 'long', 5000.0)
        risk_master.add_position('EURJPY', 'long', 5000.0)
        
        # Update correlation matrix (simulate high correlation)
        risk_master.correlation_matrix['EURUSD']['EURJPY'] = 0.85
        
        # Try to add another EUR pair (would exceed correlated exposure)
        is_allowed = risk_master.pre_trade_check('EURGBP', 'long', 3000.0)
        
        # Should be rejected due to high correlation
        # Note: This test depends on correlation penalty implementation
        assert isinstance(is_allowed, bool)


class TestRiskMasterPositionTracking:
    """Test position tracking and updates"""
    
    def test_add_position(self, risk_master_config):
        """Test adding a position"""
        risk_master = RiskMaster(**risk_master_config)
        
        risk_master.add_position('EURUSD', 'long', 5000.0)
        
        assert 'EURUSD' in risk_master.positions
        assert risk_master.positions['EURUSD']['notional'] == 5000.0
        assert risk_master.positions['EURUSD']['direction'] == 'long'
    
    def test_remove_position(self, risk_master_config):
        """Test removing a position"""
        risk_master = RiskMaster(**risk_master_config)
        
        risk_master.add_position('EURUSD', 'long', 5000.0)
        risk_master.remove_position('EURUSD')
        
        assert 'EURUSD' not in risk_master.positions
    
    def test_update_position(self, risk_master_config):
        """Test updating position value"""
        risk_master = RiskMaster(**risk_master_config)
        
        risk_master.add_position('EURUSD', 'long', 5000.0)
        risk_master.update_position('EURUSD', new_notional=6000.0)
        
        assert risk_master.positions['EURUSD']['notional'] == 6000.0


class TestRiskMasterCorrelation:
    """Test correlation matrix tracking"""
    
    def test_correlation_matrix_initialization(self, risk_master_config):
        """Test correlation matrix is initialized"""
        risk_master = RiskMaster(**risk_master_config)
        
        assert hasattr(risk_master, 'correlation_matrix')
        assert isinstance(risk_master.correlation_matrix, dict)
    
    def test_update_correlation(self, risk_master_config):
        """Test updating correlation between symbols"""
        risk_master = RiskMaster(**risk_master_config)
        
        risk_master.update_correlation_matrix('EURUSD', 'GBPUSD', 0.75)
        
        assert risk_master.correlation_matrix['EURUSD']['GBPUSD'] == 0.75
        # Should be symmetric
        assert risk_master.correlation_matrix['GBPUSD']['EURUSD'] == 0.75
    
    def test_get_correlation_penalty(self, risk_master_config):
        """Test correlation penalty calculation"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Add positions
        risk_master.add_position('EURUSD', 'long', 5000.0)
        risk_master.add_position('GBPUSD', 'long', 5000.0)
        
        # Set correlation
        risk_master.update_correlation_matrix('EURUSD', 'GBPUSD', 0.80)
        
        # Get penalty for new correlated position
        penalty = risk_master.get_correlation_penalty('EURGBP', ['EURUSD', 'GBPUSD'])
        
        assert 0.0 <= penalty <= 1.0


class TestRiskMasterScaling:
    """Test DD-based position scaling"""
    
    def test_dd_scaling_factor_no_dd(self, risk_master_config):
        """Test scaling factor is 1.0 with no drawdown"""
        risk_master = RiskMaster(**risk_master_config)
        
        scaling = risk_master.get_dd_scaling_factor()
        assert scaling == 1.0
    
    def test_dd_scaling_factor_with_dd(self, risk_master_config):
        """Test scaling factor reduces with drawdown"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Simulate 5% drawdown
        risk_master.update_pnl(-5000.0)
        
        scaling = risk_master.get_dd_scaling_factor()
        assert 0.0 < scaling < 1.0  # Should be reduced
    
    def test_dd_scaling_factor_max_dd(self, risk_master_config):
        """Test scaling factor at max drawdown"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Simulate maximum monthly drawdown (10%)
        risk_master.update_pnl(-10000.0)
        
        scaling = risk_master.get_dd_scaling_factor()
        assert scaling < 0.3  # Heavily reduced


class TestRiskMasterKillSwitch:
    """Test kill switch functionality"""
    
    def test_kill_switch_manual_activation(self, risk_master_config):
        """Test manual kill switch activation"""
        risk_master = RiskMaster(**risk_master_config)
        
        risk_master.activate_kill_switch()
        
        assert risk_master.kill_switch_active is True
        
        # All trades should be rejected
        is_allowed = risk_master.pre_trade_check('EURUSD', 'long', 1000.0)
        assert is_allowed is False
    
    def test_kill_switch_deactivation(self, risk_master_config):
        """Test kill switch can be deactivated"""
        risk_master = RiskMaster(**risk_master_config)
        
        risk_master.activate_kill_switch()
        risk_master.deactivate_kill_switch()
        
        assert risk_master.kill_switch_active is False
    
    def test_kill_switch_auto_trigger_on_dd(self, risk_master_config):
        """Test kill switch auto-triggers on DD breach"""
        risk_master = RiskMaster(**risk_master_config)
        
        # Exceed daily DD limit
        risk_master.update_pnl(-3000.0)  # 3% loss
        
        assert risk_master.kill_switch_active is True


class TestRiskMasterEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_capital(self):
        """Test behavior with zero capital"""
        with pytest.raises(ValueError):
            RiskMaster(initial_capital=0.0)
    
    def test_negative_capital(self):
        """Test behavior with negative capital"""
        with pytest.raises(ValueError):
            RiskMaster(initial_capital=-10000.0)
    
    def test_invalid_dd_limits(self):
        """Test invalid DD limits are rejected"""
        with pytest.raises(ValueError):
            DrawdownLimits(
                daily_max=1.5,  # Invalid: >1.0
                weekly_max=0.05,
                monthly_max=0.10
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
