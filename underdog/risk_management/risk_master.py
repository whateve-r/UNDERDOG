"""
Risk Master Module - Portfolio-Level Risk Management
Implements daily/weekly DD limits, correlation tracking, and portfolio constraints.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd


@dataclass
class DrawdownLimits:
    """Drawdown limits configuration"""
    max_daily_dd_pct: float = 5.0  # Max daily DD %
    max_weekly_dd_pct: float = 10.0  # Max weekly DD %
    max_monthly_dd_pct: float = 15.0  # Max monthly DD %
    max_absolute_dd_pct: float = 20.0  # Max absolute DD from peak
    soft_limit_pct: float = 0.8  # Start scaling at 80% of limit


@dataclass
class ExposureLimits:
    """Position exposure limits"""
    max_total_exposure_pct: float = 100.0  # Max total portfolio exposure %
    max_per_symbol_pct: float = 10.0  # Max exposure per symbol %
    max_per_strategy_pct: float = 30.0  # Max exposure per strategy %
    max_correlated_exposure_pct: float = 40.0  # Max exposure to correlated assets %
    max_leverage: float = 2.0  # Max leverage ratio


@dataclass
class StrategyMetrics:
    """Per-strategy metrics tracking"""
    strategy_id: str
    pnl_history: deque = field(default_factory=lambda: deque(maxlen=252))  # 1 year rolling
    returns_history: deque = field(default_factory=lambda: deque(maxlen=252))
    current_exposure: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    trades_count: int = 0
    wins_count: int = 0
    
    def update_pnl(self, pnl: float, timestamp: datetime):
        """Update P&L and metrics"""
        self.pnl_history.append((timestamp, pnl))
        self.daily_pnl += pnl
        
        # Update equity
        self.current_equity += pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
    
    def get_sharpe_ratio(self, window: int = 21) -> float:
        """Calculate rolling Sharpe ratio"""
        if len(self.returns_history) < 2:
            return 0.0
        
        recent = list(self.returns_history)[-window:]
        returns = np.array([r for _, r in recent])
        
        if len(returns) < 2:
            return 0.0
        
        return np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    
    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if self.trades_count == 0:
            return 0.0
        return self.wins_count / self.trades_count


class RiskMaster:
    """
    Portfolio-level risk manager enforcing DD limits, correlations, and exposure constraints.
    """
    
    def __init__(self,
                 initial_capital: float,
                 dd_limits: Optional[DrawdownLimits] = None,
                 exposure_limits: Optional[ExposureLimits] = None,
                 correlation_window: int = 21):
        """
        Initialize Risk Master.
        
        Args:
            initial_capital: Starting account balance
            dd_limits: Drawdown limit configuration
            exposure_limits: Exposure limit configuration
            correlation_window: Rolling window for correlation calculation
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        self.dd_limits = dd_limits or DrawdownLimits()
        self.exposure_limits = exposure_limits or ExposureLimits()
        self.correlation_window = correlation_window
        
        # Strategy tracking
        self.strategies: Dict[str, StrategyMetrics] = {}
        
        # Portfolio state
        self.daily_start_capital = initial_capital
        self.weekly_start_capital = initial_capital
        self.monthly_start_capital = initial_capital
        
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        
        # Correlation matrix
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Trading state
        self.is_trading_halted = False
        self.halt_reason: Optional[str] = None
        
        # Audit trail
        self.rejection_log: List[Dict] = []
        
        # Time tracking
        self.last_daily_reset = datetime.now().date()
        self.last_weekly_reset = datetime.now().date()
        self.last_monthly_reset = datetime.now().date()
    
    def register_strategy(self, strategy_id: str) -> None:
        """Register a new strategy for tracking"""
        if strategy_id not in self.strategies:
            self.strategies[strategy_id] = StrategyMetrics(
                strategy_id=strategy_id,
                peak_equity=self.initial_capital,
                current_equity=self.initial_capital
            )
    
    def update_capital(self, new_capital: float) -> None:
        """Update current capital and peak"""
        self.current_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
    
    def update_daily_pnl(self, pnl: float, strategy_id: Optional[str] = None) -> None:
        """Update daily P&L for portfolio and strategy"""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
        
        if strategy_id and strategy_id in self.strategies:
            self.strategies[strategy_id].update_pnl(pnl, datetime.now())
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics (call at start of new trading day)"""
        self.daily_pnl = 0.0
        self.daily_start_capital = self.current_capital
        self.last_daily_reset = datetime.now().date()
        
        for strategy in self.strategies.values():
            strategy.daily_pnl = 0.0
    
    def reset_weekly_metrics(self) -> None:
        """Reset weekly metrics"""
        self.weekly_pnl = 0.0
        self.weekly_start_capital = self.current_capital
        self.last_weekly_reset = datetime.now().date()
        
        for strategy in self.strategies.values():
            strategy.weekly_pnl = 0.0
    
    def reset_monthly_metrics(self) -> None:
        """Reset monthly metrics"""
        self.monthly_pnl = 0.0
        self.monthly_start_capital = self.current_capital
        self.last_monthly_reset = datetime.now().date()
        
        for strategy in self.strategies.values():
            strategy.monthly_pnl = 0.0
    
    def get_current_drawdown_pct(self) -> float:
        """Calculate current drawdown from peak"""
        if self.peak_capital == 0:
            return 0.0
        return ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
    
    def get_daily_drawdown_pct(self) -> float:
        """Calculate daily drawdown"""
        if self.daily_start_capital == 0:
            return 0.0
        return (self.daily_pnl / self.daily_start_capital) * 100
    
    def get_weekly_drawdown_pct(self) -> float:
        """Calculate weekly drawdown"""
        if self.weekly_start_capital == 0:
            return 0.0
        return (self.weekly_pnl / self.weekly_start_capital) * 100
    
    def get_monthly_drawdown_pct(self) -> float:
        """Calculate monthly drawdown"""
        if self.monthly_start_capital == 0:
            return 0.0
        return (self.monthly_pnl / self.monthly_start_capital) * 100
    
    def check_drawdown_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check if any drawdown limits are breached.
        
        Returns:
            Tuple of (is_within_limits, violation_message)
        """
        daily_dd = abs(self.get_daily_drawdown_pct())
        weekly_dd = abs(self.get_weekly_drawdown_pct())
        monthly_dd = abs(self.get_monthly_drawdown_pct())
        total_dd = self.get_current_drawdown_pct()
        
        # Hard limit violations
        if daily_dd > self.dd_limits.max_daily_dd_pct:
            return False, f"Daily DD limit breached: {daily_dd:.2f}% > {self.dd_limits.max_daily_dd_pct}%"
        
        if weekly_dd > self.dd_limits.max_weekly_dd_pct:
            return False, f"Weekly DD limit breached: {weekly_dd:.2f}% > {self.dd_limits.max_weekly_dd_pct}%"
        
        if monthly_dd > self.dd_limits.max_monthly_dd_pct:
            return False, f"Monthly DD limit breached: {monthly_dd:.2f}% > {self.dd_limits.max_monthly_dd_pct}%"
        
        if total_dd > self.dd_limits.max_absolute_dd_pct:
            return False, f"Absolute DD limit breached: {total_dd:.2f}% > {self.dd_limits.max_absolute_dd_pct}%"
        
        return True, None
    
    def get_dd_scaling_factor(self) -> float:
        """
        Calculate position sizing scaling factor based on DD proximity to limits.
        Returns 1.0 if far from limits, scales down as approaching limits.
        """
        daily_dd = abs(self.get_daily_drawdown_pct())
        weekly_dd = abs(self.get_weekly_drawdown_pct())
        
        soft_daily = self.dd_limits.max_daily_dd_pct * self.dd_limits.soft_limit_pct
        soft_weekly = self.dd_limits.max_weekly_dd_pct * self.dd_limits.soft_limit_pct
        
        # Calculate scaling for each period
        daily_scale = 1.0
        if daily_dd > soft_daily:
            daily_scale = max(0.0, 1.0 - (daily_dd - soft_daily) / (self.dd_limits.max_daily_dd_pct - soft_daily))
        
        weekly_scale = 1.0
        if weekly_dd > soft_weekly:
            weekly_scale = max(0.0, 1.0 - (weekly_dd - soft_weekly) / (self.dd_limits.max_weekly_dd_pct - soft_weekly))
        
        # Use most conservative scaling
        return min(daily_scale, weekly_scale)
    
    def update_correlation_matrix(self) -> None:
        """
        Update correlation matrix based on strategy returns.
        Uses exponentially weighted correlation for recency bias.
        """
        if len(self.strategies) < 2:
            return
        
        # Collect returns for each strategy
        strategy_returns = {}
        min_length = float('inf')
        
        for sid, metrics in self.strategies.items():
            if len(metrics.returns_history) > 0:
                returns = [r for _, r in metrics.returns_history]
                strategy_returns[sid] = returns
                min_length = min(min_length, len(returns))
        
        if min_length < 2:
            return
        
        # Truncate to common length
        for sid in strategy_returns:
            strategy_returns[sid] = strategy_returns[sid][-min_length:]
        
        # Create DataFrame and calculate correlation
        df = pd.DataFrame(strategy_returns)
        self.correlation_matrix = df.ewm(span=self.correlation_window).corr()
    
    def get_correlated_exposure(self, strategy_id: str, threshold: float = 0.7) -> float:
        """
        Calculate total exposure to strategies correlated with given strategy.
        
        Args:
            strategy_id: Strategy to check
            threshold: Correlation threshold (default 0.7)
        
        Returns:
            Total correlated exposure as % of capital
        """
        if self.correlation_matrix is None or strategy_id not in self.strategies:
            return 0.0
        
        try:
            correlations = self.correlation_matrix.loc[strategy_id]
            correlated_strategies = correlations[correlations.abs() > threshold].index.tolist()
            
            total_exposure = sum(
                self.strategies[sid].current_exposure
                for sid in correlated_strategies
                if sid in self.strategies
            )
            
            return (total_exposure / self.current_capital) * 100
        except:
            return 0.0
    
    def check_exposure_limits(self, 
                            strategy_id: str,
                            symbol: str,
                            proposed_size: float,
                            proposed_value: float) -> Tuple[bool, Optional[str]]:
        """
        Check if proposed position violates exposure limits.
        
        Args:
            strategy_id: Strategy requesting position
            symbol: Trading symbol
            proposed_size: Proposed position size
            proposed_value: Proposed position value in account currency
        
        Returns:
            Tuple of (is_allowed, rejection_reason)
        """
        # Calculate exposure percentages
        proposed_exposure_pct = (proposed_value / self.current_capital) * 100
        
        # Total portfolio exposure
        total_exposure = sum(s.current_exposure for s in self.strategies.values()) + proposed_value
        total_exposure_pct = (total_exposure / self.current_capital) * 100
        
        if total_exposure_pct > self.exposure_limits.max_total_exposure_pct:
            return False, f"Total exposure limit: {total_exposure_pct:.1f}% > {self.exposure_limits.max_total_exposure_pct}%"
        
        # Per-symbol exposure
        # TODO: Track per-symbol exposure across strategies
        
        # Per-strategy exposure
        if strategy_id in self.strategies:
            strategy_exposure = self.strategies[strategy_id].current_exposure + proposed_value
            strategy_exposure_pct = (strategy_exposure / self.current_capital) * 100
            
            if strategy_exposure_pct > self.exposure_limits.max_per_strategy_pct:
                return False, f"Strategy exposure limit: {strategy_exposure_pct:.1f}% > {self.exposure_limits.max_per_strategy_pct}%"
        
        # Correlated exposure
        correlated_exposure_pct = self.get_correlated_exposure(strategy_id)
        if correlated_exposure_pct + proposed_exposure_pct > self.exposure_limits.max_correlated_exposure_pct:
            return False, f"Correlated exposure limit: {correlated_exposure_pct:.1f}% > {self.exposure_limits.max_correlated_exposure_pct}%"
        
        return True, None
    
    def pre_trade_check(self,
                       strategy_id: str,
                       symbol: str,
                       side: str,
                       size: float,
                       price: float,
                       stop_loss: Optional[float] = None) -> Tuple[bool, Optional[str], float]:
        """
        Comprehensive pre-trade risk check.
        
        Args:
            strategy_id: Strategy requesting trade
            symbol: Trading symbol
            side: "buy" or "sell"
            size: Position size
            price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Tuple of (is_approved, rejection_reason, adjusted_size)
        """
        # Check if trading is halted
        if self.is_trading_halted:
            return False, f"Trading halted: {self.halt_reason}", 0.0
        
        # Check drawdown limits
        dd_ok, dd_msg = self.check_drawdown_limits()
        if not dd_ok:
            self.halt_trading(dd_msg)
            self._log_rejection(strategy_id, symbol, size, dd_msg)
            return False, dd_msg, 0.0
        
        # Calculate position value
        position_value = size * price
        
        # Check exposure limits
        exposure_ok, exposure_msg = self.check_exposure_limits(
            strategy_id, symbol, size, position_value
        )
        if not exposure_ok:
            self._log_rejection(strategy_id, symbol, size, exposure_msg)
            return False, exposure_msg, 0.0
        
        # Apply DD scaling factor
        scaling_factor = self.get_dd_scaling_factor()
        adjusted_size = size * scaling_factor
        
        # Verify stop loss exists (required for risk management)
        if stop_loss is None:
            self._log_rejection(strategy_id, symbol, size, "Stop loss required")
            return False, "Stop loss required for all trades", 0.0
        
        # All checks passed
        return True, None, adjusted_size
    
    def halt_trading(self, reason: str) -> None:
        """Halt all trading"""
        self.is_trading_halted = True
        self.halt_reason = reason
        print(f"[RISK MASTER] TRADING HALTED: {reason}")
    
    def resume_trading(self) -> None:
        """Resume trading"""
        self.is_trading_halted = False
        self.halt_reason = None
        print("[RISK MASTER] Trading resumed")
    
    def _log_rejection(self, strategy_id: str, symbol: str, size: float, reason: str) -> None:
        """Log trade rejection for audit"""
        self.rejection_log.append({
            'timestamp': datetime.now(),
            'strategy_id': strategy_id,
            'symbol': symbol,
            'size': size,
            'reason': reason
        })
    
    def get_portfolio_metrics(self) -> Dict:
        """Get comprehensive portfolio metrics"""
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'total_dd_pct': self.get_current_drawdown_pct(),
            'daily_dd_pct': self.get_daily_drawdown_pct(),
            'weekly_dd_pct': self.get_weekly_drawdown_pct(),
            'monthly_dd_pct': self.get_monthly_drawdown_pct(),
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'monthly_pnl': self.monthly_pnl,
            'dd_scaling_factor': self.get_dd_scaling_factor(),
            'is_trading_halted': self.is_trading_halted,
            'halt_reason': self.halt_reason,
            'num_strategies': len(self.strategies),
            'rejections_count': len(self.rejection_log)
        }
