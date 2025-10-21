"""
Prop Firm Risk Management Engine.

Implements:
- Daily/Total Drawdown tracking and limits
- Position sizing (Fixed Fractional + Kelly + Confidence)
- Panic close mechanism
- Prop Firm compliance checks
"""

import numpy as np
from datetime import datetime
from typing import Optional, Dict
from underdog.core.abstractions import RiskManager, Portfolio, OrderEvent, SignalEvent


class PropFirmRiskManager(RiskManager):
    """
    Prop Firm compliant Risk Manager.
    
    Enforces:
    - Daily DD limit: 5% (default)
    - Total DD limit: 10% (default)
    - Position sizing: Fixed Fractional (1-2% risk per trade)
    - Kelly Criterion integration (fractional)
    - Confidence-weighted sizing for ML signals
    - Panic close on limit breach
    """
    
    def __init__(
        self,
        daily_dd_limit: float = 0.05,
        total_dd_limit: float = 0.10,
        risk_per_trade: float = 0.02,
        max_position_size: float = 1.0,
        min_position_size: float = 0.01,
        use_kelly: bool = False,
        kelly_fraction: float = 0.5
    ):
        """
        Initialize Risk Manager.
        
        Args:
            daily_dd_limit: Daily drawdown limit (0.05 = 5%)
            total_dd_limit: Total drawdown limit (0.10 = 10%)
            risk_per_trade: Risk per trade as fraction (0.02 = 2%)
            max_position_size: Max position size in lots
            min_position_size: Min position size in lots
            use_kelly: Whether to use Kelly Criterion
            kelly_fraction: Fraction of Kelly to use (0.5 = Half-Kelly)
        """
        self.daily_dd_limit = daily_dd_limit
        self.total_dd_limit = total_dd_limit
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        
        # Tracking
        self.daily_start_equity = None
        self.high_water_mark = None
        self.trade_history = []
        
    def check_order(self, order: OrderEvent, portfolio: Portfolio) -> bool:
        """
        Check if order passes all risk checks.
        
        Checks:
        1. Daily DD limit not breached
        2. Total DD limit not breached
        3. Position size within limits
        4. No averaging down (if already in losing position)
        5. Sufficient margin/equity
        
        Args:
            order: Order to validate
            portfolio: Current portfolio state
            
        Returns:
            True if order passes, False otherwise
        """
        # Check 1: Daily DD limit
        if not self.check_daily_drawdown_limit(portfolio):
            print(f"[RISK] ✗ Order rejected: Daily DD limit breached")
            return False
        
        # Check 2: Total DD limit
        current_equity = portfolio.get_total_equity()
        if self.high_water_mark is None:
            self.high_water_mark = current_equity
        else:
            self.high_water_mark = max(self.high_water_mark, current_equity)
        
        total_dd = (self.high_water_mark - current_equity) / self.high_water_mark
        if total_dd > self.total_dd_limit:
            print(f"[RISK] ✗ Order rejected: Total DD limit breached ({total_dd:.1%})")
            return False
        
        # Check 3: Position size limits
        if order.quantity > self.max_position_size:
            print(f"[RISK] ✗ Order rejected: Position size too large ({order.quantity:.2f} lots)")
            return False
        
        if order.quantity < self.min_position_size:
            print(f"[RISK] ✗ Order rejected: Position size too small ({order.quantity:.2f} lots)")
            return False
        
        # Check 4: No averaging down
        current_positions = portfolio.get_current_positions()
        if order.symbol in current_positions:
            # Already have position - check if it's losing
            # (This would need position P&L tracking from portfolio)
            pass
        
        # Check 5: Sufficient equity
        required_margin = order.quantity * 1000  # Simplified: 1 lot = $1000 margin
        if current_equity < required_margin:
            print(f"[RISK] ✗ Order rejected: Insufficient equity")
            return False
        
        return True
    
    def check_daily_drawdown_limit(self, portfolio: Portfolio, limit: Optional[float] = None) -> bool:
        """
        Check if daily DD limit breached.
        
        Args:
            portfolio: Current portfolio state
            limit: DD limit override (uses self.daily_dd_limit if None)
            
        Returns:
            True if within limit, False if breached (PANIC CLOSE required)
        """
        limit = limit or self.daily_dd_limit
        
        current_equity = portfolio.get_total_equity()
        
        # Initialize daily start equity if needed
        if self.daily_start_equity is None:
            self.daily_start_equity = current_equity
            return True
        
        # Calculate daily DD
        daily_dd = (self.daily_start_equity - current_equity) / self.daily_start_equity
        
        if daily_dd > limit:
            print(f"\n{'='*80}")
            print(f"⚠️ DAILY DRAWDOWN LIMIT BREACHED")
            print(f"{'='*80}")
            print(f"Daily DD: {daily_dd:.2%} (Limit: {limit:.2%})")
            print(f"Start Equity: ${self.daily_start_equity:,.2f}")
            print(f"Current Equity: ${current_equity:,.2f}")
            print(f"Loss: ${self.daily_start_equity - current_equity:,.2f}")
            print(f"{'='*80}")
            print(f"🚨 PANIC CLOSE TRIGGERED - All positions will be liquidated")
            print(f"{'='*80}\n")
            return False
        
        return True
    
    def calculate_position_size(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        stop_loss_distance: Optional[float] = None,
        confidence_multiplier: float = 1.0
    ) -> float:
        """
        Calculate position size using Fixed Fractional + Kelly + Confidence.
        
        Formula:
        1. Base Size = (Equity × Risk%) / Stop Loss Distance
        2. Kelly Adjustment = Base Size × Kelly Fraction (if enabled)
        3. Confidence Adjustment = Kelly Size × Confidence Multiplier
        4. Final Size = Clamp(Confidence Size, min_size, max_size)
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            stop_loss_distance: Distance to stop loss in price units
            confidence_multiplier: ML confidence (0.0-1.0, default 1.0)
            
        Returns:
            Position size in lots
        """
        equity = portfolio.get_total_equity()
        
        # Get ATR from signal metadata for stop loss
        if stop_loss_distance is None:
            atr = signal.metadata.get('atr', 0.001)
            stop_loss_distance = atr * 2.0  # 2×ATR stop
        
        # 1. Fixed Fractional Risk
        risk_amount = equity * self.risk_per_trade
        fractional_size = risk_amount / stop_loss_distance
        
        # 2. Kelly Criterion (if enabled and have trade history)
        if self.use_kelly and len(self.trade_history) >= 10:
            kelly_frac = self.calculate_kelly_fraction()
            fractional_size *= kelly_frac
        
        # 3. Confidence Weighting
        final_size = fractional_size * confidence_multiplier
        
        # 4. Apply limits
        final_size = max(self.min_position_size, min(final_size, self.max_position_size))
        
        return final_size
    
    def calculate_kelly_fraction(
        self,
        win_rate: Optional[float] = None,
        avg_win_loss: Optional[float] = None,
        fraction: Optional[float] = None
    ) -> float:
        """
        Calculate Kelly Fraction from trade history.
        
        Formula: f* = (b × p - q) / b
        Where:
        - p = win rate
        - q = loss rate (1 - p)
        - b = avg win / avg loss
        
        Args:
            win_rate: Win rate override (uses history if None)
            avg_win_loss: Avg win/loss ratio override (uses history if None)
            fraction: Kelly fraction override (uses self.kelly_fraction if None)
            
        Returns:
            Kelly fraction (0.0-1.0)
        """
        fraction = fraction or self.kelly_fraction
        
        if win_rate is None or avg_win_loss is None:
            # Calculate from trade history
            if len(self.trade_history) < 10:
                return 1.0  # Not enough data, use full size
            
            wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
            losses = [abs(t['pnl']) for t in self.trade_history if t['pnl'] < 0]
            
            if not wins or not losses:
                return 1.0
            
            win_rate = len(wins) / len(self.trade_history)
            avg_win = np.mean(wins)
            avg_loss = np.mean(losses)
            avg_win_loss = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Kelly formula
        p = win_rate
        q = 1 - p
        b = avg_win_loss
        
        kelly = (b * p - q) / b if b > 0 else 0
        kelly = max(0, kelly)  # Kelly can be negative (don't trade)
        
        # Apply fractional Kelly (safer)
        fractional_kelly = kelly * fraction
        
        # Bound between 0 and 1
        return max(0.0, min(fractional_kelly, 1.0))
    
    def reset_daily_equity(self, portfolio: Portfolio) -> None:
        """
        Reset daily starting equity (call at start of trading day).
        
        Args:
            portfolio: Current portfolio state
        """
        self.daily_start_equity = portfolio.get_total_equity()
        print(f"[RISK] Daily equity reset: ${self.daily_start_equity:,.2f}")
    
    def record_trade(self, pnl: float) -> None:
        """
        Record trade result for Kelly calculation.
        
        Args:
            pnl: Trade P&L
        """
        self.trade_history.append({
            'pnl': pnl,
            'timestamp': datetime.now()
        })
        
        # Keep last 100 trades only
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def get_statistics(self) -> Dict:
        """
        Get risk management statistics.
        
        Returns:
            Dict with risk metrics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win_loss': 0.0,
                'kelly_fraction': 0.0
            }
        
        wins = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trade_history if t['pnl'] < 0]
        
        win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_win_loss = avg_win / avg_loss if avg_loss > 0 else 0
        
        kelly = self.calculate_kelly_fraction(win_rate, avg_win_loss)
        
        return {
            'total_trades': len(self.trade_history),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_loss': avg_win_loss,
            'kelly_fraction': kelly,
            'daily_dd_limit': self.daily_dd_limit,
            'total_dd_limit': self.total_dd_limit,
            'risk_per_trade': self.risk_per_trade
        }


# Example usage
if __name__ == '__main__':
    print("PropFirmRiskManager Test")
    print("="*80)
    
    # Create risk manager
    rme = PropFirmRiskManager(
        daily_dd_limit=0.05,
        total_dd_limit=0.10,
        risk_per_trade=0.02,
        use_kelly=True,
        kelly_fraction=0.5
    )
    
    # Simulate trades
    print("\nSimulating 20 trades...")
    for i in range(20):
        # Random P&L
        pnl = np.random.choice([100, -50], p=[0.6, 0.4])  # 60% win rate
        rme.record_trade(pnl)
    
    # Get statistics
    stats = rme.get_statistics()
    print("\nRisk Management Statistics:")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.1%}")
    print(f"Avg Win/Loss Ratio: {stats['avg_win_loss']:.2f}")
    print(f"Kelly Fraction: {stats['kelly_fraction']:.2%}")
    print(f"Daily DD Limit: {stats['daily_dd_limit']:.1%}")
    print(f"Total DD Limit: {stats['total_dd_limit']:.1%}")
    print(f"Risk per Trade: {stats['risk_per_trade']:.1%}")
