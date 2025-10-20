"""
Strategy Matrix - Portfolio-Level Strategy Coordinator
Aggregates signals from multiple strategies, manages correlations, and applies portfolio risk rules.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from underdog.risk_management.risk_master import RiskMaster, StrategyMetrics
from underdog.risk_management.position_sizing import PositionSizer, SizingConfig
from underdog.strategies.fuzzy_logic.mamdani_inference import ConfidenceScorer


@dataclass
class StrategySignal:
    """Signal from a single strategy"""
    strategy_id: str
    symbol: str
    side: str  # "buy" or "sell"
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    confidence_score: float
    size_suggestion: float
    timestamp: datetime
    meta: Dict = field(default_factory=dict)


@dataclass
class AggregatedSignal:
    """Aggregated signal after strategy matrix processing"""
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    final_size: float
    combined_confidence: float
    participating_strategies: List[str]
    approved: bool
    rejection_reason: Optional[str] = None


class StrategyMatrix:
    """
    Portfolio-level coordinator for multiple strategies.
    
    Responsibilities:
    - Aggregate signals from multiple strategies
    - Calculate dynamic correlations between strategies
    - Apply portfolio-level risk constraints
    - Prioritize strategies based on performance
    - Manage capital allocation across strategies
    """
    
    def __init__(self,
                 risk_master: RiskMaster,
                 position_sizer: PositionSizer,
                 confidence_scorer: Optional[ConfidenceScorer] = None):
        """
        Initialize strategy matrix.
        
        Args:
            risk_master: Risk Master instance for portfolio-level risk
            position_sizer: Position sizer for calculating sizes
            confidence_scorer: Optional fuzzy logic confidence scorer
        """
        self.risk_master = risk_master
        self.position_sizer = position_sizer
        self.confidence_scorer = confidence_scorer
        
        # Strategy registry
        self.strategies: Dict[str, Dict] = {}
        
        # Signal queue
        self.pending_signals: List[StrategySignal] = []
        
        # Capital allocation
        self.strategy_allocations: Dict[str, float] = {}  # strategy_id -> allocation %
        
        # Performance tracking
        self.strategy_sharpe: Dict[str, float] = {}
        self.strategy_win_rates: Dict[str, float] = {}
    
    def register_strategy(self,
                         strategy_id: str,
                         allocation_pct: float = 10.0,
                         priority: int = 1,
                         meta: Optional[Dict] = None) -> None:
        """
        Register a strategy with the matrix.
        
        Args:
            strategy_id: Unique strategy identifier
            allocation_pct: Initial capital allocation percentage
            priority: Strategy priority (higher = more important)
            meta: Optional metadata
        """
        self.strategies[strategy_id] = {
            'allocation_pct': allocation_pct,
            'priority': priority,
            'active': True,
            'meta': meta or {}
        }
        
        self.strategy_allocations[strategy_id] = allocation_pct
        self.risk_master.register_strategy(strategy_id)
        
        print(f"[STRATEGY MATRIX] Registered strategy: {strategy_id} (allocation: {allocation_pct}%)")
    
    def deactivate_strategy(self, strategy_id: str, reason: str = "") -> None:
        """Temporarily deactivate a strategy"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['active'] = False
            print(f"[STRATEGY MATRIX] Deactivated {strategy_id}: {reason}")
    
    def activate_strategy(self, strategy_id: str) -> None:
        """Reactivate a strategy"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['active'] = True
            print(f"[STRATEGY MATRIX] Activated {strategy_id}")
    
    def submit_signal(self, signal: StrategySignal) -> None:
        """
        Submit a signal from a strategy.
        
        Args:
            signal: Strategy signal to process
        """
        # Validate strategy is registered and active
        if signal.strategy_id not in self.strategies:
            print(f"[STRATEGY MATRIX] Unknown strategy: {signal.strategy_id}")
            return
        
        if not self.strategies[signal.strategy_id]['active']:
            print(f"[STRATEGY MATRIX] Strategy {signal.strategy_id} is inactive")
            return
        
        self.pending_signals.append(signal)
        print(f"[STRATEGY MATRIX] Signal received: {signal.strategy_id} {signal.side} {signal.symbol} @ {signal.confidence_score:.2f}")
    
    def process_signals(self) -> List[AggregatedSignal]:
        """
        Process all pending signals and generate aggregated orders.
        
        Returns:
            List of aggregated signals ready for execution
        """
        if not self.pending_signals:
            return []
        
        # Group signals by symbol
        signals_by_symbol: Dict[str, List[StrategySignal]] = {}
        for signal in self.pending_signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)
        
        aggregated_signals = []
        
        for symbol, signals in signals_by_symbol.items():
            # Aggregate signals for this symbol
            agg_signal = self._aggregate_symbol_signals(symbol, signals)
            
            if agg_signal:
                aggregated_signals.append(agg_signal)
        
        # Clear processed signals
        self.pending_signals.clear()
        
        return aggregated_signals
    
    def _aggregate_symbol_signals(self, symbol: str, signals: List[StrategySignal]) -> Optional[AggregatedSignal]:
        """
        Aggregate multiple signals for the same symbol.
        
        Args:
            symbol: Trading symbol
            signals: List of signals for this symbol
        
        Returns:
            Aggregated signal or None if signals conflict/cancel
        """
        if not signals:
            return None
        
        # Separate by side
        buy_signals = [s for s in signals if s.side.lower() == "buy"]
        sell_signals = [s for s in signals if s.side.lower() == "sell"]
        
        # Check for conflicting signals
        if buy_signals and sell_signals:
            # Conflicting signals - compare combined confidence
            buy_conf = sum(s.confidence_score for s in buy_signals)
            sell_conf = sum(s.confidence_score for s in sell_signals)
            
            if abs(buy_conf - sell_conf) < 0.2:  # Too close, skip
                return AggregatedSignal(
                    symbol=symbol,
                    side="neutral",
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=None,
                    final_size=0.0,
                    combined_confidence=0.0,
                    participating_strategies=[],
                    approved=False,
                    rejection_reason="Conflicting signals with similar confidence"
                )
            
            # Use dominant side
            dominant_signals = buy_signals if buy_conf > sell_conf else sell_signals
        else:
            dominant_signals = buy_signals or sell_signals
        
        if not dominant_signals:
            return None
        
        # Calculate combined confidence (weighted average)
        total_weight = sum(s.confidence_score for s in dominant_signals)
        combined_confidence = total_weight / len(dominant_signals)
        
        # Determine side and prices (use signal with highest confidence)
        best_signal = max(dominant_signals, key=lambda s: s.confidence_score)
        side = best_signal.side
        entry_price = best_signal.entry_price
        stop_loss = best_signal.stop_loss
        take_profit = best_signal.take_profit
        
        # Calculate aggregate position size
        participating_strategies = [s.strategy_id for s in dominant_signals]
        
        # Check correlation penalty
        correlation_penalty = self._calculate_correlation_penalty(participating_strategies)
        adjusted_confidence = combined_confidence * correlation_penalty
        
        # Calculate position size
        account_balance = self.risk_master.current_capital
        
        # Get strategy performance metrics for Kelly
        strategy_metrics = self.risk_master.strategies.get(best_signal.strategy_id)
        win_rate = strategy_metrics.get_win_rate() if strategy_metrics else None
        
        sizing_result = self.position_sizer.calculate_size(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss=stop_loss,
            pip_value=10.0,  # TODO: Calculate per symbol
            confidence_score=adjusted_confidence,
            win_rate=win_rate,
            avg_win=None,  # TODO: Track from metrics
            avg_loss=None
        )
        
        if sizing_result['rejected']:
            return AggregatedSignal(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                final_size=0.0,
                combined_confidence=adjusted_confidence,
                participating_strategies=participating_strategies,
                approved=False,
                rejection_reason=sizing_result.get('rejection_reason', 'Position sizing rejected')
            )
        
        base_size = sizing_result['final_size']
        
        # Apply portfolio DD scaling from Risk Master
        dd_scaling = self.risk_master.get_dd_scaling_factor()
        final_size = base_size * dd_scaling
        
        # Pre-trade risk check
        approved, rejection_reason, risk_adjusted_size = self.risk_master.pre_trade_check(
            strategy_id=best_signal.strategy_id,
            symbol=symbol,
            side=side,
            size=final_size,
            price=entry_price,
            stop_loss=stop_loss
        )
        
        if approved:
            final_size = risk_adjusted_size
        
        return AggregatedSignal(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            final_size=final_size,
            combined_confidence=adjusted_confidence,
            participating_strategies=participating_strategies,
            approved=approved,
            rejection_reason=rejection_reason
        )
    
    def _calculate_correlation_penalty(self, strategy_ids: List[str]) -> float:
        """
        Calculate penalty factor based on strategy correlations.
        
        Args:
            strategy_ids: List of participating strategy IDs
        
        Returns:
            Penalty multiplier [0-1]
        """
        if len(strategy_ids) <= 1:
            return 1.0
        
        # Update correlation matrix
        self.risk_master.update_correlation_matrix()
        
        if self.risk_master.correlation_matrix is None:
            return 1.0
        
        # Calculate average correlation among participating strategies
        correlations = []
        for i, sid1 in enumerate(strategy_ids):
            for sid2 in strategy_ids[i+1:]:
                try:
                    corr = self.risk_master.correlation_matrix.loc[sid1, sid2]
                    correlations.append(abs(corr))
                except:
                    pass
        
        if not correlations:
            return 1.0
        
        avg_correlation = np.mean(correlations)
        
        # Apply penalty for high correlation
        # If avg correlation > 0.7, reduce confidence
        if avg_correlation > 0.7:
            penalty = 1.0 - (avg_correlation - 0.7) * 2  # Linear penalty
            return max(penalty, 0.5)  # Minimum 50% penalty
        
        return 1.0
    
    def rebalance_allocations(self) -> None:
        """
        Dynamically rebalance capital allocations based on strategy performance.
        Uses Sharpe ratios and win rates for allocation optimization.
        """
        # Update performance metrics
        for strategy_id in self.strategies.keys():
            if strategy_id in self.risk_master.strategies:
                metrics = self.risk_master.strategies[strategy_id]
                self.strategy_sharpe[strategy_id] = metrics.get_sharpe_ratio()
                self.strategy_win_rates[strategy_id] = metrics.get_win_rate()
        
        # Calculate new allocations based on Sharpe ratios
        total_sharpe = sum(max(s, 0) for s in self.strategy_sharpe.values())
        
        if total_sharpe > 0:
            for strategy_id in self.strategies.keys():
                sharpe = max(self.strategy_sharpe.get(strategy_id, 0), 0)
                new_allocation = (sharpe / total_sharpe) * 100
                
                # Smooth transition (exponential moving average)
                old_allocation = self.strategy_allocations.get(strategy_id, 10.0)
                self.strategy_allocations[strategy_id] = 0.7 * old_allocation + 0.3 * new_allocation
        
        print(f"[STRATEGY MATRIX] Rebalanced allocations: {self.strategy_allocations}")
    
    def get_portfolio_status(self) -> Dict:
        """Get comprehensive portfolio status"""
        return {
            'active_strategies': sum(1 for s in self.strategies.values() if s['active']),
            'total_strategies': len(self.strategies),
            'strategy_allocations': self.strategy_allocations,
            'pending_signals': len(self.pending_signals),
            'risk_metrics': self.risk_master.get_portfolio_metrics(),
            'strategy_performance': {
                sid: {
                    'sharpe': self.strategy_sharpe.get(sid, 0.0),
                    'win_rate': self.strategy_win_rates.get(sid, 0.0),
                    'allocation_pct': self.strategy_allocations.get(sid, 0.0)
                }
                for sid in self.strategies.keys()
            }
        }
