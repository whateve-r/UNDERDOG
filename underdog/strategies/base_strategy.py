"""
Base Strategy Module
Provides base classes and utilities for trading strategies.

Includes:
- BaseStrategy abstract class
- PositionTracker for time-based exits
- Time Stop implementation for mean reversion
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


@dataclass
class PositionTracker:
    """
    Tracks position lifecycle and enforces time-based exits.
    
    Time stops are critical for mean reversion strategies to prevent
    holding positions that fail to revert within expected timeframe.
    
    Recommended max bars by timeframe (Ernie Chan, 2013):
    - M5: 48 bars (4 hours)
    - M15: 20 bars (5 hours)
    - H1: 10 bars (10 hours)
    """
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    size: float
    timeframe_minutes: int = 15
    max_bars_held: int = 20  # Default for M15
    
    def bars_held(self, current_time: datetime) -> int:
        """
        Calculate number of bars held since entry.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            Number of bars held
        """
        time_delta = current_time - self.entry_time
        minutes_held = time_delta.total_seconds() / 60
        return int(minutes_held / self.timeframe_minutes)
    
    def should_time_stop(self, current_time: datetime) -> bool:
        """
        Check if position should be closed due to time stop.
        
        Mean reversion strategies expect reversion within a specific
        timeframe. If it hasn't reverted by then, the trade thesis
        is invalidated.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            True if position should be closed
        
        Example:
            >>> tracker = PositionTracker(
            ...     symbol='EURUSD',
            ...     side='long',
            ...     entry_time=datetime(2024, 1, 1, 10, 0),
            ...     entry_price=1.1000,
            ...     size=0.01,
            ...     timeframe_minutes=15,
            ...     max_bars_held=20
            ... )
            >>> current_time = datetime(2024, 1, 1, 15, 0)  # 5 hours later
            >>> tracker.should_time_stop(current_time)
            True
        """
        return self.bars_held(current_time) >= self.max_bars_held
    
    def get_time_stop_config(timeframe_minutes: int) -> int:
        """
        Get recommended max bars for timeframe.
        
        Args:
            timeframe_minutes: Timeframe in minutes
        
        Returns:
            Recommended max bars to hold
        """
        config = {
            5: 48,   # M5: 4 hours
            15: 20,  # M15: 5 hours
            60: 10,  # H1: 10 hours
        }
        return config.get(timeframe_minutes, 20)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement:
    - generate_signals(): Signal generation logic
    - validate_signal(): Pre-trade validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy.
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config or {}
        self.positions: Dict[str, PositionTracker] = {}
    
    @abstractmethod
    def generate_signals(self, data: Any) -> Any:
        """
        Generate trading signals from market data.
        
        Args:
            data: Market data (format depends on strategy)
        
        Returns:
            Trading signals
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Any) -> bool:
        """
        Validate signal before execution.
        
        Args:
            signal: Trading signal to validate
        
        Returns:
            True if signal is valid
        """
        pass
    
    def check_time_stops(self, current_time: datetime) -> Dict[str, bool]:
        """
        Check all positions for time stop violations.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            Dict mapping symbol to should_exit bool
        """
        exits = {}
        for symbol, tracker in self.positions.items():
            if tracker.should_time_stop(current_time):
                exits[symbol] = True
        return exits
