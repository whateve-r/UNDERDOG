"""
Lean Engine (QuantConnect) simple test.
Test basic functionality with SMA Crossover strategy.
"""

from AlgorithmImports import *

class SMACrossoverLean(QCAlgorithm):
    
    def Initialize(self):
        """Initialize algorithm settings."""
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 3, 31)
        self.SetCash(10000)
        
        # Add Forex data
        self.symbol = self.AddForex("EURUSD", Resolution.Hour, Market.Oanda).Symbol
        
        # Create indicators
        self.fast_ma = self.SMA(self.symbol, 10, Resolution.Hour)
        self.slow_ma = self.SMA(self.symbol, 30, Resolution.Hour)
        
        # Warm up period
        self.SetWarmUp(30, Resolution.Hour)
        
        self.Debug("Strategy initialized")
        
    def OnData(self, data):
        """Process new data."""
        if self.IsWarmingUp:
            return
            
        if not data.ContainsKey(self.symbol):
            return
            
        if not self.fast_ma.IsReady or not self.slow_ma.IsReady:
            return
        
        # Get current values
        fast = self.fast_ma.Current.Value
        slow = self.slow_ma.Current.Value
        
        # Trading logic
        if not self.Portfolio.Invested:
            if fast > slow:
                self.SetHoldings(self.symbol, 1.0)
                self.Debug(f"BUY - Fast: {fast:.5f} > Slow: {slow:.5f}")
        else:
            if fast < slow:
                self.Liquidate(self.symbol)
                self.Debug(f"SELL - Fast: {fast:.5f} < Slow: {slow:.5f}")
    
    def OnEndOfAlgorithm(self):
        """Called when algorithm finishes."""
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:.2f}")
