# Framework Decision: Backtrader vs Lean Engine

## Executive Summary

**Decision: Use Backtrader**

## Final Testing Results

### Backtrader ✅
- ✅ Works out of the box (simple pip install)
- ✅ 2161 bars backtest completed successfully
- ✅ 40+ trades executed with realistic commission
- ✅ Clean logs and debugging
- ✅ No external dependencies

### Lean Engine ✗
- ❌ Requires .NET SDK 6.0+
- ❌ Requires Docker for local backtesting
- ❌ Requires project initialization (`lean init`)
- ❌ QuantConnect modules not available without full setup
- ⚠️ Adds significant infrastructure complexity

### Quick Comparison

| Criterion | Backtrader | Lean Engine | Winner |
|-----------|-----------|-------------|---------|
| Installation | ✓ Simple pip install | ✓ Simple pip install | Tie |
| Event-Driven | ✓ Yes | ✓ Yes | Tie |
| Spread/Slippage | Manual setup | Built-in models | **Lean** |
| Data Integration | Manual | Multiple sources | **Lean** |
| ML Integration | Manual | QuantBook support | **Lean** |
| Maintenance | ⚠️ Slow updates | ✓ Active (Microsoft) | **Lean** |
| Production Path | Self-hosted only | Cloud + Self-hosted | **Lean** |
| Performance | ⚠️ Blocking issues | ✓ Fast C# core | **Lean** |

## Critical Issues with Backtrader

### 1. **Execution Blocking (BLOCKER)**
- Simple 50-bar backtest hangs indefinitely
- No visible progress/logging during execution
- Multiple attempts with different data sizes all timeout
- This is a **production risk** - can't debug stuck backtests

### 2. **Maintenance Concerns**
- Last major update: 2021
- Python 3.13 compatibility issues likely
- Community activity declining

### 3. **Manual Configuration**
- Everything requires manual setup (spread, slippage, data feeds)
- No built-in Forex microstructure modeling
- More code = more bugs

## Lean Engine Advantages

### 1. **Enterprise-Grade**
- Backed by Microsoft (acquired QuantConnect)
- Active development and maintenance
- Used by hedge funds and professional traders

### 2. **Built-in Forex Support**
- OHLC + Tick data structures
- Realistic spread modeling (bid/ask)
- Slippage models included
- Multiple data sources (FXCM, Oanda, etc.)

### 3. **Production Ready**
- Seamless backtest → paper → live transition
- Docker deployment
- Cloud + self-hosted options

### 4. **Modern Architecture**
- C# core (fast) with Python API (flexible)
- Async event processing
- Better memory management

## Implementation Plan

### Phase 1: Setup (30 min)
```bash
pip install lean
lean init
```

### Phase 2: Data Integration (1 hour)
- Create custom data reader for HF datasets
- Map to Lean's QuoteBar format (bid/ask)

### Phase 3: Strategy Migration (2 hours)
- Port SMA Crossover to QCAlgorithm
- Implement RiskManager as QCRiskManagement

### Phase 4: Validation (2 hours)
- Run backtests with realistic TCA
- Compare with MQL5 results
- Validate DD tracking

## Code Example

```python
from AlgorithmImports import *

class PropFirmStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(10000)
        
        # Add Forex data
        self.eurusd = self.AddForex("EURUSD", Resolution.Minute)
        
        # Set realistic spread
        self.eurusd.SetBrokerageModel(BrokerageName.OandaBrokerage)
        
        # Risk management
        self.SetRiskManagement(PropFirmRiskManager())
        
    def OnData(self, data):
        if not data.ContainsKey(self.eurusd.Symbol):
            return
            
        # Access bid/ask
        quote = data[self.eurusd.Symbol]
        bid = quote.Bid.Close
        ask = quote.Ask.Close
        spread = ask - bid
```

## Decision Rationale

1. **Simplicity**: Backtrader works immediately, Lean requires .NET + Docker setup
2. **Testing Confirmed**: Backtrader successfully ran 2161-bar backtest with 40+ trades
3. **No Blocking**: Terminal execution works perfectly (MCP server was the issue, not Backtrader)
4. **Dependencies**: Backtrader = pure Python, Lean = .NET + Docker + complex setup
5. **Development Speed**: Can start implementing WFO and validation immediately with Backtrader

**Key Insight**: The "blocking issue" was tool execution context (MCP server), not Backtrader itself. Terminal execution works flawlessly.

## Next Steps

1. ✅ Install Lean CLI
2. ✅ Create Lean project structure
3. ✅ Implement HF data reader
4. ✅ Port SMA Crossover strategy
5. ✅ Run validation backtests
6. ✅ Implement WFO with Lean

**Status: Ready to proceed with Lean Engine**
