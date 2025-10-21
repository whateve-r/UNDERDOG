# MT5 Live Trading - Quick Reference Card

## 🚀 Quick Start Commands

### 1. Demo Paper Trading Test (10 órdenes)
```bash
poetry run python scripts/demo_paper_trading.py \
  --account 12345678 \
  --password "tu_password" \
  --server "ICMarkets-Demo" \
  --symbol EURUSD \
  --volume 0.01 \
  --orders 10
```

### 2. Check Python Environment
```bash
poetry run python -c "import MetaTrader5 as mt5; print('MT5 version:', mt5.__version__)"
```

### 3. Test MT5 Connection (Manual)
```python
import MetaTrader5 as mt5

# Initialize
if not mt5.initialize():
    print("MT5 initialize failed")
    quit()

# Login
account = 12345678
password = "xxx"
server = "ICMarkets-Demo"

if mt5.login(account, password=password, server=server):
    print("✅ Login successful")
    print(f"Balance: ${mt5.account_info().balance}")
else:
    print("❌ Login failed:", mt5.last_error())

mt5.shutdown()
```

---

## 📊 Common Tasks

### Get Open Positions
```python
from underdog.execution import MT5Executor

executor = MT5Executor(account=..., password=..., server=...)
executor.initialize()

positions = executor.get_open_positions()
for pos in positions:
    print(f"Ticket: {pos.ticket}, Symbol: {pos.symbol}, Profit: ${pos.profit:.2f}")

executor.shutdown()
```

### Check Current Drawdown
```python
daily_dd, total_dd = executor.calculate_drawdown()
print(f"Daily DD: {daily_dd:.2f}%")
print(f"Total DD: {total_dd:.2f}%")
```

### Emergency Close All Positions
```python
closed_count = executor.emergency_close_all(reason="Manual stop")
print(f"Closed {closed_count} positions")
```

### Execute Single Order
```python
from underdog.execution import OrderType

result = executor.execute_order(
    symbol="EURUSD",
    order_type=OrderType.BUY,
    volume=0.1,
    sl_pips=20,
    tp_pips=40,
    comment="Manual_Order"
)

if result.status == OrderStatus.SUCCESS:
    print(f"✅ Ticket: {result.ticket}, Price: {result.price}")
else:
    print(f"❌ Rejected: {result.error_message}")
```

---

## 🔍 Monitoring & Debugging

### View Execution Logs (Bridge)
```python
from underdog.bridges import BacktraderMT5Bridge

# After running with bridge
signals_df = bridge.get_signal_log()
executions_df = bridge.get_execution_log()

print(signals_df.tail(10))
print(executions_df.tail(10))

# Export to CSV
bridge.export_logs("data/test_results/my_session.csv")
```

### Get Statistics
```python
stats = bridge.get_statistics()
print(f"Total Signals: {stats['total_signals']}")
print(f"Successful Orders: {stats['successful_orders']}")
print(f"Success Rate: {stats['success_rate']:.2f}%")
print(f"DD Rejections: {stats['dd_rejections']}")
```

---

## 🐛 Troubleshooting

### MT5 Not Connecting?
```bash
# Check if MT5 is running
tasklist /FI "IMAGENAME eq terminal64.exe"

# If not running, start it manually first
```

### Symbol Not Found?
```python
import MetaTrader5 as mt5
mt5.initialize()

# List all symbols
symbols = mt5.symbols_get()
for s in symbols[:20]:  # First 20
    print(s.name)

# Or search for specific
eurusd = mt5.symbols_get("EURUSD*")
print([s.name for s in eurusd])
```

### Check Last Error
```python
import MetaTrader5 as mt5

# After any failed operation
error = mt5.last_error()
print(f"Error code: {error[0]}, Message: {error[1]}")
```

---

## 📁 File Locations

| What | Path |
|------|------|
| MT5Executor | `underdog/execution/mt5_executor.py` |
| Bridge | `underdog/bridges/bt_to_mt5.py` |
| Demo Test Script | `scripts/demo_paper_trading.py` |
| Live Strategy Example | `underdog/strategies/bt_strategies/atr_breakout_live.py` |
| Full Guide | `docs/MT5_LIVE_TRADING_GUIDE.md` |
| Test Results | `data/test_results/` |
| Logs | `logs/underdog_bot.log` |

---

## ⚡ Important Limits (PropFirm)

| Metric | FTMO | The5ers | MyForexFunds |
|--------|------|---------|--------------|
| Daily DD Limit | 5% | 5% | 5% |
| Total DD Limit | 10% | 10% | 10% |
| Profit Target (Phase 1) | 8% | 8% | 8% |
| Min Trading Days | 4 | 4 | 5 |
| Max Trading Period | 30 days | 30 days | 30 days |

**Critical**: Sistema rechaza órdenes automáticamente si Daily DD >= 5% o Total DD >= 10%

---

## 🔐 Security Notes

### Never Commit Credentials
```bash
# Add to .gitignore
echo "mt5_credentials.py" >> .gitignore
echo "*.env" >> .gitignore
```

### Use Environment Variables
```bash
# .env file (NOT committed)
MT5_ACCOUNT=12345678
MT5_PASSWORD=xxx
MT5_SERVER=ICMarkets-Demo

# In script
import os
account = int(os.getenv('MT5_ACCOUNT'))
password = os.getenv('MT5_PASSWORD')
server = os.getenv('MT5_SERVER')
```

---

## 📞 Support

- MT5 Python Docs: https://www.mql5.com/en/docs/integration/python_metatrader5
- FTMO Support: https://ftmo.com/en/support/
- Project Issues: Check logs first (`logs/underdog_bot.log`)

---

**Last Updated**: 21 Octubre 2025  
**Version**: 1.0 - MT5 Infrastructure Initial Release
