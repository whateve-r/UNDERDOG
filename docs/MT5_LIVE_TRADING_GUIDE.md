# MT5 Live Trading - Quick Start Guide

## 🎯 Objetivo

Este documento explica cómo ejecutar el sistema de trading en vivo con MetaTrader 5, desde la configuración inicial hasta el paper trading en cuenta DEMO.

**CRÍTICO**: Antes de ejecutar en cuenta REAL, DEBES completar 30 días de paper trading exitoso en DEMO.

---

## 📋 Pre-requisitos

### 1. MetaTrader 5 Instalado

- **Windows**: Descargar de [MetaQuotes](https://www.metatrader5.com/en/download)
- **Verificación**: Abrir MT5, verificar que se inicia correctamente

### 2. Cuenta DEMO MT5

Necesitas una cuenta DEMO de un broker para testing:

**Recomendados**:
- **ICMarkets**: Demo con condiciones similares a real
- **FTMO**: Demo para familiarizarte con su plataforma

**Obtener cuenta DEMO**:
1. Abrir MT5
2. File → Open an Account
3. Buscar broker (ej: "ICMarkets")
4. Seleccionar "Open a demo account"
5. Completar formulario
6. **GUARDAR**: Account Number, Password, Server

### 3. Python Package `MetaTrader5`

```bash
poetry add MetaTrader5
```

O con pip:
```bash
pip install MetaTrader5
```

---

## 🚀 Ejecución Paso a Paso

### PASO 1: Validar MT5Executor (10 órdenes de prueba)

Este script ejecuta 10 órdenes en DEMO para validar que todo funciona:

```bash
poetry run python scripts/demo_paper_trading.py \
  --account 12345678 \
  --password "tu_password" \
  --server "ICMarkets-Demo" \
  --symbol EURUSD \
  --volume 0.01 \
  --orders 10
```

**Reemplaza**:
- `12345678` → Tu número de cuenta DEMO
- `tu_password` → Tu contraseña DEMO
- `ICMarkets-Demo` → El servidor de tu broker

**Criterios de éxito**:
- ✅ Mínimo 8/10 órdenes ejecutadas exitosamente
- ✅ Zero violaciones de DD límites (5% daily, 10% total)
- ✅ Emergency close funciona correctamente
- ✅ CSV exportado a `data/test_results/demo_paper_trading_YYYYMMDD_HHMMSS.csv`

**Si falla**: Revisar:
1. MT5 está abierto y conectado
2. Credenciales correctas
3. Símbolo disponible (algunos brokers usan "EURUSDm" en lugar de "EURUSD")

---

### PASO 2: Paper Trading con Estrategia Real

Una vez validado el ejecutor, prueba con una estrategia Backtrader:

```python
# scripts/live_paper_trading.py (crear este archivo)

from underdog.execution.mt5_executor import MT5Executor
from underdog.bridges.bt_to_mt5 import BacktraderMT5Bridge
from underdog.strategies.bt_strategies.atr_breakout_live import ATRBreakoutLive
import backtrader as bt

# 1. Crear MT5Executor
executor = MT5Executor(
    account=12345678,
    password="tu_password",
    server="ICMarkets-Demo",
    max_daily_dd=5.0,
    max_total_dd=10.0
)

if not executor.initialize():
    print("❌ Error al conectar con MT5")
    exit(1)

# 2. Crear Bridge
bridge = BacktraderMT5Bridge(
    executor=executor,
    default_volume=0.1,
    default_sl_pips=20,
    default_tp_pips=40
)

# 3. Configurar Cerebro con datos en tiempo real
# NOTA: Para live trading, necesitas un data feed en tiempo real
# Opción A: Usar MT5 como data feed (requiere implementar MT5DataFeed)
# Opción B: Usar API de terceros (ej: IQFeed, Interactive Brokers)
# Por ahora, esto es un esqueleto que muestra la estructura

cerebro = bt.Cerebro()

# 4. Agregar estrategia con bridge
cerebro.addstrategy(
    ATRBreakoutLive,
    mt5_bridge=bridge,
    symbol="EURUSD",
    volume=0.1,
    atr_multiplier=2.0
)

# 5. TODO: Agregar data feed en tiempo real
# data = MT5LiveDataFeed(symbol="EURUSD", timeframe="M5")
# cerebro.adddata(data)

# 6. Ejecutar
print("🚀 Iniciando paper trading...")
cerebro.run()

# 7. Exportar logs
bridge.export_logs("data/test_results/live_paper_trading.csv")

# 8. Estadísticas
stats = bridge.get_statistics()
print("\n📊 ESTADÍSTICAS:")
print(f"Total signals: {stats['total_signals']}")
print(f"Successful orders: {stats['successful_orders']}")
print(f"Success rate: {stats['success_rate']:.2f}%")
print(f"DD rejections: {stats['dd_rejections']}")

# 9. Cleanup
executor.shutdown()
```

---

## 📊 Monitoreo Durante Paper Trading

### Métricas Críticas a Vigilar

1. **Daily Drawdown** (debe ser < 5%)
   ```python
   daily_dd, total_dd = executor.calculate_drawdown()
   print(f"Daily DD: {daily_dd:.2f}%")
   ```

2. **Posiciones Abiertas**
   ```python
   positions = executor.get_open_positions()
   for pos in positions:
       print(f"Ticket: {pos.ticket}, Profit: ${pos.profit:.2f}, Duration: {pos.duration_hours:.1f}h")
   ```

3. **Uptime del Bot**
   - Objetivo: >99.9% uptime
   - En 30 días = máximo 43 minutos de downtime permitidos

4. **Logs de Ejecución**
   - `bridge.get_execution_log()` → Todas las órdenes enviadas
   - `bridge.get_signal_log()` → Todas las señales generadas

---

## 🔴 Emergency Stop

Si necesitas cerrar TODAS las posiciones inmediatamente:

```python
# En tu script
closed_count = executor.emergency_close_all(reason="Manual intervention")
print(f"Closed {closed_count} positions")
```

O desde consola Python:
```python
from underdog.execution.mt5_executor import MT5Executor

executor = MT5Executor(account=12345678, password="xxx", server="ICMarkets-Demo")
executor.initialize()
executor.emergency_close_all(reason="Emergency stop")
executor.shutdown()
```

---

## 📈 30 Días Paper Trading (Validación Completa)

Antes de ir a FTMO o cuenta REAL, ejecutar **30 días continuos** en DEMO:

### Criterios GO/NO-GO para Producción

| Métrica | Objetivo | CRÍTICO |
|---------|----------|---------|
| Uptime | >99.9% | ✅ SÍ |
| Max Daily DD | <5% | ✅ SÍ |
| Max Total DD | <8% | ✅ SÍ |
| Rentabilidad | >0% | 🟡 Deseable |
| Zero alertas críticas | 0 | ✅ SÍ |
| Reconexión automática | Funciona | ✅ SÍ |

**Si ALGUNO falla**: NO pasar a producción. Arreglar y repetir 30 días.

### Monitoreo Diario (5 min/día)

```bash
# Ver estado general
poetry run python scripts/daily_check.py  # TODO: Crear este script

# Revisar logs
tail -n 100 logs/underdog_bot.log

# Verificar posiciones abiertas
poetry run python scripts/check_positions.py  # TODO: Crear este script
```

---

## 🎓 Siguiente Paso: FTMO Challenge

Una vez pasados 30 días de paper trading exitoso:

1. **Comprar FTMO Challenge**: €155 para cuenta 50k
2. **Configurar cuenta FTMO DEMO** (Phase 1 empieza en DEMO)
3. **Cambiar credenciales en script**:
   ```python
   executor = MT5Executor(
       account=FTMO_ACCOUNT_NUMBER,
       password=FTMO_PASSWORD,
       server="FTMO-Server",
       max_daily_dd=5.0,  # FTMO límite
       max_total_dd=10.0   # FTMO límite
   )
   ```
4. **Ejecutar 30 días**: Objetivo 8% profit, <5% daily DD, <10% total DD
5. **Si pasas Phase 1**: Repetir en Phase 2 (14 días, mismo objetivo)
6. **Si pasas Phase 2**: ¡CUENTA FUNDED! €2,000-4,000/mes potencial

---

## 🛡️ Seguridad y Risk Management

### DD Limits Enforcement

El sistema **RECHAZA** órdenes si:
- Daily DD >= 5%
- Total DD >= 10%

```python
# Pre-ejecución check (automático en execute_order)
is_valid, reason = executor._validate_drawdown_limits()
if not is_valid:
    print(f"❌ Orden rechazada: {reason}")
```

### Auto-Reconnect

Si se pierde conexión MT5:
- **5 intentos** de reconexión
- **10 segundos** entre intentos
- Si falla: Alert y shutdown

```python
# Configurar reintentos
executor = MT5Executor(
    ...,
    reconnect_attempts=5,
    reconnect_delay=10
)
```

---

## 📝 Logging y Audit Trail

Todos los eventos críticos se loguean:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/underdog_bot.log'),
        logging.StreamHandler()
    ]
)
```

**Eventos logueados**:
- ✅ Órdenes ejecutadas (ticket, precio, volumen)
- ❌ Órdenes rechazadas (razón, DD al momento)
- 🔄 Reconexiones exitosas/fallidas
- 🚨 Emergency stop triggered
- 📊 DD checks (cada orden)

---

## 🐛 Troubleshooting

### Error: "MT5 initialize() failed"

**Causa**: MT5 no instalado o no está corriendo

**Solución**:
1. Abrir MetaTrader 5 manualmente
2. Verificar que se conecta correctamente
3. Ejecutar script nuevamente

### Error: "MT5 login failed"

**Causa**: Credenciales incorrectas o servidor incorrecto

**Solución**:
1. Verificar Account Number, Password, Server en MT5
2. Revisar que el servidor tiene "-Demo" si es demo (ej: "ICMarkets-Demo")
3. Intentar login manual en MT5 primero

### Error: "Symbol EURUSD not found"

**Causa**: Broker usa nombre diferente (ej: "EURUSDm")

**Solución**:
1. Abrir MT5 → Market Watch
2. Buscar el símbolo exacto
3. Usar ese nombre en `symbol` parameter

### Error: "Order rejected - retcode 10016"

**Causa**: Mercado cerrado (fuera de horario de trading)

**Solución**:
- Forex trading: Lunes 00:00 - Viernes 23:59 GMT
- Ejecutar durante horas de mercado

### Error: "DD limit breach"

**Causa**: Drawdown actual >= límite configurado

**Solución**:
- Esto es ESPERADO (protección funcionando)
- Verificar equity actual vs starting balance
- Considerar `emergency_close_all()` si DD muy alto

---

## 📚 Recursos Adicionales

- **MetaTrader 5 Python Docs**: https://www.mql5.com/en/docs/integration/python_metatrader5
- **FTMO Rules**: https://ftmo.com/en/evaluation-process/
- **Backtrader Docs**: https://www.backtrader.com/docu/

---

## ✅ Checklist Pre-Launch

Antes de ejecutar en DEMO:

- [ ] MT5 instalado y funcionando
- [ ] Cuenta DEMO creada y probada manualmente
- [ ] `MetaTrader5` package instalado (`poetry add MetaTrader5`)
- [ ] Script `demo_paper_trading.py` ejecutado exitosamente (10 órdenes)
- [ ] Logs configurados (`logs/underdog_bot.log`)
- [ ] Emergency stop probado

Antes de ejecutar en FTMO/REAL:

- [ ] 30 días paper trading completados exitosamente
- [ ] Uptime >99.9% validado
- [ ] Max DD <8% observado
- [ ] Zero alertas críticas en 30 días
- [ ] Reconexión automática probada y funcional
- [ ] Monitoreo Prometheus+Grafana configurado (opcional pero recomendado)

---

**¡Éxito en tu journey hacia el trading algorítmico!** 🚀

Para preguntas: Revisar logs, código fuente, o consultar documentación MT5.
