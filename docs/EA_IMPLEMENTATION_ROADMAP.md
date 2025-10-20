# ROADMAP DE IMPLEMENTACIÓN - SISTEMA DE TRADING MULTIESTRATÉGICO

**Proyecto**: UNDERDOG - Trading System for Prop Firms  
**Objetivo**: 7 EAs descorrelacionados para pasar challenges de Prop Firms  
**Compliance**: Drawdown Diario 5% | Total 10% | Floating 1% (Instant Funding)

---

## 🎯 FILOSOFÍA DEL SISTEMA

### Matriz de Descorrelación Estratégica

El éxito en Prop Firms requiere **portfolio diversificado** para evitar Regime Crashes:

| EA | Régimen Primario | Regímenes Secundarios | Timeframe | R:R |
|----|------------------|----------------------|-----------|-----|
| **FxSuperTrendRSI** | Trend (Largo Plazo) | Momentum (RSI) | M5 | 1:1.5 |
| **FxParabolicEMA** | Trend Adaptativo | Volatilidad (EMA) | M15 | Variable |
| **FXPairArbitrage** | Mean Reversion (Stat) | Cointegración | H1 | 1:0.6 |
| **FxKeltnerBreakout** | Breakout | Volumen (OBV) | M15 | Asimétrico |
| **FxEmaScalper** | Momentum/Scalping | Pendiente Fuzzy | M5 | 1:1.5 |
| **FxBollingerCCI** | Mean Reversion | Extremos CCI | M15 | 1:0.6 |
| **FXATRBreakout** | Breakout Volatilidad | Spike ATR + RSI | M15 | 1:1.5 |

**Balance del portafolio**:
- ✅ 3 EAs de Trend/Momentum (SuperTrend, Parabolic, Keltner)
- ✅ 2 EAs de Mean Reversion (PairArbitrage, BollingerCCI)
- ✅ 2 EAs de Breakout (Keltner, ATRBreakout)
- ✅ Cobertura de M5, M15, H1 para diversificación temporal

---

## 📋 STATUS DE IMPLEMENTACIÓN

### ✅ COMPLETADO (Fase 1-2)

#### Infraestructura Core
- [x] **TensorFlow 2.20.0** + NumPy 2.3.4 (119 packages)
- [x] **Redis State Cache** (`redis_cache.py` - 400 líneas)
- [x] **CVaR Risk Metrics** (`cvar.py` - 500 líneas)
- [x] **Confidence-Weighted Sizing** (`position_sizing.py`)
- [x] **Event-Driven Backtesting** (`event_driven.py` - 300 líneas)
- [x] **Polyglot Architecture** (`POLYGLOT_ARCHITECTURE.md` - 1,800 líneas)

#### Módulos Críticos (Nuevos)
- [x] **Floating Drawdown Monitor** (`floating_drawdown_monitor.py` - 450 líneas)
  - Monitoreo en tiempo real de DD diario/total/flotante
  - Emergency close all positions
  - Integración con Redis
  - Callback system para alertas

- [x] **Fault-Tolerant Executor** (`fault_tolerant_executor.py` - 500 líneas)
  - Retry loop (max 5 intentos, 300ms sleep)
  - Slippage dinámico (20 → 50 pips)
  - Manejo de errores transitorios (Requote, Off Quotes, etc.)
  - Estadísticas de ejecución

---

### 🟡 EN PROGRESO (Fase 3)

#### Base de Datos
- [ ] **DuckDB Analytics Store** (`database/duckdb_store.py`)
  - Queries 100x más rápidas que Pandas
  - Ingestión de Parquet/HDF5
  - Compatible con dataframes
  
  ```python
  # Prioridad: ALTA
  # Estimación: 300 líneas, 2-3 días
  
  class DuckDBStore:
      def query_ohlcv(self, symbol, start, end) -> pd.DataFrame
      def ingest_parquet(self, file_path)
      def create_features_table(self)
  ```

#### Estrategias de Trading (Python-based EAs)

**EA #1: FxSuperTrendRSI** (Trend/Momentum)
```python
# File: underdog/strategies/ea_supertrend_rsi.py
# Prioridad: ALTA (Trend Following)
# Estimación: 400 líneas

class FxSuperTrendRSI:
    """
    SuperTrend + RSI filtrado.
    
    LÓGICA:
    - Entry: ST_direction + RSI momentum (>65 BUY, <35 SELL)
    - SL: max(ATR*multiplier, ST_Line)
    - Exit: Trailing Stop con ADX hysteresis
    - Timeframe: M5
    
    PARÁMETROS:
    - Supertrend_Multiplier: 3.5-4.0 (optimizar vs 3.0)
    - RSI_Period: 14
    - RSI_Buy_Level: 65 (vs 60)
    - RSI_Sell_Level: 35 (vs 40)
    - TrailingStart_Pips: 15
    - ADX_Exit_Threshold: 15 (nuevo - hysteresis)
    """
```

**EA #2: FxParabolicEMA** (Adaptive Trend)
```python
# File: underdog/strategies/ea_parabolic_ema.py
# Prioridad: ALTA
# Estimación: 450 líneas

class FxParabolicEMA:
    """
    Parabolic SAR + EMA Adaptativa (polimórfica).
    
    LÓGICA:
    - Entry: SAR reversal + alineación con EMA
    - EMA period: dinámico (12-70) según ATR
    - MEJORA: Dual-regime EMA (ADX alto = EMA rápida, ADX bajo = EMA lenta)
    - SL: SAR anterior (con fallback ATR)
    
    PARÁMETROS:
    - K_EMA_Min_Period: 12
    - K_EMA_Max_Period: 70
    - MinATR_Pips: 20 (por instrumento)
    - SAR_Step: 0.015 (vs 0.02 para M15)
    - ADX_Trend_Threshold: 25 (nuevo)
    """
```

**EA #3: FXPairArbitrage** (Statistical Arbitrage)
```python
# File: underdog/strategies/ea_pair_arbitrage.py
# Prioridad: MEDIA (Nicho - Mean Reversion)
# Estimación: 500 líneas

class FXPairArbitrage:
    """
    Mean reversion multi-pair con hedge ratio dinámico.
    
    LÓGICA:
    - Entry: Z-Score > 2.0 (spread overextended)
    - Exit: Z-Score < 0.5 (reversion)
    - SL: Z-Score > 3.0 (stop loss estadístico)
    - Beta: OLS regression (200 bars)
    - MEJORA: Cointegration stability monitor
    
    PARÁMETROS:
    - SpreadLookback: 200 (H1)
    - Open_ZScore_Threshold: 2.0-2.5
    - Close_ZScore_Threshold: 0.5
    - MaxLoss_ZScore: 3.0
    - Cointegracion_Check_Bars: 5 (nuevo)
    """
```

**EA #4: FxKeltnerBreakout** (Volume Confirmed Breakout)
```python
# File: underdog/strategies/ea_keltner_breakout.py
# Prioridad: ALTA
# Estimación: 400 líneas

class FxKeltnerBreakout:
    """
    Keltner Channel + OBV confirmation.
    
    LÓGICA:
    - Entry: 2 cierres consecutivos fuera de canal + OBV cross
    - Filter: S/N ratio (ATR * BreakoutATRMultiple >= 1.0)
    - SL: Banda opuesta (asimétrico)
    - Exit: Precio cruza EMA central
    - MEJORA: Volume spike filter (>1.5x average)
    
    PARÁMETROS:
    - KC_Multiplier: 2.0
    - MinChannelWidthATRRatio: 0.5-0.8
    - BreakoutATRMultiple: 1.2-1.5 (vs 1.0)
    - OBV_Signal_Period: 20
    - VolumeSpike_Multiplier: 1.5 (nuevo)
    """
```

**EA #5: FxEmaScalper** (Fuzzy Slope Scalper)
```python
# File: underdog/strategies/ea_ema_scalper.py
# Prioridad: MEDIA-ALTA (Scalping)
# Estimación: 450 líneas

class FxEmaScalper:
    """
    EMA Slope + Fuzzy Logic (trapezoidal membership).
    
    LÓGICA:
    - Entry: Pendiente EMA en zona óptima (fuzzy >= 0.5)
    - CRITICAL FIX: Normalizar slope por ATR
      slope_normalized = slope_pips_per_bar / current_ATR
    - SL: Fijo 10 pips
    - TP: Fijo 15 pips (R:R 1:1.5)
    
    PARÁMETROS:
    - EMA_Period: 50
    - Slope_Pips_A: 1.0 → Convertir a ATR ratio (0.3x)
    - Slope_Pips_B: 3.0 → (0.8x)
    - Slope_Pips_C: 10.0 → (2.0x)
    - Slope_Pips_D: 20.0 → (4.0x)
    - Fuzzy_Threshold_Open: 0.5
    """
```

**EA #6: FxBollingerCCI** (Fuzzy Mean Reversion)
```python
# File: underdog/strategies/ea_bollinger_cci.py
# Prioridad: MEDIA
# Estimación: 450 líneas

class FxBollingerCCI:
    """
    Bollinger Bands + CCI extremes + Fuzzy Logic.
    
    LÓGICA:
    - Entry: Precio fuera BB + CCI extremo (>100 o <-100)
    - Fuzzy: MIN(u_proximity, u_cci_extreme) >= 0.5
    - MEJORA: Confidence-Weighted Sizing
      signal_strength 0.95 → risk 2.0%
      signal_strength 0.51 → risk 0.5%
    - SL: 50 pips
    - TP: 30 pips (R:R 1:0.6)
    
    PARÁMETROS:
    - BB_Period: 20
    - BB_Deviation: 2.0
    - CCI_Period: 14
    - Fuzzy_CCI_Min: 100
    - Fuzzy_CCI_Max: 250
    - Fuzzy_Threshold_High: 0.5
    - RiskPerTrade_Base: 1.5%
    - Risk_Modulator: 0.5 (nuevo)
    """
```

**EA #7: FXATRBreakout** (ATR Spike Fuzzy)
```python
# File: underdog/strategies/ea_atr_breakout.py
# Prioridad: MEDIA-ALTA
# Estimación: 450 líneas

class FXATRBreakout:
    """
    Range breakout + ATR spike + RSI momentum.
    
    LÓGICA:
    - Entry: Precio rompe rango (4-12 bars) + ATR spike + RSI delta
    - Fuzzy: MIN(ATR_ratio > 1.5, RSI_delta) >= 0.6
    - MEJORA: Re-entry on false breakout
      Si SL hit + reversal pattern → trade opuesto
    - SL: ATR * 2.0
    - TP: ATR * 3.0 (R:R 1:1.5)
    
    PARÁMETROS:
    - Breakout_Rango_Lookback: 8-12 (vs 4)
    - RSI_Period: 14
    - Fuzzy_ATR_Ratio_High: 2.5
    - Fuzzy_RSI_Delta_High: 30
    - Fuzzy_Threshold_Open: 0.6
    - FalseBreakout_ReEntry: true (nuevo)
    """
```

---

### 🔴 CRÍTICO - COMPLIANCE MODULES

#### Drawdown Flotante Monitor
```python
# Status: ✅ IMPLEMENTADO
# File: underdog/risk_management/floating_drawdown_monitor.py

# INTEGRACIÓN EN CADA EA:
def on_tick():
    # 1. CHECK DD (ANTES DE TODO)
    dd_state = dd_monitor.check_drawdown()
    
    if dd_state.is_breached:
        return  # HALT TRADING
    
    if dd_state.level == DrawdownLevel.CRITICAL:
        # Reducir riesgo a 0.5%
        risk_pct = 0.5
    elif dd_state.level == DrawdownLevel.WARNING:
        # Riesgo normal con precaución
        risk_pct = 1.0
    else:
        # Riesgo normal
        risk_pct = 1.5
    
    # 2. Proceder con lógica de trading
    ...
```

#### Fault-Tolerant Execution
```python
# Status: ✅ IMPLEMENTADO
# File: underdog/execution/fault_tolerant_executor.py

# USO EN CADA EA:
executor = FaultTolerantExecutor()

# Abrir posición con retry
result = executor.open_position(
    symbol="EURUSD",
    order_type=mt5.ORDER_TYPE_BUY,
    volume=lot_size,
    sl=sl_price,
    tp=tp_price,
    comment=f"{ea_name}_ENTRY"
)

if not result.success:
    logger.error(f"Failed after {result.attempts} attempts")
```

---

### 🎨 ARQUITECTURA DE EAs (Template)

Cada EA debe seguir esta estructura:

```python
# underdog/strategies/ea_<nombre>.py

from dataclasses import dataclass
from typing import Optional, Dict, List
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from underdog.risk_management.floating_drawdown_monitor import (
    FloatingDrawdownMonitor, DrawdownLevel
)
from underdog.execution.fault_tolerant_executor import FaultTolerantExecutor
from underdog.risk_management.position_sizing import (
    PositionSizer, confidence_weighted_sizing
)
from underdog.database.redis_cache import RedisStateCache


@dataclass
class EAConfig:
    """Configuración del EA"""
    # Parámetros de estrategia
    ...
    
    # Risk management
    risk_per_trade: float = 1.5  # %
    max_daily_dd: float = 5.0
    max_total_dd: float = 10.0
    max_floating_dd: float = 1.0
    
    # Execution
    magic_number: int = 123456
    symbol: str = "EURUSD"
    timeframe: int = mt5.TIMEFRAME_M15


class EA_Base:
    """
    Clase base para todos los EAs.
    
    ESTRUCTURA:
    1. __init__: Inicializar componentes
    2. on_tick: Lógica principal (llamada en cada tick)
    3. generate_signal: Generar señales de trading
    4. manage_positions: Gestionar posiciones abiertas
    5. calculate_size: Position sizing
    """
    
    def __init__(self, config: EAConfig):
        self.config = config
        
        # Core components
        self.dd_monitor = FloatingDrawdownMonitor()
        self.executor = FaultTolerantExecutor()
        self.sizer = PositionSizer()
        self.redis = RedisStateCache()
        
        # State
        self.positions = {}
        self.signals = []
        
        print(f"[{self.__class__.__name__}] Initialized")
    
    def on_tick(self):
        """
        Main tick handler.
        
        **CRITICAL ORDER**:
        1. Check drawdown (FIRST)
        2. Manage existing positions
        3. Generate new signals
        4. Execute entries
        """
        # 1. CHECK DD
        dd_state = self.dd_monitor.check_drawdown()
        
        if dd_state.is_breached:
            return
        
        if dd_state.level == DrawdownLevel.CRITICAL:
            # Solo gestionar posiciones, no abrir nuevas
            self.manage_positions()
            return
        
        # 2. MANAGE POSITIONS
        self.manage_positions()
        
        # 3. GENERATE SIGNALS
        signal = self.generate_signal()
        
        # 4. EXECUTE
        if signal:
            self.execute_signal(signal)
    
    def generate_signal(self) -> Optional[Dict]:
        """Generar señal de trading (OVERRIDE en cada EA)"""
        raise NotImplementedError
    
    def manage_positions(self):
        """Gestionar posiciones abiertas (trailing, exits, etc.)"""
        raise NotImplementedError
    
    def calculate_size(self, signal: Dict) -> float:
        """Calcular lotaje con confidence weighting"""
        confidence = signal.get('confidence', 1.0)
        
        # Base size (fixed fractional)
        base_size = self.sizer.calculate_size(
            account_balance=mt5.account_info().balance,
            entry_price=signal['entry_price'],
            stop_loss=signal['sl'],
            confidence_score=confidence
        )
        
        return base_size['final_size']
    
    def execute_signal(self, signal: Dict):
        """Ejecutar señal con fault-tolerant executor"""
        lot_size = self.calculate_size(signal)
        
        result = self.executor.open_position(
            symbol=self.config.symbol,
            order_type=signal['type'],
            volume=lot_size,
            sl=signal['sl'],
            tp=signal['tp'],
            comment=f"{self.__class__.__name__}_ENTRY",
            magic=self.config.magic_number
        )
        
        if result.success:
            self.positions[result.ticket] = signal
            print(f"[{self.__class__.__name__}] Position opened: #{result.ticket}")
        else:
            print(f"[{self.__class__.__name__}] Failed to open: {result.comment}")
```

---

### 🛠️ HERRAMIENTAS AUXILIARES

#### Regime Filter Híbrido
```python
# File: underdog/strategies/regime_filter.py
# Prioridad: ALTA
# Estimación: 200 líneas

class RegimeFilter:
    """
    Filtro de régimen con hysteresis logic.
    
    REGÍMENES:
    - TREND: ADX > 25 && ATR > avg*1.2
    - RANGE: ADX < 20 && BB_width < avg*0.8
    - VOLATILE: ATR > avg*1.5 && ADX < 20
    
    HYSTERESIS:
    - Entry threshold: ADX > 25
    - Exit threshold: ADX < 20
    - Evita parpadeo de régimen
    """
    
    def get_regime(self, symbol, timeframe) -> str:
        """Retorna: 'TREND', 'RANGE', 'VOLATILE'"""
        pass
    
    def should_ea_trade(self, ea_type, current_regime) -> bool:
        """
        Determinar si EA debe operar en régimen actual.
        
        REGLAS:
        - Mean Reversion EAs: NO operar si TREND
        - Breakout EAs: NO operar si RANGE estrecho
        - Trend EAs: NO operar si VOLATILE sin dirección
        """
        pass
```

---

### 📊 PRIORIDADES DE IMPLEMENTACIÓN

#### **Sprint 1: Infraestructura Crítica** (1 semana)
1. ✅ Drawdown Monitor
2. ✅ Fault-Tolerant Executor
3. [ ] DuckDB Store (para datos históricos)
4. [ ] Regime Filter

#### **Sprint 2: EAs Core** (2 semanas)
1. [ ] FxSuperTrendRSI (Trend)
2. [ ] FxKeltnerBreakout (Breakout)
3. [ ] FxParabolicEMA (Adaptive)

#### **Sprint 3: EAs Complementarios** (2 semanas)
4. [ ] FxEmaScalper (Scalping)
5. [ ] FXATRBreakout (Volatility)

#### **Sprint 4: EAs Mean Reversion** (1.5 semanas)
6. [ ] FxBollingerCCI (Fuzzy MR)
7. [ ] FXPairArbitrage (Statistical)

#### **Sprint 5: Testing & Optimization** (2 semanas)
- Backtesting de cada EA (Walk-Forward)
- Optimización de parámetros (CVaR-based)
- Monte Carlo validation
- Demo account testing

---

### 🎯 MÉTRICAS DE ÉXITO (Prop Firm Compliance)

| Métrica | Target | Crítico |
|---------|--------|---------|
| **Max DD Diario** | < 5% | < 3% |
| **Max DD Total** | < 10% | < 7% |
| **DD Flotante** | < 1% | < 0.8% |
| **Sharpe Ratio** | > 1.5 | > 2.0 |
| **Calmar Ratio** | > 2.0 | > 3.0 |
| **Win Rate** | > 55% | > 60% |
| **Profit Factor** | > 1.5 | > 2.0 |
| **CVaR (95%)** | < 2% | < 1.5% |

---

## 🚀 PRÓXIMOS PASOS INMEDIATOS

1. **Crear estructura de carpetas para EAs**
2. **Implementar DuckDB Store** (datos históricos)
3. **Desarrollar Regime Filter** (hysteresis)
4. **Comenzar con FxSuperTrendRSI** (EA más simple)
5. **Testing en demo account** (1 EA a la vez)

**Timeline Total**: ~8-10 semanas para sistema completo  
**Estado Actual**: 30% completado (infraestructura core lista)

---

**Última Actualización**: Octubre 2025  
**Responsable**: UNDERDOG Development Team
