"""
Floating Drawdown Monitor - CRITICAL COMPLIANCE MODULE
Sistema de monitoreo en tiempo real de drawdown flotante para Prop Firms.

**CRÍTICO**: Los Prop Firms (especialmente Instant Funding) aplican breach
en tiempo real basado en drawdown FLOTANTE, no solo al cierre.

Reglas típicas:
- Daily DD: 5% del balance inicial del día
- Total DD: 10% del balance inicial de la cuenta
- Instant Funding: Breach si equity < balance_inicial - 1.0%

El FXP_Manager externo puede no reaccionar con latencia suficiente.
Por tanto, el EA debe tener su propia lógica de emergency shutdown.
"""
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, Dict, Callable
import MetaTrader5 as mt5
from enum import Enum


class DrawdownLevel(Enum):
    """Niveles de severidad del drawdown"""
    SAFE = "safe"           # DD < 50% del límite
    WARNING = "warning"     # DD entre 50-80% del límite
    CRITICAL = "critical"   # DD entre 80-95% del límite
    BREACH = "breach"       # DD >= límite → CERRAR TODO


@dataclass
class DrawdownLimits:
    """Límites de drawdown configurables por Prop Firm"""
    # Daily limits
    max_daily_dd_pct: float = 5.0      # % del balance inicial del día
    max_daily_dd_absolute: Optional[float] = None  # USD absolutos (opcional)
    
    # Total limits
    max_total_dd_pct: float = 10.0     # % del balance inicial de la cuenta
    max_total_dd_absolute: Optional[float] = None
    
    # Instant Funding special (más estricto)
    max_floating_dd_pct: float = 1.0   # % para breach inmediato
    
    # Safety margins
    warning_threshold_pct: float = 0.5  # Warning al 50% del límite
    critical_threshold_pct: float = 0.8  # Critical al 80%
    
    # Reset time
    reset_hour: int = 0  # Hora GMT de reset diario (00:00)
    reset_minute: int = 0


@dataclass
class DrawdownState:
    """Estado actual del drawdown"""
    timestamp: datetime
    
    # Account info
    equity: float
    balance: float
    
    # Daily tracking
    daily_start_balance: float
    daily_dd_absolute: float
    daily_dd_pct: float
    
    # Total tracking
    account_start_balance: float
    total_dd_absolute: float
    total_dd_pct: float
    
    # Floating DD (más estricto)
    floating_dd_absolute: float
    floating_dd_pct: float
    
    # Status
    level: DrawdownLevel
    is_breached: bool
    
    def to_dict(self) -> Dict:
        """Convertir a dict para logging/Redis"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'equity': self.equity,
            'balance': self.balance,
            'daily_dd_pct': self.daily_dd_pct,
            'total_dd_pct': self.total_dd_pct,
            'floating_dd_pct': self.floating_dd_pct,
            'level': self.level.value,
            'is_breached': self.is_breached
        }


class FloatingDrawdownMonitor:
    """
    Monitor de drawdown flotante en tiempo real.
    
    **USO EN OnTick()**:
    ```python
    def on_tick():
        # 1. Check drawdown ANTES de abrir posiciones
        dd_state = dd_monitor.check_drawdown()
        
        if dd_state.is_breached:
            # EMERGENCY SHUTDOWN
            dd_monitor.emergency_close_all()
            return  # NO operar más
        
        if dd_state.level == DrawdownLevel.CRITICAL:
            # Reducir riesgo o pausar trading
            return
        
        # 2. Proceder con lógica de trading
        ...
    ```
    
    **INTEGRACIÓN CON REDIS**:
    ```python
    # En cada check, actualizar Redis
    redis_cache.update_drawdown(dd_state.to_dict())
    ```
    """
    
    def __init__(
        self,
        limits: Optional[DrawdownLimits] = None,
        redis_cache = None,
        emergency_callback: Optional[Callable] = None
    ):
        """
        Inicializar monitor.
        
        Args:
            limits: Límites de drawdown
            redis_cache: RedisStateCache instance (opcional)
            emergency_callback: Función a ejecutar en breach (opcional)
        """
        self.limits = limits or DrawdownLimits()
        self.redis_cache = redis_cache
        self.emergency_callback = emergency_callback
        
        # State tracking
        self.daily_start_balance: Optional[float] = None
        self.account_start_balance: Optional[float] = None
        self.last_reset_date: Optional[datetime] = None
        
        # Breach flag (persistent)
        self.is_breached = False
        
        # Initialize MT5
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
        
        # Get initial balances
        self._initialize_balances()
        
        print("[DD Monitor] Initialized")
        print(f"  Daily DD Limit: {self.limits.max_daily_dd_pct:.1f}%")
        print(f"  Total DD Limit: {self.limits.max_total_dd_pct:.1f}%")
        print(f"  Floating DD Limit: {self.limits.max_floating_dd_pct:.1f}%")
    
    def _initialize_balances(self):
        """Inicializar balances de referencia"""
        account_info = mt5.account_info()
        
        if account_info is None:
            raise RuntimeError("Cannot get account info")
        
        current_balance = account_info.balance
        
        # Account start balance (primera vez o desde configuración)
        if self.account_start_balance is None:
            self.account_start_balance = current_balance
        
        # Daily start balance
        self.daily_start_balance = current_balance
        self.last_reset_date = datetime.now().date()
        
        print(f"[DD Monitor] Balances initialized:")
        print(f"  Account Start: ${self.account_start_balance:,.2f}")
        print(f"  Daily Start: ${self.daily_start_balance:,.2f}")
    
    def _check_daily_reset(self):
        """Verificar si necesita reset diario"""
        now = datetime.now()
        current_date = now.date()
        current_time = now.time()
        
        # Check si pasó la hora de reset
        reset_time = time(self.limits.reset_hour, self.limits.reset_minute)
        
        if (self.last_reset_date != current_date and 
            current_time >= reset_time):
            # Reset daily balance
            account_info = mt5.account_info()
            self.daily_start_balance = account_info.balance
            self.last_reset_date = current_date
            
            print(f"[DD Monitor] Daily reset executed at {now}")
            print(f"  New daily start balance: ${self.daily_start_balance:,.2f}")
    
    def check_drawdown(self) -> DrawdownState:
        """
        Verificar drawdown actual (LLAMAR EN CADA TICK).
        
        Returns:
            DrawdownState con métricas actuales
        """
        # Check daily reset
        self._check_daily_reset()
        
        # Get current account state
        account_info = mt5.account_info()
        
        if account_info is None:
            raise RuntimeError("Cannot get account info")
        
        equity = account_info.equity
        balance = account_info.balance
        
        # Calculate daily DD
        daily_dd_abs = self.daily_start_balance - equity
        daily_dd_pct = (daily_dd_abs / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0.0
        
        # Calculate total DD
        total_dd_abs = self.account_start_balance - equity
        total_dd_pct = (total_dd_abs / self.account_start_balance) * 100 if self.account_start_balance > 0 else 0.0
        
        # Calculate floating DD (más estricto: equity vs balance actual)
        floating_dd_abs = balance - equity
        floating_dd_pct = (floating_dd_abs / balance) * 100 if balance > 0 else 0.0
        
        # Determine severity level
        level, is_breached = self._determine_level(
            daily_dd_pct,
            total_dd_pct,
            floating_dd_pct
        )
        
        # Create state
        state = DrawdownState(
            timestamp=datetime.now(),
            equity=equity,
            balance=balance,
            daily_start_balance=self.daily_start_balance,
            daily_dd_absolute=daily_dd_abs,
            daily_dd_pct=daily_dd_pct,
            account_start_balance=self.account_start_balance,
            total_dd_absolute=total_dd_abs,
            total_dd_pct=total_dd_pct,
            floating_dd_absolute=floating_dd_abs,
            floating_dd_pct=floating_dd_pct,
            level=level,
            is_breached=is_breached
        )
        
        # Update Redis cache
        if self.redis_cache:
            self.redis_cache.update_equity(equity)
            # Cache DD state
            dd_dict = state.to_dict()
            self.redis_cache.client.hmset("drawdown:current", dd_dict)
        
        # Handle breach
        if is_breached and not self.is_breached:
            self._handle_breach(state)
            self.is_breached = True
        
        return state
    
    def _determine_level(
        self,
        daily_dd: float,
        total_dd: float,
        floating_dd: float
    ) -> tuple[DrawdownLevel, bool]:
        """
        Determinar nivel de severidad.
        
        Returns:
            (level, is_breached)
        """
        # Check breach conditions (ANY breach triggers shutdown)
        if daily_dd >= self.limits.max_daily_dd_pct:
            return DrawdownLevel.BREACH, True
        
        if total_dd >= self.limits.max_total_dd_pct:
            return DrawdownLevel.BREACH, True
        
        if floating_dd >= self.limits.max_floating_dd_pct:
            return DrawdownLevel.BREACH, True
        
        # Calculate % of limit used (worst case)
        daily_usage = daily_dd / self.limits.max_daily_dd_pct
        total_usage = total_dd / self.limits.max_total_dd_pct
        floating_usage = floating_dd / self.limits.max_floating_dd_pct
        
        max_usage = max(daily_usage, total_usage, floating_usage)
        
        # Determine level
        if max_usage >= self.limits.critical_threshold_pct:
            return DrawdownLevel.CRITICAL, False
        elif max_usage >= self.limits.warning_threshold_pct:
            return DrawdownLevel.WARNING, False
        else:
            return DrawdownLevel.SAFE, False
    
    def _handle_breach(self, state: DrawdownState):
        """
        Manejar breach de drawdown.
        
        **CRITICAL**: Cerrar todas las posiciones inmediatamente
        """
        print("="*80)
        print(" 🚨 DRAWDOWN BREACH DETECTED 🚨")
        print("="*80)
        print(f"Timestamp: {state.timestamp}")
        print(f"Daily DD: {state.daily_dd_pct:.2f}% (Limit: {self.limits.max_daily_dd_pct:.1f}%)")
        print(f"Total DD: {state.total_dd_pct:.2f}% (Limit: {self.limits.max_total_dd_pct:.1f}%)")
        print(f"Floating DD: {state.floating_dd_pct:.2f}% (Limit: {self.limits.max_floating_dd_pct:.1f}%)")
        print(f"Equity: ${state.equity:,.2f}")
        print(f"Balance: ${state.balance:,.2f}")
        print("="*80)
        
        # Emergency close all positions
        self.emergency_close_all()
        
        # Execute callback
        if self.emergency_callback:
            self.emergency_callback(state)
        
        # Log to Redis
        if self.redis_cache:
            self.redis_cache.client.lpush(
                "drawdown:breaches",
                str(state.to_dict())
            )
            self.redis_cache.set_system_state("BREACHED")
    
    def emergency_close_all(self):
        """
        EMERGENCY: Cerrar todas las posiciones abiertas.
        
        **CRITICAL FUNCTION**
        """
        print("[DD Monitor] EMERGENCY CLOSE ALL POSITIONS")
        
        positions = mt5.positions_get()
        
        if positions is None or len(positions) == 0:
            print("[DD Monitor] No open positions")
            return
        
        for position in positions:
            symbol = position.symbol
            ticket = position.ticket
            volume = position.volume
            position_type = position.type
            
            # Determine close type
            if position_type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 0,
                "comment": "DD_BREACH_EMERGENCY_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"  ERROR closing {symbol} #{ticket}: {result.comment}")
            else:
                print(f"  ✅ Closed {symbol} #{ticket} - {volume} lots")
    
    def get_current_state(self) -> Optional[DrawdownState]:
        """Get current DD state sin side effects"""
        try:
            return self.check_drawdown()
        except Exception as e:
            print(f"[DD Monitor] Error getting state: {e}")
            return None
    
    def reset_breach_flag(self):
        """
        Reset breach flag (solo después de resolver manualmente).
        
        **WARNING**: Solo usar después de confirmar que el problema
        fue resuelto y la cuenta está en orden.
        """
        self.is_breached = False
        print("[DD Monitor] Breach flag reset")


# ====================================
# EJEMPLO DE USO EN EA
# ====================================

def example_ea_integration():
    """
    Ejemplo de integración en un EA Python.
    
    Este código debe ejecutarse en CADA TICK.
    """
    from underdog.database.redis_cache import RedisStateCache
    
    # Configurar límites (Instant Funding - más estricto)
    limits = DrawdownLimits(
        max_daily_dd_pct=5.0,      # 5% diario
        max_total_dd_pct=10.0,     # 10% total
        max_floating_dd_pct=1.0,   # 1% flotante (CRÍTICO)
        warning_threshold_pct=0.5,  # Warning al 50%
        critical_threshold_pct=0.8  # Critical al 80%
    )
    
    # Inicializar con Redis
    redis_cache = RedisStateCache()
    
    def emergency_callback(state: DrawdownState):
        """Callback ejecutado en breach"""
        # Enviar alerta por email/Telegram
        print("📧 Sending breach alert to admin...")
        # Pausar todos los EAs
        redis_cache.set_system_state("PAUSED")
    
    # Crear monitor
    dd_monitor = FloatingDrawdownMonitor(
        limits=limits,
        redis_cache=redis_cache,
        emergency_callback=emergency_callback
    )
    
    # Simular OnTick loop
    print("\n" + "="*60)
    print(" EJEMPLO: EA OnTick() Loop con DD Monitor")
    print("="*60)
    
    for tick in range(5):
        print(f"\n--- Tick #{tick+1} ---")
        
        # 1. CHECK DRAWDOWN (CRÍTICO - ANTES DE TODO)
        dd_state = dd_monitor.check_drawdown()
        
        print(f"Level: {dd_state.level.value.upper()}")
        print(f"Daily DD: {dd_state.daily_dd_pct:.2f}%")
        print(f"Floating DD: {dd_state.floating_dd_pct:.2f}%")
        
        # 2. HANDLE DD LEVEL
        if dd_state.is_breached:
            print("❌ BREACHED - TRADING HALTED")
            break
        
        if dd_state.level == DrawdownLevel.CRITICAL:
            print("⚠️ CRITICAL - Reduced risk mode")
            # Reducir lotaje, pausar nuevas entradas
            continue
        
        if dd_state.level == DrawdownLevel.WARNING:
            print("⚡ WARNING - Caution mode")
            # Operar con precaución
        
        # 3. PROCEDER CON LÓGICA DE TRADING
        print("✅ SAFE - Normal trading")
        # ... Lógica de señales, entries, exits ...


if __name__ == "__main__":
    print("Floating Drawdown Monitor - Compliance Module for Prop Firms")
    print("="*80)
    
    try:
        example_ea_integration()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nNOTE: Este ejemplo requiere:")
        print("  1. MetaTrader 5 running")
        print("  2. Redis server running")
        print("  3. Valid MT5 account connection")
