"""
Fault-Tolerant Order Execution
Sistema de ejecución robusto con reintentos y manejo de errores.

Maneja errores transitorios comunes en MT5:
- ERR_REQUOTE (10004): Requote - precio cambió
- ERR_OFF_QUOTES (136): Off quotes - sin cotización
- ERR_PRICE_CHANGED (135): Precio cambió durante ejecución
- ERR_TRADE_TIMEOUT (10038): Timeout de trade
- ERR_TRADE_MODIFY_DENIED (10025): Modificación denegada

**PROTOCOL**:
1. Retry hasta MaxRetries (3-5x)
2. Sleep entre intentos (300ms)
3. Slippage dinámico (incrementar en cada retry)
4. Logging completo de errores
"""
import MetaTrader5 as mt5
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from enum import Enum
import time
from datetime import datetime


class OrderAction(Enum):
    """Tipos de acción de orden"""
    OPEN = "open"
    CLOSE = "close"
    MODIFY = "modify"


@dataclass
class RetryConfig:
    """Configuración de reintentos"""
    max_retries: int = 5                # Máximo de reintentos
    sleep_ms: int = 300                 # Pausa entre reintentos (ms)
    base_deviation: int = 20            # Slippage base (pips)
    deviation_increment: int = 10       # Incremento de slippage por retry
    max_deviation: int = 50             # Slippage máximo permitido
    retry_on_errors: list = None        # Códigos de error que gatillan retry
    
    def __post_init__(self):
        if self.retry_on_errors is None:
            # Errores transitorios que justifican retry
            self.retry_on_errors = [
                10004,  # ERR_REQUOTE
                136,    # ERR_OFF_QUOTES
                135,    # ERR_PRICE_CHANGED
                10038,  # ERR_TRADE_TIMEOUT
                10006,  # ERR_TRADE_INVALID_VOLUME
                10018,  # ERR_MARKET_CLOSED
            ]


@dataclass
class OrderResult:
    """Resultado de ejecución de orden"""
    success: bool
    ticket: Optional[int] = None
    retcode: Optional[int] = None
    comment: Optional[str] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    attempts: int = 1
    error_history: list = None
    
    def __post_init__(self):
        if self.error_history is None:
            self.error_history = []


class FaultTolerantExecutor:
    """
    Ejecutor de órdenes con tolerancia a fallos.
    
    **USO**:
    ```python
    executor = FaultTolerantExecutor()
    
    # Abrir posición
    result = executor.open_position(
        symbol="EURUSD",
        order_type=mt5.ORDER_TYPE_BUY,
        volume=0.10,
        sl=1.0950,
        tp=1.1100,
        comment="EA_Entry"
    )
    
    if result.success:
        print(f"Position opened: #{result.ticket}")
    else:
        print(f"Failed after {result.attempts} attempts: {result.comment}")
    ```
    """
    
    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        logger: Optional[Callable] = None
    ):
        """
        Inicializar executor.
        
        Args:
            config: Configuración de reintentos
            logger: Función de logging personalizada
        """
        self.config = config or RetryConfig()
        self.logger = logger or print
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'total_successes': 0,
            'total_failures': 0,
            'retries_used': 0
        }
        
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
        
        self.logger("[Executor] Fault-tolerant executor initialized")
    
    def _should_retry(self, retcode: int) -> bool:
        """Verificar si error justifica retry"""
        return retcode in self.config.retry_on_errors
    
    def _calculate_deviation(self, attempt: int) -> int:
        """Calcular slippage dinámico por intento"""
        deviation = self.config.base_deviation + (attempt - 1) * self.config.deviation_increment
        return min(deviation, self.config.max_deviation)
    
    def _get_fill_type(self, symbol: str) -> int:
        """Determinar tipo de llenado según símbolo"""
        symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            return mt5.ORDER_FILLING_RETURN
        
        # Check filling modes
        filling_mode = symbol_info.filling_mode
        
        if filling_mode & mt5.SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK  # Fill or Kill
        elif filling_mode & mt5.SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC  # Immediate or Cancel
        else:
            return mt5.ORDER_FILLING_RETURN
    
    def open_position(
        self,
        symbol: str,
        order_type: int,  # mt5.ORDER_TYPE_BUY or ORDER_TYPE_SELL
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "",
        magic: int = 0
    ) -> OrderResult:
        """
        Abrir posición con retry logic.
        
        Args:
            symbol: Símbolo (ej. "EURUSD")
            order_type: mt5.ORDER_TYPE_BUY o ORDER_TYPE_SELL
            volume: Volumen en lotes
            sl: Stop Loss (opcional)
            tp: Take Profit (opcional)
            comment: Comentario
            magic: Magic number
        
        Returns:
            OrderResult con resultado de la operación
        """
        error_history = []
        
        for attempt in range(1, self.config.max_retries + 1):
            self.stats['total_attempts'] += 1
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            
            if tick is None:
                error_msg = f"Cannot get tick for {symbol}"
                error_history.append((attempt, 0, error_msg))
                
                if attempt < self.config.max_retries:
                    time.sleep(self.config.sleep_ms / 1000.0)
                    continue
                else:
                    self.stats['total_failures'] += 1
                    return OrderResult(
                        success=False,
                        comment=error_msg,
                        attempts=attempt,
                        error_history=error_history
                    )
            
            # Set price based on order type
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            
            # Calculate dynamic deviation
            deviation = self._calculate_deviation(attempt)
            
            # Get fill type
            type_filling = self._get_fill_type(symbol)
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl if sl else 0.0,
                "tp": tp if tp else 0.0,
                "deviation": deviation,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": type_filling,
            }
            
            # Log attempt
            if attempt > 1:
                self.logger(f"[Executor] Retry {attempt}/{self.config.max_retries} "
                          f"(deviation: {deviation} pips)")
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                error_msg = "order_send returned None"
                error_history.append((attempt, 0, error_msg))
                
                if attempt < self.config.max_retries:
                    time.sleep(self.config.sleep_ms / 1000.0)
                    continue
                else:
                    self.stats['total_failures'] += 1
                    return OrderResult(
                        success=False,
                        comment=error_msg,
                        attempts=attempt,
                        error_history=error_history
                    )
            
            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # SUCCESS
                self.stats['total_successes'] += 1
                if attempt > 1:
                    self.stats['retries_used'] += attempt - 1
                
                return OrderResult(
                    success=True,
                    ticket=result.order,
                    retcode=result.retcode,
                    comment=result.comment,
                    price=result.price,
                    volume=result.volume,
                    attempts=attempt,
                    error_history=error_history
                )
            else:
                # ERROR
                error_msg = f"{result.retcode}: {result.comment}"
                error_history.append((attempt, result.retcode, error_msg))
                
                self.logger(f"[Executor] Attempt {attempt} failed: {error_msg}")
                
                # Check if should retry
                if self._should_retry(result.retcode) and attempt < self.config.max_retries:
                    # Sleep before retry
                    time.sleep(self.config.sleep_ms / 1000.0)
                    continue
                else:
                    # Give up
                    self.stats['total_failures'] += 1
                    return OrderResult(
                        success=False,
                        retcode=result.retcode,
                        comment=result.comment,
                        attempts=attempt,
                        error_history=error_history
                    )
        
        # Should not reach here
        self.stats['total_failures'] += 1
        return OrderResult(
            success=False,
            comment="Max retries exceeded",
            attempts=self.config.max_retries,
            error_history=error_history
        )
    
    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        comment: str = "EA_Close"
    ) -> OrderResult:
        """
        Cerrar posición con retry logic.
        
        Args:
            ticket: Ticket de la posición
            volume: Volumen a cerrar (None = cerrar todo)
            comment: Comentario
        
        Returns:
            OrderResult
        """
        # Get position info
        position = mt5.positions_get(ticket=ticket)
        
        if position is None or len(position) == 0:
            return OrderResult(
                success=False,
                comment=f"Position #{ticket} not found",
                attempts=1
            )
        
        position = position[0]
        
        # Determine close parameters
        symbol = position.symbol
        close_volume = volume if volume else position.volume
        
        # Opposite order type
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
        else:
            order_type = mt5.ORDER_TYPE_BUY
        
        error_history = []
        
        for attempt in range(1, self.config.max_retries + 1):
            self.stats['total_attempts'] += 1
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            
            if tick is None:
                error_msg = f"Cannot get tick for {symbol}"
                error_history.append((attempt, 0, error_msg))
                
                if attempt < self.config.max_retries:
                    time.sleep(self.config.sleep_ms / 1000.0)
                    continue
                else:
                    self.stats['total_failures'] += 1
                    return OrderResult(
                        success=False,
                        comment=error_msg,
                        attempts=attempt,
                        error_history=error_history
                    )
            
            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
            deviation = self._calculate_deviation(attempt)
            type_filling = self._get_fill_type(symbol)
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": close_volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": deviation,
                "magic": position.magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": type_filling,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                error_msg = "order_send returned None"
                error_history.append((attempt, 0, error_msg))
                
                if attempt < self.config.max_retries:
                    time.sleep(self.config.sleep_ms / 1000.0)
                    continue
                else:
                    self.stats['total_failures'] += 1
                    return OrderResult(
                        success=False,
                        comment=error_msg,
                        attempts=attempt,
                        error_history=error_history
                    )
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.stats['total_successes'] += 1
                if attempt > 1:
                    self.stats['retries_used'] += attempt - 1
                
                return OrderResult(
                    success=True,
                    ticket=ticket,
                    retcode=result.retcode,
                    comment=result.comment,
                    price=result.price,
                    volume=result.volume,
                    attempts=attempt,
                    error_history=error_history
                )
            else:
                error_msg = f"{result.retcode}: {result.comment}"
                error_history.append((attempt, result.retcode, error_msg))
                
                self.logger(f"[Executor] Close attempt {attempt} failed: {error_msg}")
                
                if self._should_retry(result.retcode) and attempt < self.config.max_retries:
                    time.sleep(self.config.sleep_ms / 1000.0)
                    continue
                else:
                    self.stats['total_failures'] += 1
                    return OrderResult(
                        success=False,
                        retcode=result.retcode,
                        comment=result.comment,
                        attempts=attempt,
                        error_history=error_history
                    )
        
        self.stats['total_failures'] += 1
        return OrderResult(
            success=False,
            comment="Max retries exceeded",
            attempts=self.config.max_retries,
            error_history=error_history
        )
    
    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> OrderResult:
        """
        Modificar SL/TP de posición con retry logic.
        
        Args:
            ticket: Ticket de la posición
            sl: Nuevo Stop Loss (None = no cambiar)
            tp: Nuevo Take Profit (None = no cambiar)
        
        Returns:
            OrderResult
        """
        position = mt5.positions_get(ticket=ticket)
        
        if position is None or len(position) == 0:
            return OrderResult(
                success=False,
                comment=f"Position #{ticket} not found",
                attempts=1
            )
        
        position = position[0]
        symbol = position.symbol
        
        # Use current values if not provided
        new_sl = sl if sl is not None else position.sl
        new_tp = tp if tp is not None else position.tp
        
        error_history = []
        
        for attempt in range(1, self.config.max_retries + 1):
            self.stats['total_attempts'] += 1
            
            # Prepare modify request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": new_tp,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                error_msg = "order_send returned None"
                error_history.append((attempt, 0, error_msg))
                
                if attempt < self.config.max_retries:
                    time.sleep(self.config.sleep_ms / 1000.0)
                    continue
                else:
                    self.stats['total_failures'] += 1
                    return OrderResult(
                        success=False,
                        comment=error_msg,
                        attempts=attempt,
                        error_history=error_history
                    )
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.stats['total_successes'] += 1
                if attempt > 1:
                    self.stats['retries_used'] += attempt - 1
                
                return OrderResult(
                    success=True,
                    ticket=ticket,
                    retcode=result.retcode,
                    comment=result.comment,
                    attempts=attempt,
                    error_history=error_history
                )
            else:
                error_msg = f"{result.retcode}: {result.comment}"
                error_history.append((attempt, result.retcode, error_msg))
                
                self.logger(f"[Executor] Modify attempt {attempt} failed: {error_msg}")
                
                if self._should_retry(result.retcode) and attempt < self.config.max_retries:
                    time.sleep(self.config.sleep_ms / 1000.0)
                    continue
                else:
                    self.stats['total_failures'] += 1
                    return OrderResult(
                        success=False,
                        retcode=result.retcode,
                        comment=result.comment,
                        attempts=attempt,
                        error_history=error_history
                    )
        
        self.stats['total_failures'] += 1
        return OrderResult(
            success=False,
            comment="Max retries exceeded",
            attempts=self.config.max_retries,
            error_history=error_history
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de ejecución"""
        total = self.stats['total_attempts']
        
        return {
            **self.stats,
            'success_rate': (self.stats['total_successes'] / total * 100) if total > 0 else 0.0,
            'avg_retries': (self.stats['retries_used'] / self.stats['total_successes']) if self.stats['total_successes'] > 0 else 0.0
        }
    
    def print_stats(self):
        """Imprimir estadísticas"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print(" EXECUTOR STATISTICS")
        print("="*60)
        print(f"Total Attempts:    {stats['total_attempts']}")
        print(f"Successes:         {stats['total_successes']}")
        print(f"Failures:          {stats['total_failures']}")
        print(f"Success Rate:      {stats['success_rate']:.1f}%")
        print(f"Retries Used:      {stats['retries_used']}")
        print(f"Avg Retries:       {stats['avg_retries']:.2f}")
        print("="*60)


# ====================================
# EJEMPLO DE USO
# ====================================

def example_usage():
    """Ejemplo de uso del executor"""
    
    # Configurar retry
    config = RetryConfig(
        max_retries=5,
        sleep_ms=300,
        base_deviation=20,
        deviation_increment=10,
        max_deviation=50
    )
    
    # Crear executor
    executor = FaultTolerantExecutor(config=config)
    
    print("\n" + "="*60)
    print(" EJEMPLO: Fault-Tolerant Order Execution")
    print("="*60)
    
    # 1. Abrir posición
    print("\n1. Opening BUY position on EURUSD...")
    result = executor.open_position(
        symbol="EURUSD",
        order_type=mt5.ORDER_TYPE_BUY,
        volume=0.01,
        sl=1.0950,
        tp=1.1100,
        comment="TEST_ENTRY"
    )
    
    if result.success:
        print(f"✅ Position opened: #{result.ticket}")
        print(f"   Price: {result.price}")
        print(f"   Attempts: {result.attempts}")
        
        # 2. Modificar SL/TP
        print(f"\n2. Modifying SL/TP...")
        modify_result = executor.modify_position(
            ticket=result.ticket,
            sl=1.0970,  # Move SL
            tp=1.1150   # Move TP
        )
        
        if modify_result.success:
            print(f"✅ Position modified")
            print(f"   Attempts: {modify_result.attempts}")
        else:
            print(f"❌ Modify failed: {modify_result.comment}")
        
        # 3. Cerrar posición
        print(f"\n3. Closing position...")
        close_result = executor.close_position(
            ticket=result.ticket,
            comment="TEST_CLOSE"
        )
        
        if close_result.success:
            print(f"✅ Position closed")
            print(f"   Close price: {close_result.price}")
            print(f"   Attempts: {close_result.attempts}")
        else:
            print(f"❌ Close failed: {close_result.comment}")
    else:
        print(f"❌ Failed to open position: {result.comment}")
        print(f"   Attempts: {result.attempts}")
        print(f"   Error history:")
        for attempt, code, msg in result.error_history:
            print(f"     Attempt {attempt}: {msg}")
    
    # Estadísticas
    executor.print_stats()


if __name__ == "__main__":
    try:
        example_usage()
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nNOTE: Este ejemplo requiere MetaTrader 5 running y conectado")
