"""
Redis State Cache
Gestor de estado en memoria para recuperación rápida ante fallos.

Cache crítico para:
- Drawdown flotante en tiempo real
- Posiciones abiertas
- Capital actual (equity)
- Métricas de sesión

Ventajas:
- Recuperación < 1ms
- Persistencia opcional (RDB snapshots)
- Atomic operations para consistencia
"""
import redis
from typing import Dict, Optional, List
from datetime import datetime
import json
from dataclasses import dataclass, asdict


@dataclass
class PositionState:
    """Estado de posición abierta"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    entry_time: str  # ISO format
    size: float
    stop_loss: float
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    strategy_id: str = "unknown"


class RedisStateCache:
    """
    Cache de estado en memoria para recuperación ante fallos.
    
    Funcionalidad crítica para Prop Firms:
    - Recuperar estado completo después de crash de Python
    - Monitorizar drawdown flotante en tiempo real
    - Auditoría de capital y riesgo
    
    Example:
        >>> cache = RedisStateCache()
        >>> cache.update_equity(100000.0)
        >>> cache.cache_position("EURUSD", position)
        >>> # Después de crash...
        >>> equity = cache.get_equity()
        >>> positions = cache.get_all_positions()
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        """
        Conectar a Redis.
        
        Args:
            host: Redis host (default: localhost)
            port: Redis port (default: 6379)
            db: Redis database number (0-15)
            password: Redis password (si está configurado)
        """
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.client.ping()
            print(f"[Redis] Connected to {host}:{port} (db={db})")
            
        except redis.ConnectionError as e:
            print(f"[Redis] ERROR: Cannot connect to Redis at {host}:{port}")
            print(f"[Redis] Make sure Redis is running: docker run -d -p 6379:6379 redis:alpine")
            raise e
    
    # ====================================
    # EQUITY & CAPITAL MANAGEMENT
    # ====================================
    
    def update_equity(self, equity: float) -> None:
        """
        Actualizar equity actual.
        
        Args:
            equity: Capital total (cash + unrealized P&L)
        """
        timestamp = datetime.now().isoformat()
        
        # Update current equity
        self.client.set("equity:current", equity)
        self.client.set("equity:timestamp", timestamp)
        
        # Update peak equity (for drawdown calculation)
        peak = float(self.client.get("equity:peak") or 0)
        if equity > peak:
            self.client.set("equity:peak", equity)
            self.client.set("equity:peak_timestamp", timestamp)
        
        # Add to time series (for equity curve)
        self.client.zadd(
            "equity:history",
            {timestamp: equity}
        )
    
    def get_equity(self) -> Optional[float]:
        """Obtener equity actual"""
        equity = self.client.get("equity:current")
        return float(equity) if equity else None
    
    def get_drawdown(self) -> Dict[str, float]:
        """
        Calcular drawdown flotante actual.
        
        Returns:
            Dict con drawdown absoluto y porcentual
        """
        equity = self.get_equity() or 0.0
        peak = float(self.client.get("equity:peak") or equity)
        
        if peak == 0:
            return {"drawdown_abs": 0.0, "drawdown_pct": 0.0}
        
        dd_abs = peak - equity
        dd_pct = (dd_abs / peak) * 100
        
        return {
            "drawdown_abs": dd_abs,
            "drawdown_pct": dd_pct,
            "peak_equity": peak,
            "current_equity": equity
        }
    
    def get_equity_curve(self, limit: int = 1000) -> List[Dict]:
        """
        Obtener equity curve reciente.
        
        Args:
            limit: Número máximo de puntos
        
        Returns:
            Lista de {timestamp, equity}
        """
        # Get last N entries from sorted set
        history = self.client.zrange(
            "equity:history",
            -limit,  # Last N items
            -1,
            withscores=True
        )
        
        return [
            {"timestamp": ts, "equity": float(equity)}
            for ts, equity in history
        ]
    
    # ====================================
    # POSITION MANAGEMENT
    # ====================================
    
    def cache_position(self, symbol: str, position: PositionState) -> None:
        """
        Cachear posición abierta.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            position: PositionState object
        """
        position_json = json.dumps(asdict(position))
        self.client.hset("positions:open", symbol, position_json)
        
        # Update position count
        self.client.set("positions:count", self.client.hlen("positions:open"))
    
    def get_position(self, symbol: str) -> Optional[PositionState]:
        """Obtener posición específica"""
        position_json = self.client.hget("positions:open", symbol)
        
        if position_json:
            position_dict = json.loads(position_json)
            return PositionState(**position_dict)
        
        return None
    
    def get_all_positions(self) -> Dict[str, PositionState]:
        """
        Recuperar todas las posiciones abiertas.
        
        Returns:
            Dict mapping symbol → PositionState
        """
        positions_raw = self.client.hgetall("positions:open")
        
        positions = {}
        for symbol, position_json in positions_raw.items():
            position_dict = json.loads(position_json)
            positions[symbol] = PositionState(**position_dict)
        
        return positions
    
    def remove_position(self, symbol: str) -> None:
        """Eliminar posición cerrada"""
        self.client.hdel("positions:open", symbol)
        self.client.set("positions:count", self.client.hlen("positions:open"))
    
    def get_position_count(self) -> int:
        """Número de posiciones abiertas"""
        count = self.client.get("positions:count")
        return int(count) if count else 0
    
    # ====================================
    # SESSION METRICS
    # ====================================
    
    def update_session_metrics(self, metrics: Dict) -> None:
        """
        Actualizar métricas de sesión.
        
        Args:
            metrics: Dict con métricas (win_rate, total_trades, etc.)
        """
        for key, value in metrics.items():
            self.client.hset("session:metrics", key, value)
        
        self.client.hset("session:metrics", "last_update", datetime.now().isoformat())
    
    def get_session_metrics(self) -> Dict:
        """Obtener métricas de sesión actual"""
        metrics = self.client.hgetall("session:metrics")
        
        # Convert numeric strings to floats
        for key, value in metrics.items():
            if key != "last_update":
                try:
                    metrics[key] = float(value)
                except ValueError:
                    pass
        
        return metrics
    
    def increment_trade_count(self) -> int:
        """Incrementar contador de trades"""
        return self.client.hincrby("session:metrics", "total_trades", 1)
    
    def update_win_rate(self, won: bool) -> None:
        """Actualizar win rate"""
        wins = int(self.client.hget("session:metrics", "total_wins") or 0)
        losses = int(self.client.hget("session:metrics", "total_losses") or 0)
        
        if won:
            wins += 1
            self.client.hset("session:metrics", "total_wins", wins)
        else:
            losses += 1
            self.client.hset("session:metrics", "total_losses", losses)
        
        # Calculate win rate
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0.0
        self.client.hset("session:metrics", "win_rate", win_rate)
    
    # ====================================
    # RISK LIMITS
    # ====================================
    
    def check_daily_loss_limit(self, max_loss_pct: float = 5.0) -> Dict:
        """
        Verificar si se alcanzó límite de pérdida diaria.
        
        Args:
            max_loss_pct: Pérdida máxima permitida (%)
        
        Returns:
            Dict con status y detalles
        """
        dd = self.get_drawdown()
        current_dd_pct = dd['drawdown_pct']
        
        limit_breached = current_dd_pct >= max_loss_pct
        
        return {
            "limit_breached": limit_breached,
            "current_drawdown_pct": current_dd_pct,
            "max_allowed_pct": max_loss_pct,
            "remaining_pct": max(0, max_loss_pct - current_dd_pct)
        }
    
    def set_risk_limit(self, limit_type: str, value: float) -> None:
        """
        Configurar límite de riesgo.
        
        Args:
            limit_type: Tipo de límite (max_dd, max_positions, etc.)
            value: Valor del límite
        """
        self.client.hset("risk:limits", limit_type, value)
    
    def get_risk_limits(self) -> Dict:
        """Obtener todos los límites de riesgo"""
        limits = self.client.hgetall("risk:limits")
        
        # Convert to float
        return {k: float(v) for k, v in limits.items()}
    
    # ====================================
    # SYSTEM STATE
    # ====================================
    
    def set_system_state(self, state: str) -> None:
        """
        Actualizar estado del sistema.
        
        Args:
            state: Estado (RUNNING, PAUSED, ERROR, STOPPED)
        """
        self.client.set("system:state", state)
        self.client.set("system:state_timestamp", datetime.now().isoformat())
    
    def get_system_state(self) -> str:
        """Obtener estado actual del sistema"""
        return self.client.get("system:state") or "UNKNOWN"
    
    def heartbeat(self) -> None:
        """Actualizar heartbeat (sistema vivo)"""
        self.client.set("system:heartbeat", datetime.now().isoformat())
        self.client.expire("system:heartbeat", 30)  # Expira en 30s
    
    def is_system_alive(self, timeout_seconds: int = 30) -> bool:
        """
        Verificar si sistema está vivo.
        
        Args:
            timeout_seconds: Timeout de heartbeat
        
        Returns:
            True si heartbeat reciente
        """
        heartbeat = self.client.get("system:heartbeat")
        
        if not heartbeat:
            return False
        
        last_beat = datetime.fromisoformat(heartbeat)
        now = datetime.now()
        
        elapsed = (now - last_beat).total_seconds()
        
        return elapsed < timeout_seconds
    
    # ====================================
    # MAINTENANCE
    # ====================================
    
    def flush_session_data(self) -> None:
        """Limpiar datos de sesión (fin de día)"""
        print("[Redis] Flushing session data...")
        
        # Clear positions
        self.client.delete("positions:open")
        self.client.delete("positions:count")
        
        # Clear session metrics
        self.client.delete("session:metrics")
        
        # Keep equity history for analysis
        # self.client.delete("equity:history")  # NO borrar
        
        print("[Redis] Session data cleared")
    
    def save_snapshot(self, filename: str = "redis_snapshot.rdb") -> None:
        """
        Guardar snapshot de Redis en disco.
        
        Args:
            filename: Nombre del archivo RDB
        """
        self.client.save()
        print(f"[Redis] Snapshot saved to {filename}")
    
    def get_cache_stats(self) -> Dict:
        """Obtener estadísticas del cache"""
        info = self.client.info("stats")
        
        return {
            "total_connections": info.get("total_connections_received", 0),
            "total_commands": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
            "connected_clients": info.get("connected_clients", 0)
        }
    
    def close(self) -> None:
        """Cerrar conexión a Redis"""
        self.client.close()
        print("[Redis] Connection closed")


# ====================================
# EJEMPLO DE USO
# ====================================

def example_usage():
    """Demostración de Redis State Cache"""
    
    # Conectar
    cache = RedisStateCache()
    
    # Simular trading session
    print("\n=== INICIO DE SESIÓN ===")
    cache.set_system_state("RUNNING")
    cache.update_equity(100000.0)
    
    # Abrir posición
    position = PositionState(
        symbol="EURUSD",
        side="long",
        entry_price=1.1000,
        entry_time=datetime.now().isoformat(),
        size=0.10,
        stop_loss=1.0950,
        take_profit=1.1100,
        strategy_id="keltner_breakout"
    )
    
    cache.cache_position("EURUSD", position)
    print(f"Posición abierta: {position}")
    
    # Simular P&L
    cache.update_equity(100500.0)  # +500 USD
    cache.increment_trade_count()
    cache.update_win_rate(won=True)
    
    # Verificar drawdown
    dd = cache.get_drawdown()
    print(f"\nDrawdown: {dd['drawdown_pct']:.2f}%")
    
    # Verificar límites de riesgo
    cache.set_risk_limit("max_daily_dd", 5.0)
    limit_check = cache.check_daily_loss_limit(max_loss_pct=5.0)
    print(f"Risk Limit: {limit_check}")
    
    # Métricas de sesión
    metrics = cache.get_session_metrics()
    print(f"\nSession Metrics: {metrics}")
    
    # Heartbeat
    cache.heartbeat()
    is_alive = cache.is_system_alive()
    print(f"Sistema vivo: {is_alive}")
    
    # Simular crash y recuperación
    print("\n=== SIMULANDO CRASH ===")
    positions_before_crash = cache.get_all_positions()
    equity_before_crash = cache.get_equity()
    
    print(f"Antes del crash:")
    print(f"  Equity: ${equity_before_crash:,.2f}")
    print(f"  Posiciones: {len(positions_before_crash)}")
    
    # ... Aquí ocurriría el crash de Python ...
    
    print("\n=== RECUPERACIÓN POST-CRASH ===")
    # Reconectar a Redis (misma instancia)
    cache_after_crash = RedisStateCache()
    
    positions_after = cache_after_crash.get_all_positions()
    equity_after = cache_after_crash.get_equity()
    
    print(f"Después de recuperación:")
    print(f"  Equity: ${equity_after:,.2f}")
    print(f"  Posiciones: {len(positions_after)}")
    print(f"  Estado recuperado: {'✅ OK' if equity_after == equity_before_crash else '❌ ERROR'}")
    
    # Stats
    stats = cache.get_cache_stats()
    print(f"\nRedis Stats:")
    print(f"  Memory: {stats['used_memory_mb']:.2f} MB")
    print(f"  Commands: {stats['total_commands']}")
    print(f"  Hit Rate: {stats['keyspace_hits']}/{stats['keyspace_hits'] + stats['keyspace_misses']}")
    
    # Cleanup
    cache.close()


if __name__ == "__main__":
    print("Redis State Cache - Sistema de Recuperación ante Fallos")
    print("="*60)
    
    try:
        example_usage()
    except redis.ConnectionError:
        print("\n[ERROR] Redis no está corriendo.")
        print("\nPara instalar Redis:")
        print("  Docker: docker run -d -p 6379:6379 redis:alpine")
        print("  Windows: choco install redis")
        print("  Linux: sudo apt install redis-server")
