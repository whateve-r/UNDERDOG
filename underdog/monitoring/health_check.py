"""
Health Check System
Monitors system components and provides health status endpoints.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enum"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component"""
    name: str
    status: HealthStatus
    last_check: datetime
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'status': self.status.value,
            'last_check': self.last_check.isoformat(),
            'message': self.message,
            'latency_ms': self.latency_ms,
            'metadata': self.metadata
        }


@dataclass
class SystemHealth:
    """Overall system health"""
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'uptime_seconds': self.uptime_seconds,
            'components': [c.to_dict() for c in self.components]
        }


class HealthChecker:
    """
    Health Checker for UNDERDOG Trading System
    
    Monitors:
    - MT5 connection status
    - ZeroMQ publisher/subscriber health
    - Risk Master state
    - ML Model freshness
    - Database connection
    - System resources
    """
    
    def __init__(self, 
                 check_interval: float = 30.0,
                 mt5_timeout: float = 5.0,
                 model_staleness_threshold: float = 86400.0):  # 24 hours
        """
        Initialize health checker
        
        Args:
            check_interval: Interval between health checks (seconds)
            mt5_timeout: Timeout for MT5 connection check (seconds)
            model_staleness_threshold: Max model age before considered stale (seconds)
        """
        self.check_interval = check_interval
        self.mt5_timeout = mt5_timeout
        self.model_staleness_threshold = model_staleness_threshold
        
        self.start_time = datetime.utcnow()
        self.last_check_time: Optional[datetime] = None
        
        # Component health cache
        self.component_health: Dict[str, ComponentHealth] = {}
        
        # Health check functions
        self.check_functions: Dict[str, Callable] = {}
    
    def register_check(self, component: str, check_func: Callable):
        """
        Register a health check function
        
        Args:
            component: Component name
            check_func: Function that returns (status, message, metadata)
        """
        self.check_functions[component] = check_func
        logger.info(f"Registered health check for: {component}")
    
    def check_mt5_connection(self, mt5_connector) -> ComponentHealth:
        """
        Check MT5 connection health
        
        Args:
            mt5_connector: MT5Connector instance
        
        Returns:
            ComponentHealth for MT5
        """
        start_time = time.time()
        
        try:
            # Check if connected
            if not mt5_connector.connected:
                return ComponentHealth(
                    name='mt5',
                    status=HealthStatus.UNHEALTHY,
                    last_check=datetime.utcnow(),
                    message='MT5 not connected',
                    latency_ms=None
                )
            
            # Try to get account info (lightweight check)
            account_info = mt5_connector.get_account_info()
            
            if account_info is None:
                return ComponentHealth(
                    name='mt5',
                    status=HealthStatus.DEGRADED,
                    last_check=datetime.utcnow(),
                    message='MT5 connected but account info unavailable',
                    latency_ms=(time.time() - start_time) * 1000
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Check latency
            if latency_ms > 1000:  # > 1 second
                status = HealthStatus.DEGRADED
                message = f'High latency: {latency_ms:.0f}ms'
            else:
                status = HealthStatus.HEALTHY
                message = 'Connected'
            
            return ComponentHealth(
                name='mt5',
                status=status,
                last_check=datetime.utcnow(),
                message=message,
                latency_ms=latency_ms,
                metadata={
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'server': account_info.server
                }
            )
        
        except Exception as e:
            logger.error(f"MT5 health check failed: {e}")
            return ComponentHealth(
                name='mt5',
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                message=f'Error: {str(e)}',
                latency_ms=None
            )
    
    def check_zeromq_health(self, zmq_connections: Dict) -> ComponentHealth:
        """
        Check ZeroMQ connections health
        
        Args:
            zmq_connections: Dict with publisher/subscriber instances
        
        Returns:
            ComponentHealth for ZeroMQ
        """
        try:
            active_count = 0
            connection_types = []
            
            for conn_type, conn in zmq_connections.items():
                if conn and hasattr(conn, 'is_alive'):
                    if conn.is_alive():
                        active_count += 1
                        connection_types.append(conn_type)
            
            if active_count == 0:
                status = HealthStatus.UNHEALTHY
                message = 'No active ZeroMQ connections'
            elif active_count < len(zmq_connections):
                status = HealthStatus.DEGRADED
                message = f'{active_count}/{len(zmq_connections)} connections active'
            else:
                status = HealthStatus.HEALTHY
                message = 'All connections active'
            
            return ComponentHealth(
                name='zeromq',
                status=status,
                last_check=datetime.utcnow(),
                message=message,
                metadata={
                    'active_connections': active_count,
                    'total_connections': len(zmq_connections),
                    'connection_types': connection_types
                }
            )
        
        except Exception as e:
            logger.error(f"ZeroMQ health check failed: {e}")
            return ComponentHealth(
                name='zeromq',
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                message=f'Error: {str(e)}'
            )
    
    def check_risk_master_health(self, risk_master) -> ComponentHealth:
        """
        Check Risk Master health
        
        Args:
            risk_master: RiskMaster instance
        
        Returns:
            ComponentHealth for Risk Master
        """
        try:
            # Check kill switch status
            if risk_master.kill_switch_active:
                status = HealthStatus.DEGRADED
                message = 'Kill switch ACTIVE'
            else:
                status = HealthStatus.HEALTHY
                message = 'Operating normally'
            
            # Get current metrics
            dd_pct = risk_master.current_drawdown_pct
            daily_dd = risk_master.daily_drawdown_pct
            
            # Check drawdown levels
            if dd_pct > risk_master.max_drawdown_pct * 0.8:
                status = HealthStatus.DEGRADED
                message = f'High drawdown: {dd_pct:.2f}%'
            
            return ComponentHealth(
                name='risk_master',
                status=status,
                last_check=datetime.utcnow(),
                message=message,
                metadata={
                    'kill_switch_active': risk_master.kill_switch_active,
                    'current_drawdown_pct': dd_pct,
                    'daily_drawdown_pct': daily_dd,
                    'open_positions': len(risk_master.positions)
                }
            )
        
        except Exception as e:
            logger.error(f"Risk Master health check failed: {e}")
            return ComponentHealth(
                name='risk_master',
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                message=f'Error: {str(e)}'
            )
    
    def check_model_freshness(self, model_last_updated: datetime) -> ComponentHealth:
        """
        Check ML model freshness
        
        Args:
            model_last_updated: Timestamp of last model update
        
        Returns:
            ComponentHealth for ML model
        """
        try:
            age_seconds = (datetime.utcnow() - model_last_updated).total_seconds()
            
            if age_seconds > self.model_staleness_threshold:
                status = HealthStatus.DEGRADED
                message = f'Model stale: {age_seconds/3600:.1f} hours old'
            elif age_seconds > self.model_staleness_threshold * 2:
                status = HealthStatus.UNHEALTHY
                message = f'Model very stale: {age_seconds/86400:.1f} days old'
            else:
                status = HealthStatus.HEALTHY
                message = 'Model fresh'
            
            return ComponentHealth(
                name='ml_model',
                status=status,
                last_check=datetime.utcnow(),
                message=message,
                metadata={
                    'age_seconds': age_seconds,
                    'last_updated': model_last_updated.isoformat()
                }
            )
        
        except Exception as e:
            logger.error(f"Model freshness check failed: {e}")
            return ComponentHealth(
                name='ml_model',
                status=HealthStatus.UNKNOWN,
                last_check=datetime.utcnow(),
                message=f'Error: {str(e)}'
            )
    
    def check_database_connection(self, db_connection) -> ComponentHealth:
        """
        Check database connection health
        
        Args:
            db_connection: Database connection instance
        
        Returns:
            ComponentHealth for database
        """
        start_time = time.time()
        
        try:
            # Simple query to test connection
            if hasattr(db_connection, 'execute'):
                db_connection.execute('SELECT 1')
            
            latency_ms = (time.time() - start_time) * 1000
            
            if latency_ms > 500:
                status = HealthStatus.DEGRADED
                message = f'High latency: {latency_ms:.0f}ms'
            else:
                status = HealthStatus.HEALTHY
                message = 'Connected'
            
            return ComponentHealth(
                name='database',
                status=status,
                last_check=datetime.utcnow(),
                message=message,
                latency_ms=latency_ms
            )
        
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                name='database',
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                message=f'Error: {str(e)}'
            )
    
    def check_all(self, **components) -> SystemHealth:
        """
        Run all health checks
        
        Args:
            **components: Component instances (mt5_connector, risk_master, etc.)
        
        Returns:
            SystemHealth with all component statuses
        """
        component_health_list = []
        
        # Check MT5
        if 'mt5_connector' in components:
            health = self.check_mt5_connection(components['mt5_connector'])
            component_health_list.append(health)
            self.component_health['mt5'] = health
        
        # Check ZeroMQ
        if 'zmq_connections' in components:
            health = self.check_zeromq_health(components['zmq_connections'])
            component_health_list.append(health)
            self.component_health['zeromq'] = health
        
        # Check Risk Master
        if 'risk_master' in components:
            health = self.check_risk_master_health(components['risk_master'])
            component_health_list.append(health)
            self.component_health['risk_master'] = health
        
        # Check Model Freshness
        if 'model_last_updated' in components:
            health = self.check_model_freshness(components['model_last_updated'])
            component_health_list.append(health)
            self.component_health['ml_model'] = health
        
        # Check Database
        if 'db_connection' in components:
            health = self.check_database_connection(components['db_connection'])
            component_health_list.append(health)
            self.component_health['database'] = health
        
        # Run custom checks
        for component_name, check_func in self.check_functions.items():
            try:
                status, message, metadata = check_func()
                health = ComponentHealth(
                    name=component_name,
                    status=status,
                    last_check=datetime.utcnow(),
                    message=message,
                    metadata=metadata
                )
                component_health_list.append(health)
                self.component_health[component_name] = health
            except Exception as e:
                logger.error(f"Custom health check failed for {component_name}: {e}")
        
        # Determine overall status
        overall_status = self._determine_overall_status(component_health_list)
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        self.last_check_time = datetime.utcnow()
        
        return SystemHealth(
            status=overall_status,
            components=component_health_list,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime_seconds
        )
    
    def _determine_overall_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """Determine overall system health from component statuses"""
        if not components:
            return HealthStatus.UNKNOWN
        
        # If any component is unhealthy, system is unhealthy
        if any(c.status == HealthStatus.UNHEALTHY for c in components):
            return HealthStatus.UNHEALTHY
        
        # If any component is degraded, system is degraded
        if any(c.status == HealthStatus.DEGRADED for c in components):
            return HealthStatus.DEGRADED
        
        # Otherwise, system is healthy
        return HealthStatus.HEALTHY
    
    def get_component_health(self, component: str) -> Optional[ComponentHealth]:
        """Get health status for specific component"""
        return self.component_health.get(component)
    
    def is_healthy(self) -> bool:
        """Check if system is healthy (no unhealthy components)"""
        return all(
            h.status != HealthStatus.UNHEALTHY 
            for h in self.component_health.values()
        )
