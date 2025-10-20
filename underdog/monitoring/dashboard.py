"""
Dashboard and API Server
FastAPI server providing health check and metrics endpoints.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from datetime import datetime
from typing import Optional
import uvicorn

from .metrics import MetricsCollector
from .health_check import HealthChecker, HealthStatus
from .alerts import AlertManager


class MonitoringDashboard:
    """
    FastAPI-based monitoring dashboard
    
    Endpoints:
    - GET /health - System health check
    - GET /metrics - Prometheus metrics
    - GET /stats - Trading statistics
    - POST /alerts/test - Test alert system
    """
    
    def __init__(self,
                 metrics_collector: MetricsCollector,
                 health_checker: HealthChecker,
                 alert_manager: Optional[AlertManager] = None,
                 host: str = "0.0.0.0",
                 port: int = 8000):
        """
        Initialize monitoring dashboard
        
        Args:
            metrics_collector: MetricsCollector instance
            health_checker: HealthChecker instance
            alert_manager: AlertManager instance (optional)
            host: Server host
            port: Server port
        """
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self.alert_manager = alert_manager
        self.host = host
        self.port = port
        
        # Create FastAPI app
        self.app = FastAPI(
            title="UNDERDOG Trading Monitor",
            description="Monitoring and telemetry for UNDERDOG algorithmic trading system",
            version="1.0.0"
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "name": "UNDERDOG Trading Monitor",
                "version": "1.0.0",
                "endpoints": {
                    "health": "/health",
                    "metrics": "/metrics",
                    "stats": "/stats",
                    "alerts": "/alerts/test"
                }
            }
        
        @self.app.get("/health")
        async def health_check(component: Optional[str] = None):
            """
            Health check endpoint
            
            Query params:
                component: Optional component name to check
            
            Returns:
                JSON with health status
            """
            if component:
                # Check specific component
                comp_health = self.health_checker.get_component_health(component)
                if not comp_health:
                    raise HTTPException(status_code=404, detail=f"Component '{component}' not found")
                return JSONResponse(content=comp_health.to_dict())
            
            # Check all components
            try:
                system_health = self.health_checker.check_all()
                status_code = 200 if system_health.status == HealthStatus.HEALTHY else 503
                return JSONResponse(
                    content=system_health.to_dict(),
                    status_code=status_code
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics", response_class=PlainTextResponse)
        async def metrics():
            """
            Prometheus metrics endpoint
            
            Returns:
                Prometheus-formatted metrics
            """
            try:
                return self.metrics_collector.export_metrics()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats")
        async def stats():
            """
            Trading statistics endpoint
            
            Returns:
                JSON with current trading metrics
            """
            try:
                snapshot = self.metrics_collector.get_snapshot()
                return {
                    "timestamp": snapshot.last_update.isoformat(),
                    "trades": {
                        "total": snapshot.total_trades,
                        "wins": snapshot.winning_trades,
                        "losses": snapshot.losing_trades,
                        "win_rate": snapshot.win_rate()
                    },
                    "financial": {
                        "current_capital": snapshot.current_capital,
                        "initial_capital": snapshot.initial_capital,
                        "realized_pnl": snapshot.realized_pnl,
                        "unrealized_pnl": snapshot.unrealized_pnl,
                        "total_return_pct": snapshot.total_return()
                    },
                    "risk": {
                        "current_drawdown": snapshot.current_drawdown,
                        "max_drawdown": snapshot.max_drawdown,
                        "daily_drawdown": snapshot.daily_drawdown,
                        "total_exposure": snapshot.total_exposure
                    },
                    "positions": {
                        "open_count": snapshot.open_positions
                    },
                    "strategies": {
                        "active_count": snapshot.active_strategies,
                        "signals_generated": snapshot.signals_generated,
                        "signals_rejected": snapshot.signals_rejected
                    },
                    "execution": {
                        "last_execution_ms": snapshot.last_execution_time_ms,
                        "avg_execution_ms": snapshot.avg_execution_time_ms
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/alerts/test")
        async def test_alert(severity: str = "info", title: str = "Test Alert"):
            """
            Test alert system
            
            Body:
                severity: Alert severity (info, warning, error, critical)
                title: Alert title
            
            Returns:
                JSON with success status
            """
            if not self.alert_manager:
                raise HTTPException(status_code=503, detail="Alert manager not configured")
            
            try:
                from .alerts import AlertSeverity
                
                severity_map = {
                    'info': AlertSeverity.INFO,
                    'warning': AlertSeverity.WARNING,
                    'error': AlertSeverity.ERROR,
                    'critical': AlertSeverity.CRITICAL
                }
                
                sev = severity_map.get(severity.lower(), AlertSeverity.INFO)
                
                success = self.alert_manager.send_alert(
                    severity=sev,
                    title=title,
                    message="This is a test alert from UNDERDOG monitoring system",
                    metadata={'test': True, 'timestamp': datetime.utcnow().isoformat()},
                    bypass_cooldown=True
                )
                
                return {
                    "success": success,
                    "message": "Alert sent" if success else "Alert failed"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/alerts/stats")
        async def alert_stats():
            """
            Alert statistics endpoint
            
            Returns:
                JSON with alert statistics
            """
            if not self.alert_manager:
                raise HTTPException(status_code=503, detail="Alert manager not configured")
            
            try:
                return self.alert_manager.get_alert_stats()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, **kwargs):
        """
        Run the dashboard server
        
        Args:
            **kwargs: Additional uvicorn config options
        """
        uvicorn.run(
            self.app,
            host=kwargs.get('host', self.host),
            port=kwargs.get('port', self.port),
            **kwargs
        )


def create_monitoring_app(metrics_collector: MetricsCollector,
                          health_checker: HealthChecker,
                          alert_manager: Optional[AlertManager] = None) -> FastAPI:
    """
    Factory function to create monitoring FastAPI app
    
    Args:
        metrics_collector: MetricsCollector instance
        health_checker: HealthChecker instance
        alert_manager: AlertManager instance (optional)
    
    Returns:
        FastAPI app
    """
    dashboard = MonitoringDashboard(
        metrics_collector=metrics_collector,
        health_checker=health_checker,
        alert_manager=alert_manager
    )
    return dashboard.app
