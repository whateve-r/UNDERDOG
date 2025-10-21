import asyncio
import zmq
import zmq.asyncio
import json
import time
import traceback
import subprocess
import os 
import signal
from typing import Dict, Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass 

# Importar schemas correctos desde underdog.core.schemas
from underdog.core.schemas.account_info import AccountInfo

# === SCHEMAS DUMMY (solo OrderResult) ===

@dataclass
class OrderResult:
    """Representa el resultado de una solicitud de trading (Placeholder)."""
    retcode: int = 10009 
    deal: int = 0
    order: int = 0
    comment: str = ""
    price: float = 0.0
    volume: float = 0.0
    request_id: Optional[int] = None

@dataclass
class HistoricalData:
    """Contenedor para una serie de datos históricos (Placeholder)."""
    bars: List[Dict[str, Any]]
    symbol: Optional[str] = None
    timeframe: Optional[str] = None

def load_connector_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carga configuración desde YAML. Soporta path personalizado o usa default.
    
    Args:
        config_path: Ruta opcional al archivo de configuración YAML.
                     Si None, usa config/runtime/env/mt5_credentials.yaml
    
    Returns:
        Dict con configuración de ZeroMQ y MT5
    """
    import yaml
    from pathlib import Path
    
    if config_path is None:
        # Ruta por defecto desde raíz del proyecto
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "config" / "runtime" / "env" / "mt5_credentials.yaml"
    else:
        config_path = Path(config_path)
    
    # Valores por defecto en caso de fallo
    default_config = {
        'zmq_host': '127.0.0.1',
        'sys_port': 25555,
        'data_port': 25556,
        'live_port': 25557,
        'stream_port': 25558,
        'mt5_exe_path': r'C:\Program Files\MetaTrader 5\terminal64.exe',
        'mql5_script': 'JsonAPI.ex5',
        'debug': False
    }
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using defaults.")
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Merge con defaults para claves faltantes
        merged_config = {**default_config, **(config or {})}
        logger.info(f"Configuration loaded from {config_path}")
        return merged_config
    
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}. Using defaults.")
        return default_config

# Placeholder para el logging
class SimpleLogger:
    def info(self, msg): print(f"[INFO] {msg}")
    def warning(self, msg): print(f"[WARN] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")
    def critical(self, msg): print(f"[CRIT] {msg}")
    def success(self, msg): print(f"[SUCCESS] {msg}")
logger = SimpleLogger()


class Mt5Connector:
    """
    Conector de MetaTrader 5 basado en ZeroMQ (ZMQ) para comunicación asíncrona.
    """
    def __init__(self, host: str = "127.0.0.1", 
                 mt5_exe: Optional[str] = None, 
                 mql5_script: Optional[str] = None,
                 sys_timeout: float = 3.0, 
                 heartbeat: int = 5, 
                 auto_restart: bool = True):
        
        self.sys_timeout = sys_timeout
        self.heartbeat = heartbeat
        self.auto_restart = auto_restart
        
        config = load_connector_config() 
        logger.info("Configuración de MT5/ZMQ cargada desde la función local.")
        
        # Asignación de valores con prioridad
        self.host = host if host != "127.0.0.1" else config.get('zmq_host', '127.0.0.1')
        self.mt5_exe = mt5_exe if mt5_exe is not None else config.get('mt5_exe_path')
        self.mql5_script = mql5_script if mql5_script is not None else config.get('mql5_script')
        
        # Puertos ZMQ (guardar como atributos para reconexión)
        self.sys_port = config.get('sys_port', 25555)
        self.data_port = config.get('data_port', 25556)
        self.live_port = config.get('live_port', 25557)
        self.stream_port = config.get('stream_port', 25558)
        
        if self.mt5_exe is None:
             logger.error("Ruta del ejecutable MT5 (mt5_exe_path) no encontrada.")
        
        self.context = zmq.asyncio.Context()
        self.last_data = time.time()
        self.mt5_process = None
        self.is_connected = False
        self._init_sockets()
        
    def _init_sockets(self):
        """Inicializa o reinicializa todos los sockets ZeroMQ."""
        for sock_name in ['sys_socket', 'live_socket', 'stream_socket', 'data_socket']:
            sock = getattr(self, sock_name, None)
            if sock:
                try:
                    sock.setsockopt(zmq.LINGER, 0)
                    sock.close()
                except Exception:
                    pass

        # REQ: Para comandos de sistema síncronos
        self.sys_socket = self.context.socket(zmq.REQ)
        self.sys_socket.connect(f"tcp://{self.host}:{self.sys_port}")
        logger.info(f"SYS socket conectado a tcp://{self.host}:{self.sys_port} (REQ)")

        # PULL: Para datos de cuenta en vivo
        self.live_socket = self.context.socket(zmq.PULL)
        self.live_socket.connect(f"tcp://{self.host}:{self.live_port}")
        logger.info(f"LIVE socket conectado a tcp://{self.host}:{self.live_port} (PULL)")
        
        # PULL: Para datos de mercado en tiempo real
        self.stream_socket = self.context.socket(zmq.PULL)
        self.stream_socket.connect(f"tcp://{self.host}:{self.stream_port}")
        logger.info(f"STREAM socket conectado a tcp://{self.host}:{self.stream_port} (PULL)")

        # PULL: Socket DATA original 
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.connect(f"tcp://{self.host}:{self.data_port}")
        logger.info(f"DATA socket conectado a tcp://{self.host}:{self.data_port} (PULL)")
        
        # CRITICAL: Give sockets time to establish connection (especially PULL sockets)
        # ZMQ PULL sockets need time to subscribe before messages arrive
        # Without this, first messages may be lost in round-robin distribution
        import time as sync_time
        sync_time.sleep(0.1)  # 100ms should be enough for socket handshake

    def _reconnect_socket(self, old_socket: zmq.asyncio.Socket, port_name: str, port: int, socket_type: str = "PULL") -> zmq.asyncio.Socket:
        """Cierra el socket existente y crea y conecta uno nuevo."""
        try:
            old_socket.setsockopt(zmq.LINGER, 0)
            old_socket.close()
        except Exception: 
            pass 
        
        new_socket = self.context.socket(zmq.REQ if socket_type=="REQ" else zmq.PULL)
        new_socket.connect(f"tcp://{self.host}:{port}")
        logger.warning(f"[{port_name} SOCKET] Reconectado a tcp://{self.host}:{port}")
        return new_socket

    # ------------------------------
    # Conexión y Watchdog
    # ------------------------------
    async def connect(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        """
        Valida la conexión con MT5/EA que ya debe estar corriendo.
        
        IMPORTANTE: MT5 Terminal debe estar abierto manualmente con JsonAPI EA activo
        ANTES de llamar a este método. Este método NO lanza MT5 automáticamente.
        
        Args:
            max_retries: Número máximo de intentos de conexión (default: 3)
            retry_delay: Segundos entre intentos (default: 2.0)
        
        Returns:
            bool: True si la conexión es exitosa, False en caso contrario
        """
        # Solo lanzar MT5 si auto_restart está habilitado Y MT5 no está corriendo
        if self.auto_restart and self.mt5_exe and not self._is_mt5_alive():
            logger.warning("auto_restart está habilitado pero MT5 no está corriendo. Lanzando MT5...")
            self._launch_mt5()
            # Dar tiempo al EA para inicializar
            await asyncio.sleep(5)
        
        logger.info(f"Validando conexión con MT5/EA (max {max_retries} intentos)...")
        
        for attempt in range(1, max_retries + 1):
            try:
                # Test rápido: enviar request de ACCOUNT
                logger.info(f"Intento {attempt}/{max_retries}: Solicitando info de cuenta...")
                info = await self.get_account_info()
                
                # Validar que recibimos datos válidos
                if info and info.balance >= 0:  # Balance puede ser 0 en cuenta nueva
                    logger.success(f"✅ Conexión exitosa con MT5/EA (Broker: {info.broker}, Balance: ${info.balance:,.2f})")
                    self.is_connected = True
                    
                    # Iniciar watchdog solo si auto_restart está habilitado
                    if self.auto_restart and self.mt5_process is not None:
                        asyncio.create_task(self._watch_mt5())
                        logger.info("Watchdog MT5 iniciado.")
                    
                    return True
                else:
                    logger.warning(f"Respuesta inválida del EA (intento {attempt}/{max_retries})")
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout en intento {attempt}/{max_retries}")
            except Exception as e:
                logger.warning(f"Error en intento {attempt}/{max_retries}: {e}")
            
            # Esperar antes del siguiente intento (excepto en el último)
            if attempt < max_retries:
                logger.info(f"Reintentando en {retry_delay}s...")
                await asyncio.sleep(retry_delay)
        
        # Todos los intentos fallaron
        logger.error(
            "❌ No se pudo establecer conexión con MT5/EA.\n"
            "   CHECKLIST:\n"
            "   1. ¿MT5 está abierto?\n"
            "   2. ¿JsonAPI EA está cargado en un gráfico?\n"
            "   3. ¿El EA muestra cara feliz 😊 (no 😞)?\n"
            "   4. ¿AutoTrading está habilitado (botón verde)?\n"
            "   5. ¿'Allow DLL imports' está habilitado en EA settings?\n"
            "   6. ¿Los logs de MT5 muestran 'Binding socket on port 25555...'?\n"
            "\n"
            "   Ver guía completa: docs/MT5_JSONAPI_SETUP.md"
        )
        return False
        
    async def disconnect(self):
        """Termina el contexto ZMQ y el proceso MT5 si fue lanzado por el script."""
        self.context.term()
        logger.info("Contexto ZMQ terminado.")
        if self.mt5_process is not None:
            self._terminate_mt5()

    # ------------------------------
    # SYS request con manejo de backoff y reconexión
    # ------------------------------
    async def sys_request(self, message: Dict[str, Any], max_retries: int = 5) -> Optional[Dict[str, Any]]:
        """
        Envía un comando al socket REQ y espera la respuesta con reintento y backoff.
        
        IMPORTANTE: Patrón REQ/REP es frágil. Si no se recibe respuesta, el socket
        queda desincronizado y debe ser reconectado ANTES del siguiente intento.
        
        Args:
            message: Diccionario con el comando (e.g., {"action": "ACCOUNT"})
            max_retries: Número máximo de intentos (default: 5)
        
        Returns:
            Diccionario con la respuesta JSON o None si es "OK" o falla
        """
        attempt = 0
        
        while attempt < max_retries:
            try:
                # Enviar request
                await self.sys_socket.send_string(json.dumps(message))
                
                # Configurar poller para timeout
                poller = zmq.asyncio.Poller()
                poller.register(self.sys_socket, zmq.POLLIN)
                
                # Timeout en milisegundos
                timeout_ms = int(self.sys_timeout * 1000)
                
                # CORRECCIÓN CRÍTICA: Usar solo poller.poll(), sin asyncio.wait_for() anidado
                # poller.poll() ya maneja el timeout correctamente
                socks = dict(await poller.poll(timeout_ms))
                
                # Verificar si hay respuesta disponible
                if self.sys_socket in socks:
                    resp = await self.sys_socket.recv_string()
                    resp_strip = resp.strip()
                    
                    # Respuesta vacía
                    if not resp_strip:
                        logger.warning(f"[SYS REQ] Respuesta vacía (intento {attempt+1}/{max_retries})")
                        # REQ/REP desincronizado → reconectar
                        self.sys_socket = self._reconnect_socket(self.sys_socket, "SYS", self.sys_port, "REQ")
                        attempt += 1
                        await asyncio.sleep(min(0.5 * (2**attempt), 5))
                        continue
                    
                    # Respuesta "OK" (acknowledge - datos vendrán por DATA socket)
                    if resp_strip == "OK":
                        # El EA JsonAPI envía "OK" por SYS y los datos reales por DATA socket
                        logger.info("[SYS REQ] Recibido 'OK', esperando datos por DATA socket...")
                        
                        try:
                            # Esperar datos por DATA socket con timeout
                            data_poller = zmq.asyncio.Poller()
                            data_poller.register(self.data_socket, zmq.POLLIN)
                            
                            # Timeout más largo para operaciones pesadas (e.g., HISTORY puede ser muy lento)
                            # HISTORY con muchos datos puede tardar 30+ segundos
                            data_timeout_ms = 60000  # 60 segundos (era 10s)
                            logger.info(f"[SYS REQ] Polling DATA socket con timeout de {data_timeout_ms/1000}s...")
                            data_socks = dict(await data_poller.poll(data_timeout_ms))
                            
                            if self.data_socket in data_socks:
                                logger.info("[SYS REQ] Datos disponibles en DATA socket, recibiendo...")
                                data_resp = await self.data_socket.recv_string()
                                logger.info(f"[SYS REQ] Recibidos {len(data_resp)} caracteres del DATA socket")
                                data_json = json.loads(data_resp)
                                
                                # Verificar si hay error en la respuesta
                                if data_json.get('error'):
                                    logger.error(f"[SYS REQ] EA retornó error: {data_json}")
                                
                                # Actualizar timestamp de última comunicación
                                self.last_data = time.time()
                                return data_json
                            else:
                                logger.error(f"[SYS REQ] Timeout de {data_timeout_ms/1000}s esperando datos por DATA socket")
                                logger.error(f"[SYS REQ] Comando enviado: {json.dumps(message)}")
                                return None
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"[SYS REQ] Respuesta de DATA socket no es JSON: {e}")
                            logger.error(f"[SYS REQ] Data recibida: {data_resp[:500]}...")
                            return None
                        except Exception as e:
                            logger.error(f"[SYS REQ] Error leyendo DATA socket: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            return None
                    
                    # Parse JSON
                    try:
                        data = json.loads(resp_strip)
                        # Success - actualizar timestamp de última comunicación
                        self.last_data = time.time()
                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"[SYS REQ] Respuesta no es JSON válido: {resp_strip[:100]}...")
                        logger.error(f"[SYS REQ] JSONDecodeError: {e}")
                        # REQ/REP desincronizado → reconectar
                        self.sys_socket = self._reconnect_socket(self.sys_socket, "SYS", self.sys_port, "REQ")
                        attempt += 1
                        await asyncio.sleep(min(0.5 * (2**attempt), 5))
                        continue
                        
                else:
                    # Timeout - no hay respuesta
                    logger.warning(f"[SYS REQ] Timeout de {self.sys_timeout}s (intento {attempt+1}/{max_retries})")
                    # REQ/REP desincronizado → DEBE reconectar socket
                    self.sys_socket = self._reconnect_socket(self.sys_socket, "SYS", self.sys_port, "REQ")
                    
            except asyncio.CancelledError:
                # Permitir cancelación limpia del task
                logger.info("[SYS REQ] Request cancelado por el usuario")
                raise
                
            except Exception as e:
                logger.error(f"[SYS REQ] Error inesperado (intento {attempt+1}/{max_retries}): {e}")
                # REQ/REP potencialmente dañado → reconectar
                self.sys_socket = self._reconnect_socket(self.sys_socket, "SYS", self.sys_port, "REQ")
            
            # Backoff exponencial antes del siguiente intento
            attempt += 1
            if attempt < max_retries:
                backoff = min(0.5 * (2**attempt), 5)  # Max 5 segundos
                await asyncio.sleep(backoff)
        
        # Todos los intentos fallaron
        logger.critical(f"[SYS REQ] Falló después de {max_retries} intentos. Comando: {message.get('action', 'UNKNOWN')}")
        return None

    # ------------------------------
    # Listeners asíncronos 
    # ------------------------------
    async def listen_socket(self, socket_attr_name: str, callback: Callable[[Dict[str, Any]], Awaitable[None]], port_name: str, port: int, socket_type: str = "PULL"):
        """
        Función genérica para escuchar un socket y manejar timeouts (heartbeat).
        """
        socket = getattr(self, socket_attr_name) 
        
        while True:
            try:
                msg = await asyncio.wait_for(socket.recv_string(), timeout=self.heartbeat)
                self.last_data = time.time()
                data = json.loads(msg)
                
                await callback(data)
                
            except asyncio.TimeoutError:
                logger.warning(f"[{port_name} SOCKET] No hay datos por {self.heartbeat}s. Forzando reconexión.")
                
                new_socket = self._reconnect_socket(socket, port_name, port, socket_type)
                setattr(self, socket_attr_name, new_socket)
                socket = new_socket 
                
            except asyncio.CancelledError:
                logger.info(f"[{port_name} SOCKET] Listener cancelado.")
                break
            except Exception as e:
                logger.error(f"[{port_name} SOCKET] Excepción en el listener: {e}. Reconectando...")
                
                new_socket = self._reconnect_socket(socket, port_name, port, socket_type)
                setattr(self, socket_attr_name, new_socket)
                socket = new_socket 

    async def listen_live(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]): 
        return await self.listen_socket("live_socket", callback, "LIVE", 25557, "PULL")
        
    async def listen_stream(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]): 
        return await self.listen_socket("stream_socket", callback, "STREAM", 25558, "PULL")

    async def listen_data(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]): 
        return await self.listen_socket("data_socket", callback, "DATA", 25556, "PULL")


    async def run_listeners(self, live_cb: Callable, stream_cb: Callable, data_cb: Optional[Callable] = None):
        """Lanza todos los listeners de datos en tareas concurrentes."""
        tasks = [
            self.listen_live(live_cb), 
            self.listen_stream(stream_cb)
        ]
        if data_cb:
            tasks.append(self.listen_data(data_cb))
            
        await asyncio.gather(*tasks)

    # ------------------------------
    # Operaciones de Trading (simplificado)
    # ------------------------------
    async def get_account_info(self) -> Optional[AccountInfo]:
        """Obtiene la información de la cuenta y la mapea al Schema AccountInfo."""
        
        response = await self.sys_request({"action": "ACCOUNT"})
        
        # JsonAPI EA devuelve las claves en minúsculas directamente (no anidadas en 'Account')
        if response and isinstance(response, dict):
            try:
                # Verificar que error = false
                if response.get('error', False):
                    logger.error(f"EA devolvió error: {response}")
                    return None
                
                return AccountInfo(
                    balance=float(response.get('balance', 0.0)),
                    equity=float(response.get('equity', 0.0)),
                    margin=float(response.get('margin', 0.0)),
                    margin_free=float(response.get('margin_free', 0.0)),
                    margin_level=float(response.get('margin_level', 0.0)),
                    leverage=1,  # JsonAPI no devuelve leverage en ACCOUNT
                    name='',  # JsonAPI no devuelve name en ACCOUNT
                    server=str(response.get('server', '')),
                    broker=str(response.get('broker', '')),
                    currency=str(response.get('currency', 'USD')),
                    trading_allowed=bool(response.get('trading_allowed', 0)),  # Convert 1/0 to bool
                    bot_trading=bool(response.get('bot_trading', 0))  # Convert 1/0 to bool
                )
            except Exception as e:
                logger.error(f"Error al mapear datos de cuenta a Schema AccountInfo: {e}")
                logger.error(f"Response recibida: {response}")
                return None
        
        logger.warning("No se recibió respuesta válida del EA para ACCOUNT")
        return None

    # ------------------------------
    # Watchdog MT5 y relaunch automático
    # ------------------------------
    async def _watch_mt5(self):
        """Tarea de vigilancia para asegurar que el terminal MT5 y el EA están activos."""
        while True:
            await asyncio.sleep(5) 
            
            if self.last_data + self.heartbeat * 2 < time.time():
                logger.warning("[WATCHDOG] No se detectan datos recientes. Revisando estado de MT5...")
                if not self._is_mt5_alive():
                    logger.error("[WATCHDOG] MT5 no está corriendo o ha muerto. Relanzando...")
                    self._launch_mt5()
                    
                    self._init_sockets()
                    
                    await asyncio.sleep(10) # Espera a que el EA se inicialice
                    
                else:
                    logger.info("[WATCHDOG] MT5 está vivo, pero el flujo de datos se detuvo. Intentando un PING (SYS request)...")
                    await self.sys_request({"action": "PING"})
                    
    def _is_mt5_alive(self) -> bool:
        """
        Verifica si el proceso MT5 está en ejecución.
        
        CORRECCIÓN CRÍTICA: No depender de self.mt5_process (solo válido si Python lanzó MT5).
        Usa psutil para buscar 'terminal64.exe' en procesos del sistema.
        """
        try:
            import psutil
            
            # Buscar proceso terminal64.exe
            for proc in psutil.process_iter(['name']):
                try:
                    if proc.info['name'].lower() == 'terminal64.exe':
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return False
            
        except ImportError:
            # Fallback: si psutil no está instalado, usar método antiguo
            logger.warning("psutil no está instalado. Usando fallback (menos robusto).")
            if self.mt5_process is None:
                return False
            return self.mt5_process.poll() is None

    def _launch_mt5(self):
        """
        Lanza el ejecutable MT5.
        
        ⚠️ ADVERTENCIA: Este método NO es recomendado para producción.
        - El argumento /script:JsonAPI.ex5 NO carga el EA de forma persistente
        - Solo ejecuta un script una vez y termina
        - El EA debe ser cargado manualmente y guardado en perfil/template
        
        Este método solo debe usarse para testing automatizado con auto_restart=true.
        En VPS de producción, MT5 debe estar abierto manualmente con EA activo.
        """
        if self.mt5_exe is None:
            logger.error("[LAUNCH] Ruta del ejecutable MT5 (mt5_exe) no proporcionada.")
            return

        if not os.path.exists(self.mt5_exe):
            logger.critical(f"[LAUNCH] MT5 NO EXISTE en: {self.mt5_exe}")
            self.mt5_process = None
            return

        # Si MT5 ya está corriendo, no hacer nada (evitar duplicados)
        if self._is_mt5_alive():
            logger.warning("[LAUNCH] MT5 ya está corriendo. No se lanzará otra instancia.")
            return
            
        cmd = [self.mt5_exe]
        
        # NOTA: /script no carga EAs de forma persistente
        # Solo útil para scripts que se ejecutan una vez
        if self.mql5_script:
            cmd.append(f"/script:{self.mql5_script}")
        
        try:
            self.mt5_process = subprocess.Popen(cmd,
                                                 stdout=subprocess.DEVNULL, 
                                                 stderr=subprocess.DEVNULL)
            logger.success(f"[WATCHDOG] MT5 lanzado con éxito: {' '.join(cmd)}")
            
        except OSError as e:
            # Ahora el error incluye el comando exacto para un mejor diagnóstico.
            logger.critical(f"[WATCHDOG] Fallo al lanzar MT5. Error de sistema: {e}. Comando: {' '.join(cmd)}")
            self.mt5_process = None 
        except Exception as e:
            logger.critical(f"[WATCHDOG] Fallo al lanzar MT5. Error inesperado: {e}")
            self.mt5_process = None 


    def _terminate_mt5(self):
        """Intenta terminar el proceso MT5 de forma limpia o forzada."""
        if self.mt5_process:
            try:
                self.mt5_process.terminate()
                self.mt5_process.wait(timeout=5)
            except:
                self.mt5_process.kill()
            finally:
                self.mt5_process = None
                logger.warning("[WATCHDOG] Proceso MT5 terminado.")

# ----------------------------------------------------
# Nota: La función main_mt5_test y el if __name__ se mantienen
# para mostrar la estructura, aunque el código principal esté 
# en otro archivo (trading_bot_example.py)
# ----------------------------------------------------
async def main_mt5_test():
    connector = Mt5Connector()
    if await connector.connect():
        info = await connector.get_account_info()
        if info:
            logger.info(f"Equity: {info.equity}, Free Margin: {info.free_margin}")
        
        await connector.disconnect()

if __name__ == '__main__':
    # La inicialización real de asyncio debe hacerse en start_live.py
    # asyncio.run(main_mt5_test())
    pass
