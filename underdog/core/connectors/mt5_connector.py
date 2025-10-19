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

# === SCHEMAS Y FUNCIONES DUMMY ===

@dataclass
class AccountInfo:
    """Representa la información clave de la cuenta de trading (Placeholder)."""
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    leverage: int = 1
    name: Optional[str] = None
    server: Optional[str] = None

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

def load_connector_config() -> Dict[str, Any]:
    """
    Simula la carga de configuración. 
    
    *** ¡ADVERTENCIA CRÍTICA! DEBES CAMBIAR ESTA RUTA ***
    Usa una 'raw string' con 'r' para Windows, y asegúrate que el nombre del 
    ejecutable (terminal64.exe o terminal.exe) es EXACTO.
    """
    return {
        'zmq_host': '127.0.0.1',
        # *** RUTA DE EJEMPLO. MODIFICA ESTO ***
        'mt5_exe_path': r'C:\Program Files\MetaTrader 5\terminal64.exe', 
        # Corregido según tu prueba de Powershell.
        'mql5_script': 'JsonAPI.ex5' 
    }

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
        self.sys_socket.connect(f"tcp://{self.host}:25555")
        logger.info(f"SYS socket conectado a tcp://{self.host}:25555 (REQ)")

        # PULL: Para datos de cuenta en vivo
        self.live_socket = self.context.socket(zmq.PULL)
        self.live_socket.connect(f"tcp://{self.host}:25557")
        logger.info(f"LIVE socket conectado a tcp://{self.host}:25557 (PULL)")
        
        # PULL: Para datos de mercado en tiempo real
        self.stream_socket = self.context.socket(zmq.PULL)
        self.stream_socket.connect(f"tcp://{self.host}:25558")
        logger.info(f"STREAM socket conectado a tcp://{self.host}:25558 (PULL)")

        # PULL: Socket DATA original 
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.connect(f"tcp://{self.host}:25556")
        logger.info(f"DATA socket conectado a tcp://{self.host}:25556 (PULL)")

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
    async def connect(self) -> bool:
        """
        Lanza MT5 e inicia el bucle de polling para esperar la conexión del EA.
        """
        if self.mt5_exe and not self._is_mt5_alive():
            self._launch_mt5()
        
        # Aumentamos el tiempo de espera y el intervalo de reintento para dar más margen al EA
        max_wait_time = 45 
        retry_interval = 5
        start_time = time.time()
        
        logger.info(f"Verificando conexión inicial con MT5/EA. Tiempo máximo de espera: {max_wait_time}s.")

        while time.time() - start_time < max_wait_time:
            try:
                # Intentamos obtener la info de cuenta. El sys_request intentará PING/reconexión
                info = await self.get_account_info()
                
                # Condición de éxito más robusta: Balance > 0 Y que el nombre de cuenta exista
                if info and info.balance > 0 and info.name: 
                    logger.success("Conexión inicial con MT5/EA exitosa y datos recibidos.")
                    self.is_connected = True
                    if self.auto_restart:
                        if self.mt5_process is not None or not self.mt5_exe:
                             asyncio.create_task(self._watch_mt5())
                             logger.info("Watchdog MT5 iniciado.")
                    return True 

            except Exception:
                pass 
            
            elapsed = int(time.time() - start_time)
            logger.info(f"Conexión MT5 fallida (Reintento en {retry_interval}s)... Tiempo transcurrido: {elapsed}s de {max_wait_time}s.")
            await asyncio.sleep(retry_interval) 
        
        logger.critical(f"Fallo persistente al conectar con MT5/EA después de {max_wait_time} segundos. Saliendo.")
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
    async def sys_request(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Envía un comando al socket REQ y espera la respuesta con reintento y backoff.
        """
        attempt = 0
        while attempt < 5: 
            try:
                await self.sys_socket.send_string(json.dumps(message))
                
                poller = zmq.asyncio.Poller()
                poller.register(self.sys_socket, zmq.POLLIN)
                
                # Aumentamos el timeout para la primera solicitud
                current_timeout = self.sys_timeout * 1000 if attempt == 0 else self.sys_timeout * 1000
                
                socks = dict(await asyncio.wait_for(poller.poll(current_timeout), timeout=self.sys_timeout + 1)) 

                if self.sys_socket in socks:
                    resp = await self.sys_socket.recv_string()
                    resp_strip = resp.strip()
                    
                    if not resp_strip:
                        logger.warning(f"[SYS REQ] Respuesta vacía (Retry {attempt+1}).")
                        # No lanzar excepción, dejar que el flujo continúe al backoff
                    
                    if resp_strip == "OK":
                        return None 
                        
                    try:
                        return json.loads(resp_strip)
                    except json.JSONDecodeError:
                        logger.error(f"[SYS REQ] Respuesta no JSON recibida: {resp_strip}. Reconectando SYS socket.")
                        self.sys_socket = self._reconnect_socket(self.sys_socket, "SYS", 25555, "REQ")
                        raise 
                else:
                    logger.warning(f"[SYS REQ] Timeout de {self.sys_timeout}s. Reconectando SYS socket (Retry {attempt+1}).")
                    self.sys_socket = self._reconnect_socket(self.sys_socket, "SYS", 25555, "REQ")
            
            except asyncio.TimeoutError:
                logger.warning(f"[SYS REQ] Timeout de {self.sys_timeout}s. Reconectando SYS socket (Retry {attempt+1}).")
                self.sys_socket = self._reconnect_socket(self.sys_socket, "SYS", 25555, "REQ")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not isinstance(e, asyncio.CancelledError):
                    logger.error(f"[SYS REQ] Excepción en la solicitud (Retry {attempt+1}): {e}. Reconectando SYS socket.")
                self.sys_socket = self._reconnect_socket(self.sys_socket, "SYS", 25555, "REQ")
            
            attempt += 1
            await asyncio.sleep(min(0.5 * (2**attempt), 5)) 

        logger.critical(f"[SYS REQ] Solicitud fallida después de {attempt} reintentos.")
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
        
        # Eliminamos la llamada redundante a PING. El sys_request maneja la lógica de reconexión.
        response = await self.sys_request({"action": "ACCOUNT"})
        
        if response and response.get('Account'):
            info_dict = response['Account']
            try:
                return AccountInfo(
                    balance=info_dict.get('Balance', 0.0),
                    equity=info_dict.get('Equity', 0.0),
                    margin=info_dict.get('Margin', 0.0),
                    free_margin=info_dict.get('FreeMargin', 0.0),
                    leverage=info_dict.get('Leverage', 1),
                    name=info_dict.get('Name'),
                    server=info_dict.get('Server')
                )
            except Exception as e:
                logger.error(f"Error al mapear datos de cuenta a Schema AccountInfo: {e}")
                return None
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
        """Verifica si el proceso MT5 está en ejecución."""
        if self.mt5_process is None:
            return False
        return self.mt5_process.poll() is None

    def _launch_mt5(self):
        """
        Lanza el ejecutable MT5 con la configuración por defecto 
        y /script para iniciar el EA.
        """
        if self.mt5_exe is None:
            logger.error("[WATCHDOG] Ruta del ejecutable MT5 (mt5_exe) no proporcionada. No se puede lanzar.")
            return

        # 1. Comprobación de existencia (soluciona WinError 2)
        if not os.path.exists(self.mt5_exe):
            logger.critical(f"[WATCHDOG] ERROR FATAL: El archivo MT5 NO EXISTE en la ruta: '{self.mt5_exe}'. ¡Verifica la ruta ABSOLUTA!")
            self.mt5_process = None
            return

        if self._is_mt5_alive():
            self._terminate_mt5()
            
        cmd = [self.mt5_exe]
        
        # 2. Se mantiene sin /portable
        
        # 3. CRUCIAL: Carga el EA automáticamente en el gráfico que haya sido guardado.
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
