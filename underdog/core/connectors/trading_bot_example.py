import asyncio
import time
import os # <--- CORRECCIÓN: Importar el módulo 'os' para usar os.name
from typing import Dict, Any, Awaitable, Callable, Optional, List 
# Importamos la clase Mt5Connector desde el archivo que has creado
from mt5_connector import Mt5Connector, logger, AccountInfo, OrderResult, HistoricalData # <--- CORRECCIÓN AQUÍ: Importación directa

# --- 1. Definición de Callbacks (Manejadores de Eventos) ---

async def handle_live_data(data: Dict[str, Any]):
    """
    Callback llamado cuando llegan datos del socket LIVE (15557).
    Típicamente incluye información sobre posiciones cerradas, órdenes ejecutadas,
    cambios en la cuenta, o mensajes de texto del EA.
    """
    # El campo 'type' en los datos LIVE indica el tipo de evento
    event_type = data.get('type', 'UNKNOWN')
    
    if event_type == 'ACCOUNT_UPDATE':
        # Ejemplo: Notificación de un cambio en la cuenta
        logger.info(f"[LIVE] Actualización de cuenta recibida. Equity: {data.get('equity')}")
    elif event_type == 'POSITION_CLOSE':
        # Ejemplo: Notificación de una posición cerrada
        logger.info(f"[LIVE] Posición CERRADA: ID {data.get('position_id')}. Beneficio: {data.get('profit')}")
    else:
        logger.info(f"[LIVE] Dato genérico: {data}")

async def handle_stream_data(data: Dict[str, Any]):
    """
    Callback llamado cuando llegan datos del socket STREAM (15558).
    Típicamente incluye ticks o barras (OHLCV) en tiempo real.
    """
    symbol = data.get('symbol', 'N/A')
    data_type = data.get('type', 'N/A')
    
    if data_type == 'TICK':
        # Datos de tick en tiempo real
        logger.info(f"[STREAM] Tick de {symbol}: Bid={data.get('bid')}, Ask={data.get('ask')}")
    elif data_type == 'BAR':
        # Nueva barra OHLCV completada
        logger.info(f"[STREAM] Nueva BARRA de {symbol} ({data.get('timeframe')}) - Cierre: {data.get('close')}")
    else:
        logger.info(f"[STREAM] Dato genérico de stream: {data}")

# --- 2. Lógica Principal del Bot de Trading ---

class SimpleTradingBot:
    """Clase que encapsula la lógica de trading y usa el conector."""
    
    def __init__(self, connector: Mt5Connector):
        self.connector = connector
        self.running = False

    async def run_strategy(self):
        """
        Bucle principal de la estrategia. Aquí se implementa la lógica
        de trading, se toman decisiones y se envían órdenes.
        """
        self.running = True
        logger.success("Estrategia de trading iniciada.")
        
        # Ejemplo: Verificar la cuenta cada 30 segundos
        while self.running:
            try:
                # 1. Obtener información de la cuenta
                account_info: Optional[AccountInfo] = await self.connector.get_account_info()
                if account_info:
                    logger.info(f"[ESTRATEGIA] Estado actual - Equity: {account_info.equity}, Balance: {account_info.balance}")
                    
                # 2. Ejemplo de solicitud de datos históricos
                symbol_to_check = "EURUSD"
                history: Optional[HistoricalData] = await self.connector.request_history(
                    symbol=symbol_to_check,
                    chartTF="M15", # Marco de tiempo de 15 minutos
                    fromDate="2023.01.01" # Debe ser una fecha válida en formato YYYY.MM.DD
                )
                if history and history.bars:
                    logger.info(f"[ESTRATEGIA] {len(history.bars)} barras históricas de {symbol_to_check} recibidas.")
                
                # 3. Ejemplo de envío de una orden (DESCOMENTAR SOLO PARA TRADING REAL)
                # order_result: Optional[OrderResult] = await self.connector.send_order(
                #     actionType="BUY", 
                #     symbol="EURUSD", 
                #     volume=0.01, 
                #     comment="Test_Order_Python"
                # )
                # if order_result:
                #     logger.info(f"[ESTRATEGIA] Orden enviada. RetCode: {order_result.retcode}, Deal ID: {order_result.deal}")

            except Exception as e:
                logger.error(f"[ESTRATEGIA] Error en el bucle principal: {e}")
            
            # Esperar 30 segundos antes de la siguiente verificación
            await asyncio.sleep(30) 

    def stop(self):
        """Detiene la estrategia."""
        self.running = False
        logger.warning("Estrategia de trading detenida.")

# --- 3. Función de Arranque y Cleanup ---

async def main_startup():
    """Función de arranque principal que gestiona el ciclo de vida de la aplicación."""
    connector = Mt5Connector(auto_restart=True)
    
    # 1. Intentar conectar con MT5/EA
    if not await connector.connect():
        logger.critical("No se pudo establecer conexión con MT5/EA. Saliendo.")
        return

    # 2. Inicializar el bot de trading
    bot = SimpleTradingBot(connector)
    
    # 3. Lanzar tareas asíncronas concurrentes: Listeners ZMQ y Lógica de Estrategia
    # La función run_listeners no devuelve, ya que es un bucle infinito
    tasks = [
        connector.listen_live(handle_live_data),
        connector.listen_stream(handle_stream_data),
        bot.run_strategy()
    ]
    
    # Usamos asyncio.gather para ejecutar las tareas en paralelo
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.warning("Interrupción por teclado (Ctrl+C) detectada.")
    finally:
        bot.stop()
        await connector.disconnect()
        logger.info("Aplicación finalizada.")

if __name__ == '__main__':
    # Fix para la advertencia de RuntimeWarning en Windows (recomendado)
    if os.name == 'nt': # 'nt' es el nombre del sistema operativo para Windows
        try:
            # Importación local para evitar fallos en otros sistemas operativos
            from asyncio import WindowsSelectorEventLoopPolicy
            asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
        except ImportError:
            pass # Si no estamos en Windows o la política no existe, se ignora.

    print("Iniciando Bot de Trading Asíncrono...")
    try:
        # Nota: Asegúrate de tener ZMQ instalado: pip install pyzmq
        asyncio.run(main_startup())
    except Exception as e:
        logger.critical(f"Fallo fatal al iniciar el bucle asyncio: {e}")
