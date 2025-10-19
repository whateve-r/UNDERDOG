import zmq
import json
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE PUERTOS ---
SYS_PORT = 15555   # REQ/REP para enviar la solicitud
DATA_PORT = 15556  # PULL para recibir los datos
HOST = "127.0.0.1" 
SYS_ADDRESS = f"tcp://{HOST}:{SYS_PORT}"
DATA_ADDRESS = f"tcp://{HOST}:{DATA_PORT}"

# --- CONFIGURACIÓN DE TIMEOUT ---
SYS_TIMEOUT = 5000  # Timeout en REQ (5 segundos)
DATA_TIMEOUT = 20000 # Timeout en PULL (20 segundos - Necesario para que MT5 cargue la data)

def request_and_pull_data(sys_socket, data_socket, symbol, timeframe, from_timestamp):
    """Envía la solicitud de datos y espera la respuesta en el Data Socket.
    Adaptado para recibir directamente el array de velas desde MT5."""
    
    # Estructura JSON de la API para solicitar velas M15 del último mes
    request = {
        "action": "HISTORY",
        "actionType": "DATA",
        "symbol": symbol,
        "chartTF": timeframe,
        "fromDate": from_timestamp,
        "toDate": None 
    }

    try:
        # --- PASO 1: Enviar Solicitud por SYS_PORT (REQ) ---
        print("\n--- Enviando Solicitud ---")
        sys_socket.send_json(request)
        
        # El System Socket (REQ) DEBE responder con 'OK' para confirmar
        sys_ack = sys_socket.recv_string()
        if sys_ack != 'OK':
             print(f"❌ ERROR: La respuesta de reconocimiento del SYS_PORT no fue 'OK'. Respuesta: {sys_ack}")
             return None
        
        print(f"✅ Recibido ACK del EA: {sys_ack}. Esperando datos en DATA_PORT...")

        # --- PASO 2: Recibir Datos CRUDOS por DATA_PORT (PULL) ---
        raw_data_bytes = data_socket.recv()
        
        try:
            data_string = raw_data_bytes.decode('utf-8')
        except UnicodeDecodeError:
            print("❌ ERROR: No se pudo decodificar el mensaje.")
            return None
            
        # Intentar decodificar JSON. Esperamos que sea un array de arrays.
        try:
            data_points = json.loads(data_string)
            
            # ** VERIFICACIÓN CLAVE **: Si no es una lista o está vacía, algo salió mal
            if not isinstance(data_points, list) or len(data_points) == 0:
                print(f"❌ ERROR: La data recibida es un JSON válido pero no es un array de datos (tipo: {type(data_points)}).")
                print(f"Contenido crudo (Primeros 1000 chars): \n{data_string[:1000]}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: La data no es un JSON válido. Error: {e}")
            print(f"Contenido crudo (Primeros 1000 chars): \n{data_string[:1000]}")
            return None

        # --- PASO 3: Procesamiento Exitoso ---
        print(f"\n✅ Éxito! Recibidos {len(data_points)} velas históricas.")
        
        # Mostrar las primeras 5 y las últimas 5 velas para verificación
        print("\nPrimeras 5 Velas (Oldest):")
        for candle in data_points[:5]:
            # Convertir timestamp Unix a formato legible
            candle_time = datetime.fromtimestamp(candle[0]).strftime('%Y-%m-%d %H:%M')
            # Los elementos del array son: [Timestamp, Open, High, Low, Close, Volume]
            print(f"  [{candle_time}] O:{candle[1]} H:{candle[2]} L:{candle[3]} C:{candle[4]} V:{candle[5]}")

        print("\nÚltimas 5 Velas (Most Recent):")
        for candle in data_points[-5:]:
            candle_time = datetime.fromtimestamp(candle[0]).strftime('%Y-%m-%d %H:%M')
            print(f"  [{candle_time}] O:{candle[1]} H:{candle[2]} L:{candle[3]} C:{candle[4]} V:{candle[5]}")

        # Retornar la lista de velas para que tu aplicación la use
        return data_points
        
    except zmq.error.Again:
        print(f"❌ ERROR: Timeout ({DATA_TIMEOUT/1000}s). El EA no envió la data a tiempo por el DATA_PORT ({DATA_PORT}).")
        return None
    except Exception as e:
        print(f"❌ ERROR Desconocido en ZMQ o Python: {e}")
        return None

def main():
    """Función principal para inicializar ZMQ y ejecutar la solicitud."""
    
    # 1. Calcular la fecha de inicio (hace 30 días)
    start_date = datetime.now() - timedelta(days=30)
    FROM_TIMESTAMP = int(start_date.timestamp())

    print(f"Calculando data de GBPUSD M15 desde: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. Configuración del Contexto ZMQ
    context = zmq.Context()
    sys_socket = None
    data_socket = None

    try:
        # Socket para ENVIAR la solicitud (REQ)
        sys_socket = context.socket(zmq.REQ)
        sys_socket.RCVTIMEO = SYS_TIMEOUT 
        sys_socket.connect(SYS_ADDRESS)
        # Socket para RECIBIR los datos (PULL)
        data_socket = context.socket(zmq.PULL)
        data_socket.RCVTIMEO = DATA_TIMEOUT
        data_socket.connect(DATA_ADDRESS)
        
        print(f"Conectado a SYS_PORT ({SYS_PORT}) para enviar solicitud.")
        print(f"Conectado a DATA_PORT ({DATA_PORT}) para recibir datos.")
        
        # Ejecutar la solicitud de datos
        data = request_and_pull_data(sys_socket, data_socket, "GBPUSD", "M15", FROM_TIMESTAMP)
        
        if data:
            print(f"\nProcesamiento finalizado. {len(data)} velas listas para análisis.")

    except Exception as e:
        print(f"⚠️ Error al inicializar sockets: {e}")
    finally:
        # CERRAR Y TERMINAR: Garantiza que los recursos se liberen.
        if sys_socket:
            sys_socket.close()
        if data_socket:
            data_socket.close()
        context.term()
        print("\nRecursos ZMQ liberados.")


# Ejecución del script
if __name__ == "__main__":
    main()
