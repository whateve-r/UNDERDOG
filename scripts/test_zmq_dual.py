"""
Test ZMQ con patrón DUAL: SYS (REQ/REP) + DATA (PULL)
El EA envía "OK" por SYS, pero los datos reales van por DATA socket.
"""

import zmq
import json
import time

HOST = "127.0.0.1"
SYS_PORT = 25555
DATA_PORT = 25556

print("\n" + "="*70)
print("🔬 TEST ZMQ - Patrón Dual (SYS + DATA)")
print("="*70)

# Crear contexto
context = zmq.Context()

# Socket 1: SYS (REQ para enviar comandos)
print("\n1️⃣ Creando socket SYS (REQ)...")
sys_socket = context.socket(zmq.REQ)
sys_socket.connect(f"tcp://{HOST}:{SYS_PORT}")
sys_socket.setsockopt(zmq.RCVTIMEO, 5000)
sys_socket.setsockopt(zmq.SNDTIMEO, 5000)
sys_socket.setsockopt(zmq.LINGER, 0)
print(f"   ✅ Conectado a tcp://{HOST}:{SYS_PORT}")

# Socket 2: DATA (PULL para recibir datos)
print("\n2️⃣ Creando socket DATA (PULL)...")
data_socket = context.socket(zmq.PULL)
data_socket.connect(f"tcp://{HOST}:{DATA_PORT}")
data_socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 segundos
data_socket.setsockopt(zmq.LINGER, 0)
print(f"   ✅ Conectado a tcp://{HOST}:{DATA_PORT}")

print("\n" + "="*70)
print("📤 TEST: Request ACCOUNT")
print("="*70)

request = {"action": "ACCOUNT"}
request_json = json.dumps(request)

print(f"\nRequest: {request_json}")
print("Enviando por SYS socket...")

try:
    # Paso 1: Enviar request por SYS
    sys_socket.send_string(request_json)
    print("✅ Request enviado")
    
    # Paso 2: Recibir ACK por SYS
    print("\n⏳ Esperando ACK por SYS socket...")
    ack = sys_socket.recv_string()
    print(f"✅ ACK recibido: '{ack}'")
    
    if ack.strip() == "OK":
        print("   → EA confirmó que procesará el request\n")
    else:
        print(f"   ⚠️  ACK inesperado: '{ack}'\n")
    
    # Paso 3: Recibir datos por DATA socket
    print("⏳ Esperando datos por DATA socket (timeout 10s)...")
    response = data_socket.recv_string()
    print(f"✅ Datos recibidos: {len(response)} caracteres\n")
    
    print("Respuesta completa:")
    print("-" * 70)
    print(response)
    print("-" * 70)
    
    # Parse JSON
    try:
        data = json.loads(response)
        print("\n✅ JSON válido")
        print(f"\n📊 Información de Cuenta:")
        print(f"   Broker:          {data.get('broker', 'N/A')}")
        print(f"   Server:          {data.get('server', 'N/A')}")
        print(f"   Currency:        {data.get('currency', 'N/A')}")
        print(f"   Balance:         ${data.get('balance', 0):,.2f}")
        print(f"   Equity:          ${data.get('equity', 0):,.2f}")
        print(f"   Margin:          ${data.get('margin', 0):,.2f}")
        print(f"   Free Margin:     ${data.get('margin_free', 0):,.2f}")
        print(f"   Trading Allowed: {bool(data.get('trading_allowed', 0))}")
        print(f"   Bot Trading:     {bool(data.get('bot_trading', 0))}")
        print(f"   Error:           {data.get('error', 'N/A')}")
        
        if data.get('error') == False or data.get('error') == 0:
            print("\n" + "="*70)
            print("🎉 ÉXITO TOTAL - Comunicación ZMQ funcionando correctamente")
            print("="*70)
            print("\n✅ El patrón es: REQ (SYS) → 'OK' → PULL (DATA) → JSON")
            print("✅ Mt5Connector DEBE leer del DATA socket después del 'OK'\n")
        else:
            print(f"\n❌ EA devolvió error: {data}")
            
    except json.JSONDecodeError as e:
        print(f"\n❌ ERROR: Respuesta no es JSON válido")
        print(f"   {e}")
        
except zmq.Again as e:
    print(f"❌ TIMEOUT: {e}")
    print("\nPosibles causas:")
    print("  1. EA no envió datos por DATA socket")
    print("  2. Datos enviados por otro socket (LIVE/STREAM)")
    print("  3. EA tiene error interno al procesar ACCOUNT")
    print("\n💡 Verifica logs de MT5 (Toolbox → Experts)")
    
except Exception as e:
    print(f"❌ ERROR INESPERADO: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n🧹 Limpiando recursos...")
    sys_socket.close()
    data_socket.close()
    context.term()
    print("✅ Sockets cerrados\n")
