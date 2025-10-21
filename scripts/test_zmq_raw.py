"""
Test MINIMALISTA de ZMQ - Sin usar Mt5Connector
Conecta directamente a los puertos ZMQ para verificar comunicación básica.
"""

import zmq
import json
import time

HOST = "127.0.0.1"
SYS_PORT = 25555

print("\n" + "="*70)
print("🔬 TEST MINIMALISTA ZMQ - Conexión Directa")
print("="*70)

print(f"\nConfiguracion:")
print(f"  Host: {HOST}")
print(f"  Puerto: {SYS_PORT}")
print(f"  Tipo: REQ (Request)")

# Crear contexto y socket ZMQ
print("\n1️⃣ Creando contexto ZMQ...")
context = zmq.Context()

print("2️⃣ Creando socket REQ...")
socket = context.socket(zmq.REQ)

print(f"3️⃣ Conectando a tcp://{HOST}:{SYS_PORT}...")
socket.connect(f"tcp://{HOST}:{SYS_PORT}")

print("✅ Socket conectado\n")

# Configurar timeout
socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 segundos
socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 segundos
socket.setsockopt(zmq.LINGER, 0)

# Test 1: Enviar request ACCOUNT
print("="*70)
print("📤 TEST 1: Enviando request ACCOUNT")
print("="*70)

request = {"action": "ACCOUNT"}
request_json = json.dumps(request)

print(f"Request: {request_json}")
print("Enviando...")

try:
    socket.send_string(request_json)
    print("✅ Request enviado")
    
    print("\n⏳ Esperando respuesta (timeout 5s)...")
    response = socket.recv_string()
    print(f"✅ Respuesta recibida: {len(response)} caracteres\n")
    
    print("Respuesta completa:")
    print("-" * 70)
    print(response)
    print("-" * 70)
    
    # Parse JSON
    try:
        data = json.loads(response)
        print("\n✅ JSON válido")
        print(f"   Broker: {data.get('broker', 'N/A')}")
        print(f"   Server: {data.get('server', 'N/A')}")
        print(f"   Balance: ${data.get('balance', 0):,.2f}")
        print(f"   Currency: {data.get('currency', 'N/A')}")
        print(f"   Error: {data.get('error', 'N/A')}")
        
        if data.get('error') == False or data.get('error') == 0:
            print("\n🎉 CONEXIÓN EXITOSA - EA respondió correctamente")
        else:
            print(f"\n❌ EA devolvió error: {data}")
            
    except json.JSONDecodeError as e:
        print(f"\n❌ ERROR: Respuesta no es JSON válido")
        print(f"   {e}")
        
except zmq.Again:
    print("❌ TIMEOUT: No se recibió respuesta en 5 segundos")
    print("\nPosibles causas:")
    print("  1. EA no está activo (verificar cara 😊 en gráfico)")
    print("  2. EA no procesó el request (verificar logs de MT5)")
    print("  3. Puerto bloqueado por firewall")
    print("  4. Socket ZMQ en estado inválido")
    
except Exception as e:
    print(f"❌ ERROR INESPERADO: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    print("\n🧹 Limpiando recursos...")
    socket.close()
    context.term()
    print("✅ Socket cerrado\n")

print("="*70)
print("Diagnóstico completo")
print("="*70)
print("\nSi este test falla, el problema NO está en Mt5Connector.")
print("Si este test funciona, el problema está en la lógica de Mt5Connector.\n")
