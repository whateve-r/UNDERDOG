"""
Test con DATA socket subscrito ANTES de enviar comando
Algunos patrones ZMQ requieren que el subscriber esté listo antes del publisher
"""

import zmq
import json
import time

def test_history_data_first():
    """Test conectando DATA socket ANTES de enviar comando"""
    
    print("\n" + "="*70)
    print("🔍 Test: DATA socket PRIMERO, luego comando")
    print("="*70)
    
    context = zmq.Context()
    
    # PRIMERO: Conectar DATA socket
    data_socket = context.socket(zmq.PULL)
    data_socket.connect("tcp://127.0.0.1:25556")
    print("✅ DATA socket conectado PRIMERO")
    time.sleep(0.5)  # Dar tiempo a que se establezca
    
    # SEGUNDO: Conectar SYS socket
    sys_socket = context.socket(zmq.REQ)
    sys_socket.connect("tcp://127.0.0.1:25555")
    print("✅ SYS socket conectado SEGUNDO")
    time.sleep(0.5)
    
    print("\n" + "-"*70)
    
    # Comando simple: ACCOUNT (que sabemos que funciona)
    command = {"action": "ACCOUNT"}
    
    print(f"📋 Test 1: Comando ACCOUNT (debería funcionar)")
    sys_socket.send_string(json.dumps(command))
    
    sys_socket.setsockopt(zmq.RCVTIMEO, 5000)
    ack = sys_socket.recv_string()
    print(f"✅ ACK: '{ack}'")
    
    if ack.strip() == "OK":
        data_socket.setsockopt(zmq.RCVTIMEO, 10000)
        try:
            data = data_socket.recv_string()
            print(f"✅ Datos ACCOUNT recibidos: {len(data)} caracteres")
        except zmq.Again:
            print("❌ TIMEOUT en datos ACCOUNT")
            sys_socket.close()
            data_socket.close()
            context.term()
            return False
    
    print("\n" + "-"*70)
    print("📋 Test 2: Comando HISTORY con COUNT")
    
    # Ahora probar HISTORY
    command_history = {
        "action": "HISTORY",
        "actionType": "DATA",
        "symbol": "EURUSD",
        "chartTF": "M1",
        "count": 50
    }
    
    sys_socket.send_string(json.dumps(command_history))
    
    sys_socket.setsockopt(zmq.RCVTIMEO, 5000)
    ack = sys_socket.recv_string()
    print(f"✅ ACK: '{ack}'")
    
    if ack.strip() == "OK":
        print("⏳ Esperando datos HISTORY (30s)...")
        data_socket.setsockopt(zmq.RCVTIMEO, 30000)
        try:
            data = data_socket.recv_string()
            print(f"\n✅ Datos HISTORY recibidos: {len(data)} caracteres")
            
            data_json = json.loads(data)
            
            print(f"\n📄 Respuesta completa del EA:")
            print(json.dumps(data_json, indent=2))
            
            if data_json.get('error'):
                print(f"\n❌ ERROR: {data_json}")
                return False
            else:
                rates_count = len(data_json.get('rates', []))
                print(f"\n✅ Barras recibidas: {rates_count}")
                
                if rates_count == 0:
                    print("\n⚠️  WARNING: 0 barras recibidas")
                    print("   Posibles causas:")
                    print("   1. Parámetros de fecha incorrectos")
                    print("   2. Symbol no disponible en el broker")
                    print("   3. Timeframe incorrecto")
                    print("   4. EA no pudo acceder a los datos históricos")
                    return False
                else:
                    first = data_json['rates'][0]
                    last = data_json['rates'][-1]
                    print(f"   Primera barra: {first}")
                    print(f"   Última barra: {last}")
                    print("\n🎉 TEST PASÓ")
                    return True
                
        except zmq.Again:
            print("❌ TIMEOUT en datos HISTORY")
            print("\n💡 El DATA socket funciona para ACCOUNT pero NO para HISTORY")
            print("   Esto sugiere un problema en el EA con el comando HISTORY")
            return False
        except Exception as e:
            print(f"❌ ERROR parseando respuesta: {e}")
            print(f"   Respuesta RAW: {data[:500]}")
            return False
    
    sys_socket.close()
    data_socket.close()
    context.term()
    return False

if __name__ == "__main__":
    success = test_history_data_first()
    exit(0 if success else 1)
