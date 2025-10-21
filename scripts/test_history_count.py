"""
Test alternativo: HISTORY con COUNT en lugar de fechas
"""

import zmq
import json

def test_history_count():
    """Test del comando HISTORY usando COUNT en lugar de fechas"""
    
    print("\n" + "="*70)
    print("🔍 Test: HISTORY con COUNT (últimas 100 barras)")
    print("="*70)
    
    context = zmq.Context()
    
    # Socket SYS (REQ)
    sys_socket = context.socket(zmq.REQ)
    sys_socket.connect("tcp://127.0.0.1:25555")
    print("✅ SYS socket conectado")
    
    # Socket DATA (PULL)
    data_socket = context.socket(zmq.PULL)
    data_socket.connect("tcp://127.0.0.1:25556")
    print("✅ DATA socket conectado")
    
    print("\n" + "-"*70)
    
    # Comando HISTORY con COUNT
    command = {
        "action": "HISTORY",
        "actionType": "DATA",
        "symbol": "EURUSD",
        "chartTF": "M1",
        "count": 100  # Últimas 100 barras
    }
    
    print(f"📋 Comando: {json.dumps(command, indent=2)}")
    
    # Enviar
    sys_socket.send_string(json.dumps(command))
    print("\n✅ Comando enviado")
    
    # Esperar ACK
    print("⏳ Esperando ACK...")
    sys_socket.setsockopt(zmq.RCVTIMEO, 5000)
    
    try:
        ack = sys_socket.recv_string()
        print(f"✅ ACK: '{ack}'")
        
        if ack.strip() == "OK":
            print("\n⏳ Esperando datos (30s timeout)...")
            data_socket.setsockopt(zmq.RCVTIMEO, 30000)
            
            try:
                data = data_socket.recv_string()
                print(f"\n✅ Datos recibidos: {len(data)} caracteres")
                
                data_json = json.loads(data)
                
                if data_json.get('error'):
                    print(f"\n❌ ERROR:")
                    print(f"   {data_json}")
                else:
                    print(f"\n✅ ÉXITO:")
                    if 'rates' in data_json:
                        print(f"   Barras recibidas: {len(data_json['rates'])}")
                        if len(data_json['rates']) > 0:
                            first = data_json['rates'][0]
                            last = data_json['rates'][-1]
                            print(f"   Primera: time={first.get('time')}, close={first.get('close')}")
                            print(f"   Última: time={last.get('time')}, close={last.get('close')}")
                    
                    print("\n🎉 TEST PASÓ")
                    return True
                    
            except zmq.Again:
                print("\n❌ TIMEOUT esperando datos")
                return False
                
    except zmq.Again:
        print("\n❌ TIMEOUT esperando ACK")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False
    finally:
        sys_socket.close()
        data_socket.close()
        context.term()

if __name__ == "__main__":
    success = test_history_count()
    exit(0 if success else 1)
