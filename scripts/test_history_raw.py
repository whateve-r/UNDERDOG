"""
Test RAW de comando HISTORY para diagnosticar problema de timeout
"""

import zmq
import json
import time

def test_history_raw():
    """Test directo del comando HISTORY sin abstracciones"""
    
    print("\n" + "="*70)
    print("🔍 Test RAW: Comando HISTORY")
    print("="*70)
    
    context = zmq.Context()
    
    # Socket SYS (REQ)
    sys_socket = context.socket(zmq.REQ)
    sys_socket.connect("tcp://127.0.0.1:25555")
    print("✅ SYS socket conectado (REQ)")
    
    # Socket DATA (PULL)
    data_socket = context.socket(zmq.PULL)
    data_socket.connect("tcp://127.0.0.1:25556")
    print("✅ DATA socket conectado (PULL)")
    
    print("\n" + "-"*70)
    print("📤 Enviando comando HISTORY...")
    print("-"*70)
    
    # Comando HISTORY
    command = {
        "action": "HISTORY",
        "actionType": "DATA",
        "symbol": "EURUSD",
        "chartTF": "M1",
        "fromDate": 1727737200,  # 2024-10-01
        "toDate": 1728255600     # 2024-10-07
    }
    
    print(f"\n📋 Comando: {json.dumps(command, indent=2)}")
    
    # Enviar por SYS
    sys_socket.send_string(json.dumps(command))
    print("\n✅ Comando enviado por SYS socket")
    
    # Esperar ACK por SYS
    print("\n⏳ Esperando ACK por SYS socket (timeout 5s)...")
    sys_socket.setsockopt(zmq.RCVTIMEO, 5000)
    
    try:
        ack = sys_socket.recv_string()
        print(f"✅ ACK recibido: '{ack}'")
        
        if ack.strip() == "OK":
            print("\n⏳ Esperando datos por DATA socket (timeout 60s)...")
            data_socket.setsockopt(zmq.RCVTIMEO, 60000)
            
            try:
                data = data_socket.recv_string()
                print(f"\n✅ Datos recibidos: {len(data)} caracteres")
                
                # Parse JSON
                try:
                    data_json = json.loads(data)
                    
                    if data_json.get('error'):
                        print(f"\n❌ ERROR en respuesta del EA:")
                        print(f"   lastError: {data_json.get('lastError')}")
                        print(f"   description: {data_json.get('description')}")
                        print(f"   function: {data_json.get('function')}")
                    else:
                        print(f"\n✅ Respuesta válida:")
                        print(f"   Tiene campo 'rates': {'rates' in data_json}")
                        if 'rates' in data_json:
                            print(f"   Número de barras: {len(data_json['rates'])}")
                            if len(data_json['rates']) > 0:
                                print(f"   Primera barra: {data_json['rates'][0]}")
                                print(f"   Última barra: {data_json['rates'][-1]}")
                    
                    print("\n🎉 TEST EXITOSO")
                    return True
                    
                except json.JSONDecodeError as e:
                    print(f"\n❌ ERROR: Respuesta no es JSON válido")
                    print(f"   Error: {e}")
                    print(f"   Primeros 500 chars: {data[:500]}")
                    return False
                    
            except zmq.Again:
                print("\n❌ TIMEOUT: No se recibieron datos por DATA socket en 60s")
                print("\n💡 POSIBLES CAUSAS:")
                print("   1. EA no está procesando el comando HISTORY correctamente")
                print("   2. Parámetros del comando son incorrectos")
                print("   3. EA tiene un error interno")
                print("\n📋 REVISA LOS LOGS DE MT5 (Tab 'Experts'):")
                print("   - ¿Hay algún mensaje de error del JsonAPI EA?")
                print("   - ¿Dice 'Processing: {...}' con tu comando?")
                print("   - ¿Hay algún mensaje de 'InformClientSocket' o 'ERROR'?")
                return False
        else:
            print(f"\n❌ ACK inesperado (esperaba 'OK'): {ack}")
            return False
            
    except zmq.Again:
        print("\n❌ TIMEOUT: No se recibió ACK por SYS socket en 5s")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys_socket.close()
        data_socket.close()
        context.term()
        print("\n🔌 Sockets cerrados")

if __name__ == "__main__":
    success = test_history_raw()
    exit(0 if success else 1)
