"""
Test HISTORY con OUTPUT COMPLETO de la respuesta del EA
"""

import zmq
import json

def test_history_full_output():
    """Test HISTORY mostrando respuesta completa del EA"""
    
    print("\n" + "="*70)
    print("🔍 Test HISTORY - Output Completo")
    print("="*70)
    
    context = zmq.Context()
    
    sys_socket = context.socket(zmq.REQ)
    sys_socket.connect("tcp://127.0.0.1:25555")
    
    data_socket = context.socket(zmq.PULL)
    data_socket.connect("tcp://127.0.0.1:25556")
    
    print("✅ Sockets conectados\n")
    
    # Comando HISTORY
    command = {
        "action": "HISTORY",
        "actionType": "DATA",
        "symbol": "EURUSD",
        "chartTF": "M1",
        "count": 50
    }
    
    print(f"📤 Enviando: {json.dumps(command, indent=2)}\n")
    
    sys_socket.send_string(json.dumps(command))
    
    # ACK
    sys_socket.setsockopt(zmq.RCVTIMEO, 5000)
    ack = sys_socket.recv_string()
    print(f"📨 ACK: '{ack}'\n")
    
    if ack.strip() != "OK":
        print(f"❌ ACK inesperado")
        return False
    
    # Datos
    print("⏳ Esperando datos (30s)...\n")
    data_socket.setsockopt(zmq.RCVTIMEO, 30000)
    
    try:
        data = data_socket.recv_string()
        print(f"✅ Recibidos {len(data)} caracteres\n")
        print("=" * 70)
        print("RESPUESTA COMPLETA DEL EA:")
        print("=" * 70)
        
        data_json = json.loads(data)
        print(json.dumps(data_json, indent=2))
        
        print("\n" + "=" * 70)
        print("ANÁLISIS:")
        print("=" * 70)
        
        if data_json.get('error'):
            print(f"❌ ERROR: True")
            print(f"   lastError: {data_json.get('lastError')}")
            print(f"   description: {data_json.get('description')}")
            print(f"   function: {data_json.get('function')}")
            return False
        else:
            rates = data_json.get('rates', [])
            print(f"✅ error: False")
            print(f"✅ Número de barras: {len(rates)}")
            
            if len(rates) > 0:
                print(f"\n📊 Primera barra:")
                print(f"   {json.dumps(rates[0], indent=4)}")
                print(f"\n📊 Última barra:")
                print(f"   {json.dumps(rates[-1], indent=4)}")
                print("\n🎉 SUCCESS!")
                return True
            else:
                print("\n⚠️  0 barras recibidas")
                print("\n💡 POSIBLES CAUSAS:")
                print("   1. chartTF incorrecto (¿EA espera número en lugar de string?)")
                print("   2. count no soportado sin fromDate/toDate")
                print("   3. Symbol no disponible")
                print("   4. EA tiene bug con CopyRates()")
                return False
                
    except zmq.Again:
        print("❌ TIMEOUT - No se recibieron datos en 30s")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        sys_socket.close()
        data_socket.close()
        context.term()

if __name__ == "__main__":
    success = test_history_full_output()
    exit(0 if success else 1)
