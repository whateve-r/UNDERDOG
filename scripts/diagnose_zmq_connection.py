"""
Diagnóstico Completo de Conexión ZMQ MT5
Verifica todos los puntos críticos para establecer comunicación con JsonAPI EA.
"""

import socket
import subprocess
import sys
import zmq
import time

def print_section(title):
    """Imprime una sección del diagnóstico"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_port_listening(port):
    """Verifica si un puerto está escuchando (LISTEN state)"""
    try:
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                # Extract PID
                parts = line.split()
                pid = parts[-1] if parts else "Unknown"
                return True, pid
        
        return False, None
    except Exception as e:
        return False, f"Error: {e}"

def check_port_connectable(host, port):
    """Intenta conectar a un puerto específico"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        return False

def test_zmq_connection(host, port, socket_type):
    """Test directo de conexión ZMQ"""
    context = zmq.Context()
    
    try:
        # Crear socket según tipo
        if socket_type == "REQ":
            sock = context.socket(zmq.REQ)
        else:
            sock = context.socket(zmq.PULL)
        
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout
        
        # Conectar
        sock.connect(f"tcp://{host}:{port}")
        
        # Si es REQ, enviar mensaje de prueba
        if socket_type == "REQ":
            sock.send_json({"action": "ACCOUNT"})
            try:
                response = sock.recv_string()
                return True, f"Response: {response[:50]}..."
            except zmq.Again:
                return False, "Timeout esperando respuesta (EA no responde)"
        else:
            # Para PULL, solo verificamos que conecte
            return True, "Socket PULL conectado (no hay datos esperados)"
            
    except zmq.ZMQError as e:
        return False, f"ZMQ Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        sock.close()
        context.term()

def check_mt5_process():
    """Verifica si MT5 está corriendo"""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq terminal64.exe'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if 'terminal64.exe' in result.stdout:
            # Extract PID
            for line in result.stdout.split('\n'):
                if 'terminal64.exe' in line:
                    parts = line.split()
                    pid = parts[1] if len(parts) > 1 else "Unknown"
                    return True, pid
        
        return False, None
    except Exception as e:
        return False, f"Error: {e}"

def check_firewall_status():
    """Verifica estado del Firewall de Windows"""
    try:
        result = subprocess.run(
            ['netsh', 'advfirewall', 'show', 'currentprofile'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if 'State' in result.stdout or 'Estado' in result.stdout:
            for line in result.stdout.split('\n'):
                if 'State' in line or 'Estado' in line:
                    return True, line.strip()
        
        return True, "No se pudo determinar"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Ejecuta diagnóstico completo"""
    
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  🔍 DIAGNÓSTICO ZMQ MT5 - JsonAPI EA".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Configuración
    HOST = "127.0.0.1"
    PORTS = {
        "SYS (REQ/REP)": (25555, "REQ"),
        "DATA (PUSH/PULL)": (25556, "PULL"),
        "LIVE (PUSH/PULL)": (25557, "PULL"),
        "STREAM (PUSH/PULL)": (25558, "PULL")
    }
    
    issues_found = []
    warnings_found = []
    
    # ============================================================
    # 1. VERIFICAR PROCESO MT5
    # ============================================================
    print_section("1. Proceso MetaTrader 5")
    
    mt5_running, mt5_pid = check_mt5_process()
    
    if mt5_running:
        print(f"✅ MT5 está corriendo (PID: {mt5_pid})")
    else:
        print(f"❌ MT5 NO está corriendo")
        issues_found.append("MT5 no está ejecutándose. Inicia terminal64.exe")
        print("\n🔧 Solución: Abre MetaTrader 5 manualmente")
    
    # ============================================================
    # 2. VERIFICAR FIREWALL
    # ============================================================
    print_section("2. Estado del Firewall de Windows")
    
    fw_running, fw_status = check_firewall_status()
    
    if fw_running:
        print(f"ℹ️  Firewall: {fw_status}")
        if 'ON' in fw_status.upper() or 'ACTIVADO' in fw_status.upper():
            warnings_found.append(
                "Firewall está ACTIVO. Si hay problemas, prueba desactivarlo temporalmente:\n"
                "   Control Panel → Windows Defender Firewall → Turn off"
            )
    else:
        print(f"⚠️  No se pudo verificar Firewall")
    
    # ============================================================
    # 3. VERIFICAR PUERTOS EN LISTENING
    # ============================================================
    print_section("3. Puertos ZMQ en Estado LISTENING")
    
    print(f"\nVerificando si MT5/EA está haciendo BIND en los puertos...\n")
    
    all_ports_listening = True
    
    for port_name, (port, _) in PORTS.items():
        listening, pid = check_port_listening(port)
        
        if listening:
            print(f"✅ Puerto {port} ({port_name}): LISTENING (PID: {pid})")
        else:
            print(f"❌ Puerto {port} ({port_name}): NO LISTENING")
            all_ports_listening = False
            issues_found.append(
                f"Puerto {port} no está en estado LISTENING. "
                f"El EA JsonAPI NO está activo o no hizo BIND correctamente."
            )
    
    if not all_ports_listening:
        print("\n" + "⚠️ " * 20)
        print("🔴 PROBLEMA CRÍTICO DETECTADO:")
        print("   Los puertos NO están en LISTENING → EA JsonAPI NO está activo\n")
        print("🔧 SOLUCIÓN:")
        print("   1. Abre MT5")
        print("   2. Navigator (Ctrl+N) → Expert Advisors → JsonAPI")
        print("   3. Arrastra JsonAPI al gráfico EURUSD M15")
        print("   4. En configuración del EA:")
        print("      ✅ Allow live trading")
        print("      ✅ Allow DLL imports  ← CRÍTICO para ZMQ")
        print("   5. Verifica esquina superior derecha: 'JsonAPI 1.12 😊'")
        print("   6. Toolbox (Ctrl+T) → Experts → Debes ver:")
        print("      'Binding System socket on port 25555...'")
        print("      'Binding Data socket on port 25556...'")
        print("      'Binding Live socket on port 25557...'")
        print("      'Binding Streaming socket on port 25558...'")
        print("⚠️ " * 20)
    
    # ============================================================
    # 4. TEST DE CONECTIVIDAD TCP
    # ============================================================
    print_section("4. Conectividad TCP a los Puertos")
    
    print(f"\nIntentando conectar vía TCP a {HOST}...\n")
    
    for port_name, (port, _) in PORTS.items():
        connectable = check_port_connectable(HOST, port)
        
        if connectable:
            print(f"✅ Puerto {port} ({port_name}): CONNECTABLE")
        else:
            print(f"❌ Puerto {port} ({port_name}): NO CONNECTABLE")
            if not all_ports_listening:
                # Ya sabemos el problema (EA no activo)
                pass
            else:
                warnings_found.append(
                    f"Puerto {port} está LISTENING pero no es CONNECTABLE. "
                    f"Posible bloqueo de Firewall local."
                )
    
    # ============================================================
    # 5. TEST DE CONEXIÓN ZMQ
    # ============================================================
    print_section("5. Test de Conexión ZMQ (REQ/REP)")
    
    if all_ports_listening:
        print(f"\nIntentando comunicación ZMQ con EA en puerto 25555...\n")
        
        success, message = test_zmq_connection(HOST, 25555, "REQ")
        
        if success:
            print(f"✅ Conexión ZMQ EXITOSA!")
            print(f"   {message}")
            print("\n🎉 LA CONEXIÓN ESTÁ FUNCIONANDO CORRECTAMENTE")
        else:
            print(f"❌ Conexión ZMQ FALLIDA")
            print(f"   {message}")
            issues_found.append(
                "Conexión ZMQ falla a pesar de que puerto está LISTENING. "
                "Posibles causas:\n"
                "   - EA no está procesando mensajes (OnTimer() no ejecutándose)\n"
                "   - DLL imports bloqueados en MT5\n"
                "   - Librería ZMQ.mqh no compilada correctamente"
            )
    else:
        print("\n⏭️  Saltando test ZMQ (puertos no están LISTENING)")
    
    # ============================================================
    # 6. VERIFICAR CONFIGURACIÓN PYTHON
    # ============================================================
    print_section("6. Configuración Python (mt5_credentials.yaml)")
    
    try:
        import yaml
        config_path = r"C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG\config\runtime\env\mt5_credentials.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"\n✅ Archivo de configuración encontrado")
        print(f"   Host: {config.get('zmq_host', 'N/A')}")
        print(f"   SYS Port: {config.get('sys_port', 'N/A')}")
        print(f"   DATA Port: {config.get('data_port', 'N/A')}")
        print(f"   LIVE Port: {config.get('live_port', 'N/A')}")
        print(f"   STREAM Port: {config.get('stream_port', 'N/A')}")
        
        # Verificar que coincidan
        if config.get('zmq_host') != HOST:
            warnings_found.append(
                f"Host en config ({config.get('zmq_host')}) != {HOST}"
            )
        
        if config.get('sys_port') != 25555:
            issues_found.append(
                f"Puerto SYS en config ({config.get('sys_port')}) != 25555 (esperado)"
            )
            
    except Exception as e:
        print(f"⚠️  Error leyendo configuración: {e}")
    
    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    print_section("📊 RESUMEN DEL DIAGNÓSTICO")
    
    if not issues_found and not warnings_found:
        print("\n🎉 TODO ESTÁ PERFECTO")
        print("   No se encontraron problemas críticos ni advertencias.")
        print("\n✅ La conexión ZMQ entre Python y MT5 JsonAPI está funcionando.")
        
    else:
        if issues_found:
            print(f"\n🔴 PROBLEMAS CRÍTICOS ENCONTRADOS ({len(issues_found)}):\n")
            for i, issue in enumerate(issues_found, 1):
                print(f"{i}. {issue}\n")
        
        if warnings_found:
            print(f"\n⚠️  ADVERTENCIAS ({len(warnings_found)}):\n")
            for i, warning in enumerate(warnings_found, 1):
                print(f"{i}. {warning}\n")
    
    print("\n" + "="*70)
    print("\n📚 Documentación: docs/MT5_JSONAPI_SETUP.md")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnóstico interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
