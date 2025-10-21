"""
Test Simple de Conexión MT5 ZMQ
Valida que JsonAPI EA está activo y respondiendo.
"""

import asyncio
from underdog.core.connectors.mt5_connector import Mt5Connector

async def test_simple_connection():
    """Test minimalista de conexión"""
    
    print("\n" + "="*70)
    print("🔌 MT5 ZMQ Connection Test (Simple)")
    print("="*70)
    print("\nPREREQUISITOS:")
    print("  1. MT5 abierto manualmente")
    print("  2. JsonAPI EA cargado en un gráfico (EURUSD, cualquier TF)")
    print("  3. EA mostrando 😊 (cara feliz)")
    print("  4. AutoTrading habilitado (botón verde)")
    print("  5. 'Allow DLL imports' habilitado en EA settings")
    print("\n" + "-"*70 + "\n")
    
    connector = None
    try:
        # Crear instancia
        print("📡 Creando instancia de Mt5Connector...")
        connector = Mt5Connector()
        
        # Intentar conectar (3 intentos, 2s entre intentos)
        print("🔄 Intentando conectar con MT5/EA...\n")
        connected = await connector.connect(max_retries=3, retry_delay=2.0)
        
        if not connected:
            print("\n" + "="*70)
            print("❌ CONEXIÓN FALLIDA")
            print("="*70)
            print("\nVer checklist arriba y guía: docs/MT5_JSONAPI_SETUP.md\n")
            return False
        
        # Test 1: Account Info
        print("\n" + "="*70)
        print("✅ CONEXIÓN EXITOSA - Ejecutando Tests")
        print("="*70 + "\n")
        
        print("📊 Test 1: Account Information")
        print("-" * 50)
        info = await connector.sys_request({"action": "ACCOUNT"})
        
        if info:
            print(f"  Broker:          {info.get('broker', 'N/A')}")
            print(f"  Server:          {info.get('server', 'N/A')}")
            print(f"  Currency:        {info.get('currency', 'N/A')}")
            print(f"  Balance:         ${info.get('balance', 0):,.2f}")
            print(f"  Equity:          ${info.get('equity', 0):,.2f}")
            print(f"  Margin:          ${info.get('margin', 0):,.2f}")
            print(f"  Free Margin:     ${info.get('margin_free', 0):,.2f}")
            print(f"  Trading Allowed: {info.get('trading_allowed', False)}")
            print(f"  Bot Trading:     {info.get('bot_trading', False)}")
            print("  ✅ PASSED\n")
        else:
            print("  ❌ FAILED: No response from EA\n")
            return False
        
        # Test 2: Balance (quick request)
        print("💰 Test 2: Balance Quick Request")
        print("-" * 50)
        balance = await connector.sys_request({"action": "BALANCE"})
        
        if balance:
            print(f"  Balance:     ${balance.get('balance', 0):,.2f}")
            print(f"  Equity:      ${balance.get('equity', 0):,.2f}")
            print(f"  Margin:      ${balance.get('margin', 0):,.2f}")
            print(f"  Free Margin: ${balance.get('margin_free', 0):,.2f}")
            print("  ✅ PASSED\n")
        else:
            print("  ❌ FAILED: No response from EA\n")
            return False
        
        # Test 3: Positions
        print("📈 Test 3: Open Positions")
        print("-" * 50)
        positions = await connector.sys_request({"action": "POSITIONS"})
        
        if positions:
            pos_list = positions.get('positions', [])
            print(f"  Open Positions: {len(pos_list)}")
            
            if pos_list and len(pos_list) > 0:
                for i, pos in enumerate(pos_list[:3], 1):  # Show max 3
                    print(f"\n  Position {i}:")
                    print(f"    Symbol: {pos.get('symbol', 'N/A')}")
                    print(f"    Type:   {pos.get('type', 'N/A')}")
                    print(f"    Volume: {pos.get('volume', 0)}")
                    print(f"    Open:   {pos.get('open', 0)}")
            else:
                print("  (No open positions)")
            
            print("  ✅ PASSED\n")
        else:
            print("  ❌ FAILED: No response from EA\n")
            return False
        
        print("="*70)
        print("🎉 TODOS LOS TESTS PASARON")
        print("="*70)
        print("\n✅ La conexión ZMQ entre Python y MT5 JsonAPI está funcionando.")
        print("✅ Puedes proceder a descargar datos históricos o ejecutar trades.\n")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrumpido por el usuario")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if connector:
            try:
                await connector.disconnect()
                print("🔌 Desconectado de MT5\n")
            except:
                pass

if __name__ == "__main__":
    import platform
    
    # CRÍTICO: Solución para advertencia ZMQ en Windows
    # Evita timeouts inesperados en fase de listening continuo
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    success = asyncio.run(test_simple_connection())
    exit(0 if success else 1)
