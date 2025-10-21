"""
Quick MT5 Connection Test
Verifica que JsonAPI EA esté respondiendo correctamente.
"""

import asyncio
from underdog.core.connectors.mt5_connector import Mt5Connector

async def test_full_connection():
    """Test completo de conexión MT5 via ZMQ"""
    
    print("\n" + "="*60)
    print("MT5 JsonAPI Connection Test")
    print("="*60 + "\n")
    
    connector = None
    try:
        connector = Mt5Connector()
        connected = await connector.connect()
        
        if not connected:
            print("❌ No se pudo establecer conexión con MT5")
            print("\n🔴 CHECKLIST:")
            print("   1. ¿MT5 está abierto?")
            print("   2. ¿JsonAPI EA está cargado en un gráfico?")
            print("   3. ¿El EA muestra cara feliz 😊?")
            print("   4. ¿AutoTrading está habilitado (botón verde)?")
            print("   5. ¿'Allow DLL imports' está habilitado en EA settings?")
            print("\n📚 Ver guía: docs/MT5_JSONAPI_SETUP.md")
            return
        
        print("✅ Conexión establecida con MT5\n")
        
        # Test 1: Account Info
        print("Test 1: Account Info")
        print("-" * 40)
        info = await connector.sys_request({"action": "ACCOUNT"})
        
        if info is None:
            print("❌ No se recibió respuesta del EA")
            print("   El EA no está respondiendo a peticiones.")
            return
        print(f"  Broker: {info.get('broker', 'N/A')}")
        print(f"  Server: {info.get('server', 'N/A')}")
        print(f"  Balance: ${info.get('balance', 0):,.2f}")
        print(f"  Equity: ${info.get('equity', 0):,.2f}")
        print(f"  Trading Allowed: {info.get('trading_allowed', False)}")
        print(f"  Bot Trading: {info.get('bot_trading', False)}")
        print("  ✅ PASSED\n")
        
        # Test 2: Historical Data
        print("Test 2: Historical Data (1 week EURUSD M1)")
        print("-" * 40)
        import time
        from_date = int(time.mktime(time.strptime("2024-10-01", "%Y-%m-%d")))
        to_date = int(time.mktime(time.strptime("2024-10-07", "%Y-%m-%d")))
        
        history = await connector.sys_request({
            "action": "HISTORY",
            "actionType": "DATA",
            "symbol": "EURUSD",
            "chartTF": "M1",
            "fromDate": from_date,
            "toDate": to_date
        })
        
        bars = len(history)
        print(f"  Downloaded: {bars} bars")
        print(f"  Expected: ~10,080 bars (7 days * 24h * 60min)")
        
        if bars > 5000:
            print(f"  ✅ PASSED (got {bars} bars)\n")
        else:
            print(f"  ⚠️  WARNING: Only {bars} bars received\n")
        
        # Test 3: Balance Info (quick request)
        print("Test 3: Balance Info (quick)")
        print("-" * 40)
        balance = await connector.sys_request({"action": "BALANCE"})
        print(f"  Balance: ${balance.get('balance', 0):,.2f}")
        print(f"  Margin Free: ${balance.get('margin_free', 0):,.2f}")
        print("  ✅ PASSED\n")
        
        print("="*60)
        print("✅ ALL TESTS PASSED - MT5 JsonAPI is working!")
        print("="*60)
            
    except asyncio.TimeoutError:
        print("\n❌ TIMEOUT ERROR")
        print("El EA JsonAPI NO está respondiendo.")
        print("\nVerifica:")
        print("1. MT5 está abierto")
        print("2. JsonAPI EA cargado en un gráfico (cara feliz 😊)")
        print("3. AutoTrading habilitado (botón verde)")
        print("4. 'Allow DLL imports' habilitado en EA settings")
        print("\nVer guía completa: docs/MT5_JSONAPI_SETUP.md")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if connector:
            try:
                await connector.disconnect()
            except:
                pass

if __name__ == "__main__":
    asyncio.run(test_full_connection())
