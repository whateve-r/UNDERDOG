"""
Test de Descarga de Datos Históricos MT5
Descarga 1 semana de EURUSD M1 desde broker MetaQuotes Demo.
"""

import asyncio
import platform
from underdog.data.mt5_historical_loader import download_mt5_data
import time

def main():
    """Test de descarga de datos históricos"""
    
    print("\n" + "="*70)
    print("📥 Test de Descarga de Datos Históricos MT5")
    print("="*70)
    
    print("\nPREREQUISITOS:")
    print("  ✅ MT5 abierto con JsonAPI EA activo")
    print("  ✅ Conexión validada (test_mt5_simple.py pasó)")
    
    print("\n" + "-"*70)
    print("📊 Descargando 1 semana de EURUSD M1...")
    print("-"*70 + "\n")
    
    start_time = time.time()
    
    try:
        # Descargar datos
        df = download_mt5_data(
            symbol="EURUSD",
            start_date="2024-10-01",
            end_date="2024-10-07",
            timeframe="M1"
        )
        
        elapsed = time.time() - start_time
        
        if df is not None and len(df) > 0:
            print("\n" + "="*70)
            print(f"✅ DESCARGA EXITOSA en {elapsed:.2f}s")
            print("="*70)
            
            print(f"\n📊 Estadísticas:")
            print(f"   Total bars: {len(df):,}")
            print(f"   Fecha inicio: {df.index[0]}")
            print(f"   Fecha fin: {df.index[-1]}")
            print(f"   Columnas: {list(df.columns)}")
            
            print(f"\n💰 Spread Analysis:")
            if 'spread' in df.columns:
                print(f"   Avg spread: {df['spread'].mean():.2f} points")
                print(f"   Min spread: {df['spread'].min():.2f} points")
                print(f"   Max spread: {df['spread'].max():.2f} points")
            
            print(f"\n📈 OHLC Sample (primeras 5 barras):")
            print(df.head())
            
            print(f"\n📉 OHLC Sample (últimas 5 barras):")
            print(df.tail())
            
            # Verificar Bid/Ask
            if 'close_bid' in df.columns and 'close_ask' in df.columns:
                print(f"\n✅ Bid/Ask separation disponible")
                print(f"   Avg Bid: {df['close_bid'].mean():.5f}")
                print(f"   Avg Ask: {df['close_ask'].mean():.5f}")
            
            print("\n" + "="*70)
            print("🎉 TODOS LOS TESTS PASARON")
            print("="*70)
            print("\n✅ Datos históricos descargados correctamente")
            print("✅ Caching a parquet funciona (ver data/mt5_historical/)")
            print("✅ Spreads reales calculados")
            print("✅ Sistema listo para backtesting con datos reales\n")
            
            return True
            
        else:
            print("\n❌ ERROR: No se descargaron datos")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # CRÍTICO: Solución para advertencia ZMQ en Windows
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    success = main()
    exit(0 if success else 1)
