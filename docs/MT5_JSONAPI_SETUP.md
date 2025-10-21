# MT5 JsonAPI EA Setup Guide

## Problema Actual

El EA JsonAPI.ex5 existe pero no está activo en ningún gráfico de MT5. Los mensajes de log muestran:
```
[WARN] [SYS REQ] Timeout de 3.0s. Reconectando SYS socket (Retry 1-4).
[CRIT] Fallo persistente al conectar con MT5/EA después de 45 segundos.
```

Esto significa que MT5 está corriendo pero el EA no está "bind" a los puertos ZMQ.

---

## Solución 1: Cargar EA Manualmente (RECOMENDADO - 2 min)

### Paso 1: Abrir MT5 Terminal
1. Abre MetaTrader 5 manualmente
2. Login a tu cuenta DEMO

### Paso 2: Cargar JsonAPI en un Gráfico
1. En MT5, ve a **Navigator** (Ctrl+N si está oculto)
2. Despliega **Expert Advisors**
3. Encuentra **JsonAPI** en la lista
4. **Arrastra y suelta** JsonAPI sobre cualquier gráfico (EURUSD M15 por ejemplo)

### Paso 3: Configurar Parámetros del EA
Cuando aparezca el diálogo de configuración:

**Pestaña "Common":**
- ✅ Allow live trading
- ✅ Allow DLL imports (CRÍTICO para ZMQ)
- ✅ Allow WebRequest for listed URL

**Pestaña "Inputs":** (deben coincidir con tu config)
```
HOST = 127.0.0.1
SYS_PORT = 25555
DATA_PORT = 25556
LIVE_PORT = 25557
STR_PORT = 25558
debug = true
liveStream = true
```

### Paso 4: Verificar EA Activo
En la esquina superior derecha del gráfico debes ver:
```
JsonAPI 1.12  😊  (cara feliz = EA corriendo)
```

Si ves 😞 (cara triste), el EA NO está activo. Revisa:
- AutoTrading habilitado (botón verde en toolbar)
- DLL imports permitidos
- No hay errores en la pestaña "Experts" del Terminal

### Paso 5: Verificar Logs de MT5
En MT5, abre **Toolbox** → **Experts** tab. Deberías ver:
```
Binding 'System' socket on port 25555...
Binding 'Data' socket on port 25556...
Binding 'Live' socket on port 25557...
Binding 'Streaming' socket on port 25558...
```

Si ves estos mensajes, el EA está **ACTIVO** ✅

### Paso 6: Test de Conexión Python
```powershell
poetry run python -c "from underdog.data.mt5_historical_loader import download_mt5_data; df = download_mt5_data('EURUSD', '2024-10-01', '2024-10-07', 'M1'); print(f'Downloaded {len(df)} bars'); print(df.head())"
```

**Esperado:** Download completo sin timeouts.

---

## Solución 2: Crear Perfil MT5 con EA Pre-cargado (PERSISTENTE)

Si quieres que el EA se cargue automáticamente cada vez que abres MT5:

### Paso 1: Cargar EA Manualmente (como en Solución 1)

### Paso 2: Guardar Perfil
1. En MT5, ve a **File** → **Save Profile As...**
2. Nombre: `JsonAPI_Default`
3. Click **Save**

### Paso 3: Configurar como Perfil por Defecto
1. **File** → **Open Data Folder**
2. Abre `profiles\JsonAPI_Default`
3. Copia el archivo `order.wnd` o `chart01.chr`
4. Pégalo en `profiles\default\`

Ahora cada vez que MT5 se inicie, el EA JsonAPI estará pre-cargado.

---

## Solución 3: Automatizar Inicio con Python (AVANZADO)

Modificar `mt5_connector.py` para usar MetaTrader5 library y cargar el EA programáticamente:

```python
import MetaTrader5 as mt5

# Inicializar MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

# Cargar EA en gráfico EURUSD M15
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M15

# Crear gráfico si no existe
chart_id = mt5.chart_open(symbol, timeframe)

# Cargar EA JsonAPI
ea_path = "Experts\\JsonAPI.ex5"
mt5.chart_add_expert(chart_id, ea_path)
```

**PROBLEMA:** Esto requiere refactorizar `mt5_connector.py` y mezclar ZMQ con MetaTrader5 library.

---

## Troubleshooting

### Error: "DLL imports not allowed"
- Solución: En configuración del EA, marca **Allow DLL imports**

### Error: "AutoTrading is disabled"
- Solución: Click botón **AutoTrading** en toolbar de MT5 (debe estar verde)

### Error: EA no aparece en Navigator
- Solución: Compila `JsonAPI.mq5`:
  1. Abre MetaEditor (F4 en MT5)
  2. File → Open → `JsonAPI.mq5`
  3. Compile (F7)
  4. Verifica que `JsonAPI.ex5` esté en `MQL5\Experts\`

### Puertos ZMQ ocupados
```powershell
# Verificar puertos en uso
netstat -ano | findstr "25555"
netstat -ano | findstr "25556"

# Si hay procesos, matalos
taskkill /PID <PID> /F
```

### EA se desconecta tras unos minutos
- Causa: MT5 cierra EAs inactivos
- Solución: En código MQL5, asegura que `OnTimer()` se ejecuta cada 1ms:
  ```cpp
  EventSetMillisecondTimer(1);  // Ya está en tu código ✅
  ```

---

## Verificación Final

Ejecuta este script de test completo:

```python
import asyncio
from underdog.core.connectors.mt5_connector import Mt5Connector

async def test_connection():
    async with Mt5Connector() as connector:
        # Test 1: Account Info
        info = await connector.sys_request({"action": "ACCOUNT"})
        print(f"✅ Account: {info}")
        
        # Test 2: Historical Data
        history = await connector.sys_request({
            "action": "HISTORY",
            "actionType": "DATA",
            "symbol": "EURUSD",
            "chartTF": "M1",
            "fromDate": 1727740800  # 2024-10-01
        })
        print(f"✅ Historical bars: {len(history)}")

asyncio.run(test_connection())
```

**Esperado:**
```
✅ Account: {'broker': 'ICMarkets', 'balance': 10000.0, ...}
✅ Historical bars: 10080
```

---

## Resumen

**Para empezar YA (2 minutos):**
1. Abre MT5 manualmente
2. Arrastra **JsonAPI** desde Navigator a cualquier gráfico
3. Habilita **Allow DLL imports** + **Allow live trading**
4. Verifica cara feliz 😊 en gráfico
5. Run test de conexión Python

**Para setup permanente (5 minutos extra):**
- Guarda perfil con EA pre-cargado
- MT5 auto-cargará JsonAPI en cada inicio

¡Listo para descargar datos históricos reales del broker! 🚀
