"""
CONFIGURACIÓN DE HUGGINGFACE PARA UNDERDOG
==========================================

Autor: UNDERDOG Development Team
Fecha: Octubre 2025

PASOS PARA CONFIGURAR HUGGINGFACE
----------------------------------

1. CREAR CUENTA EN HUGGINGFACE (si no tienes una)
   - Ve a: https://huggingface.co/join
   - Crea una cuenta gratuita

2. OBTENER TOKEN DE ACCESO
   - Ve a: https://huggingface.co/settings/tokens
   - Click en "Create new token"
   - Tipo: "Read" (solo lectura es suficiente)
   - Nombre: "UNDERDOG-Backtesting"
   - Click "Generate token"
   - COPIA EL TOKEN (lo necesitarás en el siguiente paso)

3. CONFIGURAR EL TOKEN

   OPCIÓN A - Método Automático (Recomendado):
   
   ```powershell
   poetry run python scripts/setup_hf_token.py
   ```
   
   Cuando aparezca el prompt, pega tu token y presiona Enter.

   OPCIÓN B - Variable de Entorno:
   
   ```powershell
   # Windows PowerShell
   $env:HF_TOKEN = 'hf_xxxxxxxxxxxxxxxxxxxx'
   
   # Luego ejecuta
   poetry run python scripts/setup_hf_token.py
   ```
   
   OPCIÓN C - Edición Manual:
   
   1. Abre: scripts/setup_hf_token.py
   2. Encuentra la línea: login(token='TOKEN_AQUI')
   3. Reemplaza TOKEN_AQUI con tu token
   4. Ejecuta: poetry run python scripts/setup_hf_token.py

4. VERIFICAR LA CONFIGURACIÓN

   Una vez autenticado, prueba cargar datos reales:
   
   ```python
   from underdog.data.hf_loader import HuggingFaceDataHandler
   
   handler = HuggingFaceDataHandler(
       dataset_id='elthariel/histdata_fx_1m',
       symbol='EURUSD',
       start_date='2023-01-01',
       end_date='2023-01-31'
   )
   
   print(f"✓ Datos cargados: {len(handler.df):,} barras")
   ```

5. USAR DATOS REALES EN EL DASHBOARD

   En el Streamlit dashboard:
   - Sidebar → "Data Source"
   - Marca el checkbox "Use HuggingFace Data"
   - Ejecuta el backtest

DATASETS DISPONIBLES
--------------------

1. elthariel/histdata_fx_1m (PRINCIPAL)
   - Datos de Forex a 1 minuto
   - Desde: ~2010
   - Pares: EURUSD, GBPUSD, USDJPY, USDCHF, etc.
   - Fuente: HistData.com
   - Calidad: Alta (datos reales de mercado)

2. Ehsanrs2/Forex_Factory_Calendar
   - Calendario económico
   - Eventos de alto impacto
   - Útil para filtrar news trading

COMPARACIÓN: SINTÉTICO VS REAL
------------------------------

DATOS SINTÉTICOS (Predeterminado):
✅ Rápido (genera instantáneamente)
✅ No requiere autenticación
✅ Útil para pruebas de concepto
❌ No refleja comportamiento real del mercado
❌ Volatilidad artificial
❌ No tiene gaps, slippage realista

DATOS REALES (HuggingFace):
✅ Comportamiento real del mercado
✅ Volatilidad genuina
✅ Gaps, spreads, slippage histórico
✅ Validación robusta de estrategias
❌ Requiere autenticación
⚠️ Descarga inicial más lenta (cacheado después)

RECOMENDACIÓN
-------------

1. DESARROLLO/TESTING → Usa datos sintéticos
2. VALIDACIÓN FINAL → Usa datos reales HuggingFace
3. PRODUCCIÓN → Usa datos reales + validación Monte Carlo

TROUBLESHOOTING
---------------

Error: "Could not load dataset"
→ Solución: Verifica autenticación con huggingface-cli login

Error: "Token is not valid"
→ Solución: Regenera el token en HuggingFace settings

Error: "Dataset not found"
→ Solución: Verifica que el dataset_id sea correcto

Datos muy lentos de cargar:
→ Primera vez: Dataset se descarga (~GB de datos)
→ Siguientes: Usa caché local (mucho más rápido)
→ Para acelerar: reduce el rango de fechas

PRÓXIMOS PASOS
--------------

Después de configurar HuggingFace:

1. Compara resultados sintéticos vs reales para una estrategia
2. Identifica diferencias en métricas (Sharpe, drawdown, win rate)
3. Si la estrategia funciona en datos reales → Optimiza parámetros
4. Si falla en datos reales → Revisa lógica de entrada/salida
5. Monte Carlo con datos reales → Validación robusta

SOPORTE
-------

Documentación HuggingFace: https://huggingface.co/docs
Dataset FOREX: https://huggingface.co/datasets/elthariel/histdata_fx_1m
Issues UNDERDOG: GitHub Issues del proyecto

