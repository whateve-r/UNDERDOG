# Script de Limpieza Automatizada - Proyecto UNDERDOG
# Basado en: docs/AUDIT_CLEANUP_2025_10_22.md
# IMPORTANTE: Hacer backup antes de ejecutar (git commit)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "🔍 UNDERDOG - Script de Limpieza" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Verificar que estamos en el directorio correcto
if (-not (Test-Path "underdog")) {
    Write-Host "❌ ERROR: Ejecuta este script desde la raíz del proyecto UNDERDOG" -ForegroundColor Red
    exit 1
}

Write-Host "⚠️  ADVERTENCIA: Este script eliminará 46 archivos obsoletos." -ForegroundColor Yellow
Write-Host "   Asegúrate de haber hecho commit de tus cambios." -ForegroundColor Yellow
Write-Host "`n¿Continuar? (S/N): " -ForegroundColor Yellow -NoNewline
$confirm = Read-Host

if ($confirm -ne "S" -and $confirm -ne "s") {
    Write-Host "`n❌ Cancelado por el usuario.`n" -ForegroundColor Red
    exit 0
}

$deletedCount = 0
$failedCount = 0

function Remove-FileIfExists {
    param($path)
    if (Test-Path $path) {
        try {
            Remove-Item $path -Force
            Write-Host "  ✓ Eliminado: $path" -ForegroundColor Green
            return $true
        }
        catch {
            Write-Host "  ✗ ERROR eliminando: $path" -ForegroundColor Red
            return $false
        }
    }
    else {
        Write-Host "  - No existe: $path" -ForegroundColor Gray
        return $null
    }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 1: Scripts de Diagnóstico ZMQ (8 archivos)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$files = @(
    "scripts/test_zmq_raw.py",
    "scripts/test_zmq_dual.py",
    "scripts/test_history_raw.py",
    "scripts/test_history_count.py",
    "scripts/test_history_full_output.py",
    "scripts/test_history_data_first.py",
    "scripts/diagnose_zmq_connection.py",
    "scripts/test_mt5_connection.py"
)

foreach ($file in $files) {
    $result = Remove-FileIfExists $file
    if ($result -eq $true) { $deletedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 2: Scripts de Testing Duplicados (6 archivos)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$files = @(
    "scripts/test_backtrader.py",
    "scripts/test_backtrader_simple.py",
    "scripts/test_bt_minimal.py",
    "scripts/test_talib_performance.py",
    "scripts/test_scientific_improvements.py",
    "scripts/start_backtest.py"
)

foreach ($file in $files) {
    $result = Remove-FileIfExists $file
    if ($result -eq $true) { $deletedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 3: Loaders de Datos Antiguos (9 archivos)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$files = @(
    "scripts/download_all_histdata.py",
    "scripts/download_histdata_1min_only.py",
    "scripts/backfill_histdata_parquet.py",
    "scripts/backfill_all_data.py",
    "scripts/setup_hf_token.py",
    "scripts/test_hf_loader.py",
    "scripts/explore_hf_forex_datasets.py",
    "scripts/download_indices.py",
    "scripts/download_liquid_pairs.py"
)

foreach ($file in $files) {
    $result = Remove-FileIfExists $file
    if ($result -eq $true) { $deletedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 4: Scripts de Setup Antiguos (5 archivos)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$files = @(
    "scripts/setup_db_simple.ps1",
    "scripts/setup_indicators_db.ps1",
    "scripts/setup_firewall_rules.ps1",
    "scripts/insert_indicators_to_db.py",
    "scripts/resample_and_calculate.py"
)

foreach ($file in $files) {
    $result = Remove-FileIfExists $file
    if ($result -eq $true) { $deletedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 5: Demos Incompletos (6 archivos)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$files = @(
    "scripts/demo_paper_trading.py",
    "scripts/complete_trading_workflow.py",
    "scripts/integrated_trading_system.py",
    "scripts/integration_test.py",
    "scripts/start_live.py"
)

foreach ($file in $files) {
    $result = Remove-FileIfExists $file
    if ($result -eq $true) { $deletedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 6: Monitoring Antiguo (3 archivos)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$files = @(
    "scripts/instrument_eas_prometheus.py",
    "scripts/generate_test_metrics.py",
    "scripts/start_metrics.ps1"
)

foreach ($file in $files) {
    $result = Remove-FileIfExists $file
    if ($result -eq $true) { $deletedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 7: ML Incompleto (3 archivos)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$files = @(
    "scripts/ml_preprocessing_example.py",
    "scripts/retrain_models.py",
    "scripts/generate_synthetic_data.py"
)

foreach ($file in $files) {
    $result = Remove-FileIfExists $file
    if ($result -eq $true) { $deletedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 8: UI Streamlit (1 archivo)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$files = @(
    "scripts/streamlit_dashboard.py"
)

foreach ($file in $files) {
    $result = Remove-FileIfExists $file
    if ($result -eq $true) { $deletedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 9: Tests Lean (2 archivos)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$files = @(
    "scripts/test_lean_install.py",
    "scripts/test_lean_simple.py"
)

foreach ($file in $files) {
    $result = Remove-FileIfExists $file
    if ($result -eq $true) { $deletedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 10: Módulos Obsoletos (3 archivos)" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

$files = @(
    "underdog/data/hf_loader.py",
    "underdog/data/dukascopy_loader.py",
    "underdog/database/histdata_ingestion.py"
)

foreach ($file in $files) {
    $result = Remove-FileIfExists $file
    if ($result -eq $true) { $deletedCount++ }
    elseif ($result -eq $false) { $failedCount++ }
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 11: Carpeta UI Incompleta" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

if (Test-Path "underdog/ui") {
    try {
        Remove-Item "underdog/ui" -Recurse -Force
        Write-Host "  ✓ Eliminado: underdog/ui/" -ForegroundColor Green
        $deletedCount++
    }
    catch {
        Write-Host "  ✗ ERROR eliminando: underdog/ui/" -ForegroundColor Red
        $failedCount++
    }
}
else {
    Write-Host "  - No existe: underdog/ui/" -ForegroundColor Gray
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📋 FASE 12: Consolidar risk_management → risk" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

if (Test-Path "underdog/risk_management") {
    Write-Host "⚠️  NOTA: Consolidación de risk_management → risk requiere revisión manual." -ForegroundColor Yellow
    Write-Host "   Ambas carpetas contienen archivos. Por favor, revisa y consolida manualmente." -ForegroundColor Yellow
}
else {
    Write-Host "  - No existe: underdog/risk_management/" -ForegroundColor Gray
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📊 RESUMEN DE LIMPIEZA" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

Write-Host "  ✅ Archivos eliminados: $deletedCount" -ForegroundColor Green
if ($failedCount -gt 0) {
    Write-Host "  ❌ Errores: $failedCount" -ForegroundColor Red
}
Write-Host ""

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "✅ PRÓXIMOS PASOS MANUALES" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

Write-Host "1. Actualizar underdog/data/__init__.py:" -ForegroundColor Yellow
Write-Host "   - Eliminar imports de hf_loader" -ForegroundColor White
Write-Host "   - Agregar: from underdog.data.mt5_historical_loader import MT5HistoricalDataLoader`n" -ForegroundColor White

Write-Host "2. Actualizar underdog/backtesting/bt_engine.py:" -ForegroundColor Yellow
Write-Host "   - Eliminar imports de HuggingFaceDataHandler" -ForegroundColor White
Write-Host "   - Eliminar lógica de carga de HuggingFace`n" -ForegroundColor White

Write-Host "3. Revisar y consolidar underdog/risk_management/ → underdog/risk/`n" -ForegroundColor Yellow

Write-Host "4. Opcional: Eliminar carpetas de datos antiguos (HACER BACKUP PRIMERO):" -ForegroundColor Yellow
Write-Host "   - data/raw/" -ForegroundColor White
Write-Host "   - data/historical/" -ForegroundColor White
Write-Host "   - data/parquet/`n" -ForegroundColor White

Write-Host "5. Commit de cambios:" -ForegroundColor Yellow
Write-Host "   git add ." -ForegroundColor White
Write-Host "   git commit -m 'cleanup: Remove 46 obsolete files (pre-TimescaleDB migration)'`n" -ForegroundColor White

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan
Write-Host "✅ Limpieza completada. Ver: docs/AUDIT_CLEANUP_2025_10_22.md`n" -ForegroundColor Green
