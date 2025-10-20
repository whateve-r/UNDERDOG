# Script para mantener el generador de métricas corriendo
Write-Host "Iniciando generador de métricas..." -ForegroundColor Green
Set-Location "C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG"

$process = Start-Process -FilePath "poetry" -ArgumentList "run", "python", "scripts/generate_test_metrics.py" -NoNewWindow -PassThru
Write-Host "✅ Generador iniciado (PID: $($process.Id))" -ForegroundColor Green
Write-Host "📊 Métricas disponibles en: http://localhost:8000/metrics" -ForegroundColor Cyan
Write-Host "" 
Write-Host "Para detener presiona Ctrl+C" -ForegroundColor Yellow

# Mantener el script corriendo
try {
    $process.WaitForExit()
} catch {
    Write-Host "Deteniendo generador..." -ForegroundColor Yellow
    $process.Kill()
}
