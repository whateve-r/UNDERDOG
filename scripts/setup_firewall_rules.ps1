# ==============================================================================
# UNDERDOG - Firewall Rules Setup Script
# ==============================================================================
# 
# Este script crea reglas de firewall para permitir tráfico a los servicios
# de UNDERDOG Trading System:
#   - Port 8000: Metrics Generator (Python Prometheus exporter)
#   - Port 8501: Streamlit Dashboard
#   - Port 9090: Prometheus Server (Docker)
#   - Port 3000: Grafana Dashboard (Docker)
#
# REQUISITO: Ejecutar como Administrador
# USO: 
#   1. Click derecho en PowerShell → "Ejecutar como administrador"
#   2. cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
#   3. .\scripts\setup_firewall_rules.ps1
#
# ==============================================================================

# Verificar que se ejecuta como Administrador
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "❌ ERROR: Este script requiere privilegios de Administrador" -ForegroundColor Red
    Write-Host "Por favor, ejecuta PowerShell como Administrador y vuelve a intentarlo" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Pasos:" -ForegroundColor Cyan
    Write-Host "  1. Busca 'PowerShell' en el menú Inicio" -ForegroundColor White
    Write-Host "  2. Click derecho → 'Ejecutar como administrador'" -ForegroundColor White
    Write-Host "  3. cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG" -ForegroundColor White
    Write-Host "  4. .\scripts\setup_firewall_rules.ps1" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  UNDERDOG Firewall Rules Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Definir reglas a crear
$rules = @(
    @{
        Name = "UNDERDOG Metrics Server (TCP-In)"
        Port = 8000
        Description = "Allow inbound connections to UNDERDOG metrics server (Python Prometheus exporter)"
    },
    @{
        Name = "UNDERDOG Streamlit UI (TCP-In)"
        Port = 8501
        Description = "Allow inbound connections to UNDERDOG Streamlit backtesting dashboard"
    },
    @{
        Name = "UNDERDOG Prometheus (TCP-In)"
        Port = 9090
        Description = "Allow inbound connections to Prometheus monitoring server (Docker)"
    },
    @{
        Name = "UNDERDOG Grafana (TCP-In)"
        Port = 3000
        Description = "Allow inbound connections to Grafana visualization dashboard (Docker)"
    }
)

$successCount = 0
$skipCount = 0
$failCount = 0

foreach ($rule in $rules) {
    Write-Host "Procesando puerto $($rule.Port)..." -NoNewline
    
    # Verificar si la regla ya existe
    $existingRule = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue
    
    if ($existingRule) {
        Write-Host " ⚠️  YA EXISTE (omitiendo)" -ForegroundColor Yellow
        $skipCount++
        continue
    }
    
    try {
        # Crear regla de firewall
        New-NetFirewallRule `
            -DisplayName $rule.Name `
            -Direction Inbound `
            -Protocol TCP `
            -LocalPort $rule.Port `
            -Action Allow `
            -Profile Private,Domain `
            -Description $rule.Description `
            -ErrorAction Stop | Out-Null
        
        Write-Host " ✅ CREADA" -ForegroundColor Green
        $successCount++
    }
    catch {
        Write-Host " ❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
        $failCount++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Resumen" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Reglas creadas:  $successCount" -ForegroundColor Green
Write-Host "Reglas omitidas: $skipCount" -ForegroundColor Yellow
Write-Host "Errores:         $failCount" -ForegroundColor Red
Write-Host ""

# Mostrar reglas creadas
if ($successCount -gt 0 -or $skipCount -gt 0) {
    Write-Host "Reglas de firewall activas (UNDERDOG):" -ForegroundColor Cyan
    Write-Host ""
    
    Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*UNDERDOG*"} | 
        Select-Object @{Name="Nombre";Expression={$_.DisplayName}}, 
                      @{Name="Habilitada";Expression={$_.Enabled}}, 
                      @{Name="Dirección";Expression={$_.Direction}}, 
                      @{Name="Acción";Expression={$_.Action}} | 
        Format-Table -AutoSize
}

# Verificar puertos abiertos
Write-Host ""
Write-Host "Verificando puertos en uso:" -ForegroundColor Cyan
Write-Host ""

$ports = @(8000, 8501, 9090, 3000)
foreach ($port in $ports) {
    $listening = netstat -ano | Select-String ":$port " | Select-String "LISTENING"
    
    if ($listening) {
        Write-Host "  Puerto $port : ✅ LISTENING" -ForegroundColor Green
    } else {
        Write-Host "  Puerto $port : ⏳ NO ACTIVO (iniciar servicio)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Próximos Pasos" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Verificar Docker containers:" -ForegroundColor White
Write-Host "   docker ps" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Iniciar Metrics Generator (PowerShell separado):" -ForegroundColor White
Write-Host "   cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG" -ForegroundColor Gray
Write-Host "   poetry run python scripts\generate_test_metrics.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Verificar métricas accesibles:" -ForegroundColor White
Write-Host "   curl http://localhost:8000/metrics" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Verificar Prometheus target:" -ForegroundColor White
Write-Host "   http://localhost:9090/targets" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Abrir Grafana:" -ForegroundColor White
Write-Host "   http://localhost:3000 (admin/admin123)" -ForegroundColor Gray
Write-Host ""
Write-Host "✅ Setup completado!" -ForegroundColor Green
Write-Host ""
