# =============================================================================
# Setup TimescaleDB Schema for Technical Indicators
# Creates hypertables with 36+ columns for OHLCV + indicators
# =============================================================================

Write-Host "`n" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  UNDERDOG - TimescaleDB Indicators Schema Setup                    " -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# Load Environment Variables
# =============================================================================

$envFile = Join-Path $PSScriptRoot "..\docker\.env"

if (Test-Path $envFile) {
    Write-Host "✅ Loading environment variables from: $envFile" -ForegroundColor Green
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value)
        }
    }
} else {
    Write-Host "⚠️  .env file not found. Using default values." -ForegroundColor Yellow
    Write-Host "   Expected location: $envFile" -ForegroundColor Yellow
}

# =============================================================================
# Configuration
# =============================================================================

$DB_HOST = if ($env:DB_HOST) { $env:DB_HOST } else { "localhost" }
$DB_PORT = if ($env:DB_PORT) { $env:DB_PORT } else { "5432" }
$DB_NAME = if ($env:DB_NAME) { $env:DB_NAME } else { "underdog_trading" }
$DB_USER = if ($env:DB_USER) { $env:DB_USER } else { "underdog" }
$DB_PASSWORD = if ($env:DB_PASSWORD) { $env:DB_PASSWORD } else { 
    Write-Host ""
    Write-Host "❌ ERROR: DB_PASSWORD not set!" -ForegroundColor Red
    Write-Host "   Please create docker/.env file with:" -ForegroundColor Yellow
    Write-Host "   DB_PASSWORD=your_secure_password" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "📊 Database Configuration:" -ForegroundColor Cyan
Write-Host "   Host:     $DB_HOST" -ForegroundColor White
Write-Host "   Port:     $DB_PORT" -ForegroundColor White
Write-Host "   Database: $DB_NAME" -ForegroundColor White
Write-Host "   User:     $DB_USER" -ForegroundColor White
Write-Host ""

# =============================================================================
# Check Docker Container Status
# =============================================================================

Write-Host "🐳 Checking Docker containers..." -ForegroundColor Cyan

$timescaleContainer = docker ps --filter "name=underdog-timescaledb" --format "{{.Names}}" 2>$null

if ($timescaleContainer -eq "underdog-timescaledb") {
    Write-Host "✅ TimescaleDB container is running" -ForegroundColor Green
} else {
    Write-Host "❌ TimescaleDB container not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Starting Docker containers..." -ForegroundColor Yellow
    
    $dockerComposeFile = Join-Path $PSScriptRoot "..\docker\docker-compose.yml"
    
    if (Test-Path $dockerComposeFile) {
        Set-Location (Join-Path $PSScriptRoot "..\docker")
        docker-compose up -d timescaledb
        
        Write-Host "⏳ Waiting 10 seconds for database to initialize..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
    } else {
        Write-Host "❌ docker-compose.yml not found at: $dockerComposeFile" -ForegroundColor Red
        Write-Host "   Please start TimescaleDB manually or check file location." -ForegroundColor Yellow
        exit 1
    }
}

# =============================================================================
# Test Database Connection
# =============================================================================

Write-Host ""
Write-Host "🔌 Testing database connection..." -ForegroundColor Cyan

$env:PGPASSWORD = $DB_PASSWORD

$testQuery = "SELECT version();"
$testResult = & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c $testQuery 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Database connection successful" -ForegroundColor Green
} else {
    Write-Host "❌ Database connection failed!" -ForegroundColor Red
    Write-Host "   Error: $testResult" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "   1. Verify TimescaleDB is running: docker ps" -ForegroundColor White
    Write-Host "   2. Check credentials in docker/.env" -ForegroundColor White
    Write-Host "   3. Test connection: psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME" -ForegroundColor White
    Write-Host ""
    exit 1
}

# =============================================================================
# Check if Tables Already Exist
# =============================================================================

Write-Host ""
Write-Host "📋 Checking existing tables..." -ForegroundColor Cyan

$checkTablesQuery = "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'ohlcv_%'"

$existingTables = & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c $checkTablesQuery 2>&1
if ($LASTEXITCODE -eq 0) {
    $existingTablesCount = [int]$existingTables.Trim()
} else {
    $existingTablesCount = 0
}

if ($existingTablesCount -gt 0) {
    Write-Host "WARNING: Found $existingTablesCount existing ohlcv_* tables" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Do you want to DROP and recreate them? This will DELETE all data! (yes/no)"
    
    if ($response -ne "yes") {
        Write-Host ""
        Write-Host "ERROR: Setup cancelled by user" -ForegroundColor Red
        Write-Host ""
        exit 0
    }
    
    Write-Host ""
    Write-Host "Dropping existing tables..." -ForegroundColor Yellow
    
    & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS ohlcv_1m CASCADE" 2>&1 | Out-Null
    & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS ohlcv_5m CASCADE" 2>&1 | Out-Null
    & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS ohlcv_15m CASCADE" 2>&1 | Out-Null
    & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS ohlcv_30m CASCADE" 2>&1 | Out-Null
    & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS ohlcv_1H CASCADE" 2>&1 | Out-Null
    & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS ohlcv_4H CASCADE" 2>&1 | Out-Null
    & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "DROP TABLE IF EXISTS ohlcv_1D CASCADE" 2>&1 | Out-Null
    Write-Host "OK: Existing tables dropped" -ForegroundColor Green
}

# =============================================================================
# Execute SQL Script
# =============================================================================

Write-Host ""
Write-Host "🚀 Creating OHLCV + Indicators schema..." -ForegroundColor Cyan

$sqlFile = Join-Path $PSScriptRoot "..\docker\init-indicators-db.sql"

if (-not (Test-Path $sqlFile)) {
    Write-Host "❌ SQL file not found: $sqlFile" -ForegroundColor Red
    exit 1
}

Write-Host "   Executing: $sqlFile" -ForegroundColor White
Write-Host ""

$result = & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $sqlFile 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "OK: SUCCESS - Database Schema Created!" -ForegroundColor Green
    Write-Host ""
    
    # Show table summary
    Write-Host "Created Hypertables:" -ForegroundColor Cyan
    Write-Host ""
    & psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT hypertable_name AS Table, num_chunks AS Chunks, pg_size_pretty(total_bytes::bigint) AS Size FROM timescaledb_information.hypertables WHERE hypertable_schema = 'public' AND hypertable_name LIKE 'ohlcv_%' ORDER BY hypertable_name"
    
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Download historical data with indicators:" -ForegroundColor White
    Write-Host "   poetry run python scripts/download_liquid_pairs.py --start-year 2020 --end-year 2024" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Insert Parquet files into database:" -ForegroundColor White
    Write-Host "   poetry run python scripts/insert_indicators_to_db.py --all" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Verify data:" -ForegroundColor White
    Write-Host "   poetry run python -c `"from underdog.database.db_loader import get_loader; print(get_loader().get_available_symbols())`"" -ForegroundColor Gray
    Write-Host ""
    Write-Host ""
    
} else {
    Write-Host ""
    Write-Host "ERROR: Failed to create schema" -ForegroundColor Red
    Write-Host ""
    exit 1
}

# Cleanup
Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
