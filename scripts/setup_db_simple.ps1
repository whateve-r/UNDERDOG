# Simple TimescaleDB Setup Script
# Creates hypertables for OHLCV data with technical indicators

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "  UNDERDOG - TimescaleDB Indicators Schema Setup" -ForegroundColor Green
Write-Host "========================================================================`n" -ForegroundColor Cyan

# Configuration
$DB_HOST = "localhost"
$DB_PORT = "5432"
$DB_NAME = "underdog_trading"
$DB_USER = "underdog"
$DB_PASSWORD = "underdog_trading_2024_secure"

$env:PGPASSWORD = $DB_PASSWORD

Write-Host "Database Configuration:" -ForegroundColor Cyan
Write-Host "  Host:     $DB_HOST" -ForegroundColor White
Write-Host "  Port:     $DB_PORT" -ForegroundColor White
Write-Host "  Database: $DB_NAME" -ForegroundColor White
Write-Host "  User:     $DB_USER`n" -ForegroundColor White

# Test connection
Write-Host "Testing database connection..." -ForegroundColor Cyan
$testResult = psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT version();" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Cannot connect to database!" -ForegroundColor Red
    Write-Host "Make sure Docker containers are running: docker-compose up -d`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "OK: Connection successful`n" -ForegroundColor Green

# Execute SQL script
Write-Host "Creating OHLCV + Indicators schema..." -ForegroundColor Cyan
$sqlFile = "docker\init-indicators-db.sql"

if (-not (Test-Path $sqlFile)) {
    Write-Host "`nERROR: SQL file not found: $sqlFile`n" -ForegroundColor Red
    exit 1
}

Write-Host "  Executing: $sqlFile`n" -ForegroundColor White

psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $sqlFile

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================================================" -ForegroundColor Cyan
    Write-Host "  SUCCESS - Database Schema Created!" -ForegroundColor Green
    Write-Host "========================================================================`n" -ForegroundColor Cyan
    
    Write-Host "Created Hypertables:`n" -ForegroundColor Cyan
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT hypertable_name, num_chunks, pg_size_pretty(total_bytes::bigint) AS size FROM timescaledb_information.hypertables WHERE hypertable_schema = 'public' AND hypertable_name LIKE 'ohlcv_%' ORDER BY hypertable_name;"
    
    Write-Host "`nNext Steps:" -ForegroundColor Yellow
    Write-Host "`n1. Download historical data:" -ForegroundColor White
    Write-Host "   poetry run python scripts/download_liquid_pairs.py --start-year 2020 --end-year 2024`n" -ForegroundColor Gray
    Write-Host "2. Insert into database:" -ForegroundColor White
    Write-Host "   poetry run python scripts/insert_indicators_to_db.py --all`n" -ForegroundColor Gray
    Write-Host "========================================================================`n" -ForegroundColor Cyan
} else {
    Write-Host "`nERROR: Failed to create schema`n" -ForegroundColor Red
    exit 1
}

# Cleanup
Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
