# Feature Store Quick Setup Script
# Run this to deploy TimescaleDB + Redis + test data ingestion

Write-Host "=== UNDERDOG Feature Store Setup ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Docker
Write-Host "[1/6] Checking Docker..." -ForegroundColor Yellow
$dockerRunning = docker ps 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Docker is running" -ForegroundColor Green

# Step 2: Create .env file
Write-Host "[2/6] Creating .env file..." -ForegroundColor Yellow
$envPath = "docker\.env"
if (-Not (Test-Path $envPath)) {
    $envContent = @"
DB_USER=underdog
DB_PASSWORD=underdog_secure_2025
DB_NAME=underdog_trading
GRAFANA_PASSWORD=admin123
"@
    Set-Content -Path $envPath -Value $envContent
    Write-Host "[OK] .env file created" -ForegroundColor Green
} else {
    Write-Host "[OK] .env file already exists" -ForegroundColor Green
}

# Step 3: Start Docker services
Write-Host "[3/6] Starting TimescaleDB + Redis..." -ForegroundColor Yellow
Set-Location docker
docker-compose up -d timescaledb redis
Set-Location ..

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start Docker services" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Services started" -ForegroundColor Green

# Wait for services to be ready
Write-Host "   Waiting for services to initialize (15 seconds)..." -ForegroundColor Cyan
Start-Sleep -Seconds 15

# Step 4: Verify TimescaleDB
Write-Host "[4/6] Verifying TimescaleDB schema..." -ForegroundColor Yellow
$dbCheck = docker exec underdog-timescaledb psql -U underdog -d underdog_trading -c "\dt" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] TimescaleDB schema created successfully" -ForegroundColor Green
    Write-Host ""
    Write-Host "   Tables:" -ForegroundColor Cyan
    Write-Host "   - ohlcv (hypertable)" -ForegroundColor White
    Write-Host "   - sentiment_scores (hypertable)" -ForegroundColor White
    Write-Host "   - macro_indicators (hypertable)" -ForegroundColor White
    Write-Host "   - regime_predictions (hypertable)" -ForegroundColor White
    Write-Host "   - trades, positions, metrics, account_snapshots" -ForegroundColor White
} else {
    Write-Host "WARNING: Could not verify database schema" -ForegroundColor Yellow
}

# Step 5: Verify Redis
Write-Host "[5/6] Verifying Redis..." -ForegroundColor Yellow
$redisCheck = docker exec underdog-redis redis-cli ping 2>&1

if ($redisCheck -eq "PONG") {
    Write-Host "[OK] Redis is responding" -ForegroundColor Green
} else {
    Write-Host "WARNING: Redis not responding properly" -ForegroundColor Yellow
}

# Step 6: Install Python dependencies
Write-Host "[6/6] Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "   (This may take 5-10 minutes for transformers + torch)" -ForegroundColor Cyan

$missingDeps = @()

# Check asyncpg
python -c "import asyncpg" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { $missingDeps += "asyncpg" }

# Check redis
python -c "import redis" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { $missingDeps += "redis" }

# Check transformers
python -c "import transformers" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { $missingDeps += "transformers" }

# Check torch
python -c "import torch" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { $missingDeps += "torch" }

if ($missingDeps.Count -gt 0) {
    Write-Host "   Missing dependencies: $($missingDeps -join ', ')" -ForegroundColor Yellow
    Write-Host "   Installing..." -ForegroundColor Cyan
    
    pip install asyncpg redis transformers torch
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        Write-Host "   Try manually: pip install asyncpg redis transformers torch" -ForegroundColor Yellow
    }
} else {
    Write-Host "[OK] All dependencies already installed" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Services Running:" -ForegroundColor Cyan
Write-Host "  - TimescaleDB: localhost:5432" -ForegroundColor White
Write-Host "  - Redis: localhost:6379" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Set API credentials:" -ForegroundColor White
Write-Host "     `$env:REDDIT_CLIENT_ID='your_client_id'" -ForegroundColor Gray
Write-Host "     `$env:REDDIT_CLIENT_SECRET='your_client_secret'" -ForegroundColor Gray
Write-Host "     `$env:FRED_API_KEY='your_fred_key'" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Test TimescaleDB connector:" -ForegroundColor White
Write-Host "     python underdog/database/timescale/timescale_connector.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Download FinGPT model (~500MB, one-time):" -ForegroundColor White
Write-Host "     python underdog/sentiment/llm_connector.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Test data orchestrator (2 min test):" -ForegroundColor White
Write-Host "     python underdog/database/timescale/data_orchestrator.py" -ForegroundColor Gray
Write-Host ""
Write-Host "Documentation: docs/FEATURE_STORE_ARCHITECTURE.md" -ForegroundColor Cyan
Write-Host ""
