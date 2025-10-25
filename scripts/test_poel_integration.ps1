# Test de validación de 10 episodios con POEL habilitado

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "POEL INTEGRATION TEST - 10 Episodes" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This test validates POEL integration:" -ForegroundColor Yellow
Write-Host "  - Enriched rewards (PnL + Novelty - Stability)"
Write-Host "  - Capital allocation (Calmar Ratio)"
Write-Host "  - Failure Bank recording"
Write-Host "  - Skill checkpointing"
Write-Host ""
Write-Host "Expected outcomes:" -ForegroundColor Green
Write-Host "  - Episode length > 20 steps (vs 2-8 baseline)"
Write-Host "  - DD breach rate < 50% (vs 100% baseline)"
Write-Host "  - Capital allocation not uniform"
Write-Host "  - At least 1 skill checkpointed"
Write-Host ""
Write-Host "Starting test..." -ForegroundColor Cyan
Write-Host ""

poetry run python scripts/train_marl_agent.py `
    --episodes 10 `
    --symbols EURUSD USDJPY XAUUSD GBPUSD `
    --balance 100000 `
    --poel `
    --poel-alpha 0.7 `
    --poel-beta 1.0 `
    --nrf `
    --nrf-cycle 10

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Test complete! Check logs for:" -ForegroundColor Green
Write-Host "  - POEL Meta-Agent coordination messages"
Write-Host "  - Capital allocation changes"
Write-Host "  - Failure Bank/Skill Bank sizes"
Write-Host "============================================================" -ForegroundColor Cyan
