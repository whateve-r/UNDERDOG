#!/usr/bin/env bash
# Test de validación de 10 episodios con POEL habilitado

echo "============================================================"
echo "POEL INTEGRATION TEST - 10 Episodes"
echo "============================================================"
echo ""
echo "This test validates POEL integration:"
echo "  - Enriched rewards (PnL + Novelty - Stability)"
echo "  - Capital allocation (Calmar Ratio)"
echo "  - Failure Bank recording"
echo "  - Skill checkpointing"
echo ""
echo "Expected outcomes:"
echo "  ✓ Episode length > 20 steps (vs 2-8 baseline)"
echo "  ✓ DD breach rate < 50% (vs 100% baseline)"
echo "  ✓ Capital allocation not uniform"
echo "  ✓ At least 1 skill checkpointed"
echo ""
echo "Starting test..."
echo ""

poetry run python scripts/train_marl_agent.py \
    --episodes 10 \
    --symbols EURUSD USDJPY XAUUSD GBPUSD \
    --balance 100000 \
    --poel \
    --poel-alpha 0.7 \
    --poel-beta 1.0 \
    --nrf \
    --nrf-cycle 10

echo ""
echo "============================================================"
echo "Test complete! Check logs for:"
echo "  - POEL Meta-Agent coordination messages"
echo "  - Capital allocation changes"
echo "  - Failure Bank/Skill Bank sizes"
echo "============================================================"
