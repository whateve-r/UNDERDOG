---
title: "POEL Integration Guide"
date: "2025-10-25"
status: "Complete - Ready for Testing"
---

# POEL Integration with train_marl_agent.py ✅

## Integration Status: COMPLETE

All POEL modules have been integrated into the MARL training pipeline.

## Changes Summary

### 1. Imports Added
```python
from underdog.rl.poel import (
    POELMetaAgent,
    POELRewardShaper,
    DistanceMetric,
    TrainingMode,
)
```

### 2. MARLTrainer.__init__() Modified

**New Parameters**:
- `poel_enabled: bool = True` - Enable/disable POEL

**POEL Components Initialized**:
```python
# POEL Meta-Agent
self.poel_meta_agent = POELMetaAgent(
    initial_balance=100000.0,
    symbols=['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD'],
    max_daily_dd=0.05,  # 5% daily limit
    max_total_dd=0.10,  # 10% total limit
    nrf_enabled=True,
    nrf_cycle_frequency=10,  # NRF every 10 episodes
    state_dim=31,
)

# POEL Reward Shapers (one per agent)
self.poel_shapers = {
    symbol: POELRewardShaper(
        state_dim=31,
        action_dim=1,
        alpha=0.7,  # 70% PnL, 30% exploration
        beta=1.0,   # Stability penalty weight
        novelty_metric=DistanceMetric.L2,
        max_local_dd_pct=0.15,  # 15% local DD limit
    )
    for symbol in symbols
}
```

### 3. Episode Start: POEL Initialization

```python
def train_episode(self):
    # Start POEL episode
    poel_episode_config = self.poel_meta_agent.start_episode()
    
    # Logs:
    # - Episode mode (NORMAL, NRF, CURRICULUM, EMERGENCY)
    # - Curriculum injection status
    # - NRF configuration (if active)
```

### 4. Training Loop: Enriched Rewards

**Critical Section** - Rewards are enriched BEFORE storing in buffer:

```python
# For each agent:
enriched_reward, poel_info = self.poel_shapers[symbol].compute_reward(
    state=state,
    action=action,
    raw_pnl=raw_reward,  # From environment
    new_balance=agent_balance,
)

# Update POEL Meta-Agent performance tracking
self.poel_meta_agent.update_agent_performance(
    agent_id=f"agent_{symbol}",
    symbol=symbol,
    pnl=raw_pnl,
    balance=agent_balance,
)

# NRF step (if in NRF mode)
if self.poel_meta_agent.current_mode == TrainingMode.NRF:
    nrf_metrics = self.poel_meta_agent.nrf_step(state, step)

# Store ENRICHED reward in buffer
agent.replay_buffer.add(..., reward=enriched_reward)
```

**Key Difference**:
- **Before**: `reward = raw_reward` (environment PnL only)
- **After**: `reward = enriched_reward` (α*PnL + Novelty - β*Stability)

### 5. Episode End: Meta-Agent Coordination

```python
# 1. Compute Purpose and Allocate Capital
meta_result = self.poel_meta_agent.compute_purpose_and_allocate(
    current_balance=final_balance,
    daily_dd_pct=daily_dd,
    total_dd_pct=total_dd,
    portfolio_pnl=portfolio_pnl,
)

# 2. Log capital allocation
for symbol, weight in meta_result['weights'].items():
    # weight ∝ ReLU(Calmar_Ratio)
    # Shows which agents are performing well

# 3. Record failure if DD breach
if total_dd > 0.10:
    self.poel_meta_agent.record_failure(...)

# 4. Checkpoint skills (high Calmar Ratio)
for symbol, perf in meta_result['agent_performances'].items():
    if perf['calmar_ratio'] > 2.0:
        self.poel_meta_agent.checkpoint_skill(...)

# 5. End episode
episode_summary = self.poel_meta_agent.end_episode()
```

### 6. Main Function: Command-Line Arguments

**New Arguments**:
```bash
--poel              # Enable POEL
--poel-alpha 0.7    # PnL weight (0.7 = 70% PnL, 30% exploration)
--poel-beta 1.0     # Stability penalty weight
--nrf               # Enable Neural Reward Functions
--nrf-cycle 10      # NRF every 10 episodes
```

**Example Usage**:
```bash
# Test with POEL enabled
poetry run python scripts/train_marl_agent.py \
    --episodes 10 \
    --symbols EURUSD USDJPY XAUUSD GBPUSD \
    --balance 100000 \
    --poel \
    --poel-alpha 0.7 \
    --poel-beta 1.0 \
    --nrf \
    --nrf-cycle 10

# Baseline (no POEL)
poetry run python scripts/train_marl_agent.py \
    --episodes 10 \
    --symbols EURUSD USDJPY XAUUSD GBPUSD \
    --balance 100000
```

## Validation Checklist

### ✅ Code Integration
- [x] POEL imports added
- [x] POELMetaAgent initialized
- [x] POELRewardShaper created per agent
- [x] Enriched rewards used in training
- [x] Meta-Agent coordination at episode end
- [x] Command-line arguments for POEL configuration

### 🔄 Testing (To Do)
- [ ] Run 10-episode test (`test_poel_integration.ps1`)
- [ ] Verify enriched rewards in logs
- [ ] Check capital allocation changes
- [ ] Confirm Failure Bank records DD breaches
- [ ] Validate Skill Bank checkpoints high Calmar agents

### 📊 Expected Log Output

**Episode Start**:
```
================================================================================
POEL Episode 1
================================================================================
  Mode: normal
  Curriculum Injection: False
================================================================================
```

**Training Step** (with POEL):
```
# No visible changes - enriched rewards computed internally
```

**Episode End**:
```
================================================================================
POEL META-AGENT COORDINATION
================================================================================
  Purpose Score: 487.50
  Global DD: 2.30% (Daily: 0.00%)
  Emergency Mode: False

  Capital Allocation:
    EURUSD  : 28.45% (Calmar:  1.23)
    USDJPY  : 22.10% (Calmar:  0.87)
    XAUUSD  : 32.50% (Calmar:  1.54)
    GBPUSD  : 16.95% (Calmar:  0.65)

  Episode Summary:
    Failure Bank Size: 0
    Skill Bank Size: 0
================================================================================
```

**Episode 10** (NRF Mode):
```
================================================================================
POEL Episode 10
================================================================================
  Mode: nrf
  Curriculum Injection: False
  NRF Mode Active: Cycle 1: Generating new skill with R_ψ rewards
================================================================================
```

**DD Breach** (Failure Recording):
```
================================================================================
POEL META-AGENT COORDINATION
================================================================================
  Purpose Score: -1520.34
  Global DD: 12.50% (Daily: 5.20%)
  Emergency Mode: True

  ⚠ DD BREACH: 12.50% > 10.0%
  Recording failure to Failure Bank...
  ✓ Failure recorded (breach size: 2.50%)

  Episode Summary:
    Failure Bank Size: 1
    Skill Bank Size: 0
================================================================================
```

**High Calmar** (Skill Checkpointing):
```
================================================================================
POEL META-AGENT COORDINATION
================================================================================
  Capital Allocation:
    XAUUSD  : 45.20% (Calmar:  3.14)  <-- High Calmar!

  ✓ High Calmar detected: XAUUSD = 3.14
  ✓ Skill checkpointed (Calmar=3.14, Novelty=0.782)

  Episode Summary:
    Failure Bank Size: 1
    Skill Bank Size: 1
================================================================================
```

## Performance Monitoring

### Key Metrics to Track

**Episode Length**:
- Baseline: 2-8 steps
- POEL Target: >20 steps
- Indicates: Better risk management (fewer DD breaches)

**DD Breach Rate**:
- Baseline: 100% (all episodes terminate on DD)
- POEL Target: <50%
- Indicates: Stability penalty working

**Capital Allocation**:
- Baseline: N/A (uniform 25% per agent)
- POEL: Dynamic based on Calmar Ratio
- Indicates: Meta-Agent learning to allocate capital

**Skill Bank Size**:
- Baseline: 0 (no skill discovery)
- POEL Target: >1 skill after 10 episodes
- Indicates: High-performing strategies discovered

**Failure Bank Size**:
- Baseline: N/A (failures not tracked)
- POEL: Grows with DD breaches
- Indicates: Curriculum learning active

### Logs to Check

**Training Metrics CSV** (`logs/training_metrics.csv`):
- Compare episode lengths (POEL vs baseline)
- Check DD breach frequency
- Analyze reward progression

**Console Logs**:
- POEL coordination messages
- Capital allocation changes per episode
- Skill checkpointing events
- Failure recording events

## Troubleshooting

### Issue: Enriched rewards always equal raw rewards
**Symptom**: `novelty_bonus = 0.0` in all steps
**Cause**: NoveltyDetector buffer not filling
**Fix**: Check that `poel_shaper.compute_reward()` is being called
**Validation**: Log `poel_info` dict to verify components

### Issue: Capital allocation stays uniform
**Symptom**: All agents get 25% weight every episode
**Cause**: Calmar Ratio calculation failing (insufficient history)
**Fix**: Wait at least 50 steps per agent for reliable Calmar
**Validation**: Check `meta_result['agent_performances']` for Calmar values

### Issue: No skills checkpointed
**Symptom**: `skill_bank_size = 0` after 10 episodes
**Cause**: Calmar Ratio threshold (2.0) too high
**Fix**: Lower threshold in `train_marl_agent.py` (line ~715)
**Validation**: Check agent Calmar Ratios in logs

### Issue: All episodes in NRF mode
**Symptom**: Every episode shows `Mode: nrf`
**Cause**: `nrf_cycle_frequency` misconfigured
**Fix**: Set `--nrf-cycle 10` (not `--nrf-cycle 1`)
**Validation**: Should see NRF only on episodes 10, 20, 30, etc.

## Next Steps

1. **Run 10-Episode Test**:
   ```powershell
   .\scripts\test_poel_integration.ps1
   ```

2. **Analyze Results**:
   - Episode length vs baseline
   - DD breach rate
   - Capital allocation dynamics
   - Skill/Failure bank sizes

3. **Tune Hyperparameters** (if needed):
   - `--poel-alpha`: Increase for more exploitation (PnL focus)
   - `--poel-beta`: Increase for more risk aversion (stability)
   - Calmar threshold: Lower for more skill checkpoints

4. **50-Episode Validation Run**:
   ```powershell
   poetry run python scripts/train_marl_agent.py --episodes 50 --poel
   ```

5. **Baseline Comparison**:
   ```powershell
   # Run without POEL
   poetry run python scripts/train_marl_agent.py --episodes 50
   
   # Compare:
   # - Episode length (steps)
   # - DD breach rate (%)
   # - Final Calmar Ratio
   ```

## Files Modified

- `scripts/train_marl_agent.py` (~150 lines added)
  - POEL imports
  - MARLTrainer.__init__() POEL initialization
  - train_episode() enriched rewards
  - train_episode() meta-agent coordination
  - main() command-line arguments

## Files Created

- `scripts/test_poel_integration.ps1` (PowerShell test script)
- `scripts/test_poel_integration.sh` (Bash test script)
- `docs/POEL_INTEGRATION_GUIDE.md` (this file)

---

**Integration Complete!** Ready for testing. 🚀
