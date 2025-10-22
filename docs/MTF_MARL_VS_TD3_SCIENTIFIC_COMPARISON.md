# 📊 MTF-MARL vs TD3: Scientific Comparison & Expected Performance

**Purpose**: Quantitative comparison between Single-Agent TD3 and Multi-Agent MTF-MARL architectures  
**Basis**: Theoretical predictions from Albrecht et al. (NeurIPS 2018) and empirical MARL literature  
**Decision Criteria**: Data-driven go/no-go for MTF-MARL implementation

---

## 1. ARCHITECTURAL COMPARISON

| Dimension | TD3 (Current) | MTF-MARL (Proposed) |
|-----------|---------------|---------------------|
| **Agents** | 1 (single policy) | 2 (H1 + M1, hierarchical) |
| **Timeframes** | M1 only | H4 (strategic) + M1 (tactical) |
| **Observation Space** | 24D unified | H1: 22D (macro), M1: 23D (execution + H1) |
| **Action Space** | 2D continuous | H1: 2D (trend_bias, max_limit), M1: 2D (entry_size, exit_conf) |
| **Coordination** | N/A | Explicit (M1 observes H1 limits) |
| **Replay Buffer** | Single (1M transitions) | Shared (1M joint transitions) |
| **Critics** | Twin Critics (Q1, Q2) | Joint Critic (Q_joint) |
| **Training Complexity** | O(1) agent | O(2) agents, but shared buffer |
| **DD Control** | Reactive (penalty after breach) | Proactive (H1 prevents before) |

---

## 2. EXPECTED PERFORMANCE METRICS

### A. Learning Efficiency

**Hypothesis**: MTF-MARL converges faster due to Shared Experience (ASEAC)

| Metric | TD3 | MTF-MARL | Improvement |
|--------|-----|----------|-------------|
| **Episodes to Converge** | ~800-1200 | ~500-800 | **30-40% faster** |
| **Sample Efficiency** | 1x (baseline) | 1.3-1.5x | **30-50% better** |
| **Exploration Redundancy** | High (single policy) | Low (H1 macro, M1 micro) | **Reduced 40%** |

**Explanation**: Cuando H1 toma una decisión macro (ej. "reduce riesgo por VIX alto"), M1 aprende inmediatamente de esa experiencia sin tener que descubrirla independientemente. Esto elimina la redundancia de exploración.

**Validation**:
```python
# Measure: Episodes to reach Sharpe > 1.0
td3_convergence_episodes = []  # Expected: 800-1200
mtf_marl_convergence_episodes = []  # Expected: 500-800

# If mean(mtf_marl) < 0.7 * mean(td3) → MTF-MARL is significantly faster
```

### B. Risk Control (DD Management)

**Hypothesis**: MTF-MARL reduces DD violations via proactive H1 risk manager

| Metric | TD3 (Reactive) | MTF-MARL (Proactive) | Improvement |
|--------|----------------|----------------------|-------------|
| **DD Violations (% episodes)** | 10-25% | **< 5%** | **50-80% reduction** |
| **Max Drawdown (avg)** | 6-8% | **4-6%** | **25-33% reduction** |
| **Time in DD > 3%** | 15-20% | **< 10%** | **33-50% reduction** |
| **Recovery Time** | 40-60 episodes | **20-40 episodes** | **33-50% faster** |

**Explanation**: H1 monitorea el DD acumulado en timeframe H4 y reduce `max_position_limit` **antes** de que M1 pueda tomar posiciones excesivas. TD3 solo reacciona cuando el DD ya ocurrió (penalty -1000/-10000).

**Validation**:
```python
# Measure: % episodes with DD > 5% (daily) or DD > 10% (total)
td3_violation_rate = len([e for e in episodes if e.max_dd > 0.05]) / len(episodes)
mtf_marl_violation_rate = len([e for e in episodes if e.max_dd > 0.05]) / len(episodes)

# Expected: td3_violation_rate > 0.10, mtf_marl_violation_rate < 0.05
# If mtf_marl_violation_rate < 0.5 * td3_violation_rate → MTF-MARL significantly safer
```

### C. Trading Performance

**Hypothesis**: MTF-MARL achieves higher Sharpe via better trend alignment (H1) + execution optimization (M1)

| Metric | TD3 | MTF-MARL | Improvement |
|--------|-----|----------|-------------|
| **Sharpe Ratio** | 1.0-1.3 | **1.3-1.6** | **15-30% higher** |
| **Win Rate** | 52-55% | **55-58%** | **3-5% higher** |
| **Profit Factor** | 1.3-1.5 | **1.5-1.8** | **15-20% higher** |
| **Avg Trade Duration** | 12-18 bars (M1) | **H1: 240 bars, M1: 15 bars** | Better trend capture |
| **Slippage Cost** | Higher (aggressive entries) | **Lower (M1 optimizes timing)** | **10-20% reduction** |

**Explanation**: 
- **H1** identifica trends en H4, permitiendo a M1 entrar en la dirección correcta
- **M1** optimiza el timing de entrada (espera pullbacks, no entra en extremos)
- **Joint Optimization**: H1 captura tendencias largas, M1 minimiza costos

**Validation**:
```python
# Measure: Sharpe Ratio = mean(returns) / std(returns) * sqrt(252)
td3_sharpe = calculate_sharpe(td3_returns)
mtf_marl_sharpe = calculate_sharpe(mtf_marl_returns)

# Expected: mtf_marl_sharpe > 1.15 * td3_sharpe
# If mtf_marl_sharpe > 1.3 AND td3_sharpe < 1.2 → MTF-MARL is superior
```

### D. Robustness to Market Regimes

**Hypothesis**: MTF-MARL adapts faster to regime changes (trend → range) due to H1 macro awareness

| Metric | TD3 | MTF-MARL | Improvement |
|--------|-----|----------|-------------|
| **Regime Transition Loss** | -3% to -5% equity | **-1% to -2%** | **50-60% reduction** |
| **Adaptation Time** | 80-120 episodes | **40-60 episodes** | **50% faster** |
| **Performance in Range** | Sharpe 0.5-0.8 | **Sharpe 0.8-1.1** | **30-40% better** |
| **Performance in Trend** | Sharpe 1.2-1.5 | **Sharpe 1.5-1.8** | **20-25% better** |

**Explanation**: H1 observa `regime_trend`, `regime_range`, y macro indicators (VIX, yield curve). Cuando detecta cambio de trend a range, reduce `max_position_limit` para evitar whipsaws. M1 respeta ese límite y reduce tamaño de trades.

**Validation**:
```python
# Measure: Performance drop during regime transition
regime_transitions = identify_transitions(regime_predictions)  # trend → range
td3_transition_loss = []
mtf_marl_transition_loss = []

for transition in regime_transitions:
    td3_loss = td3_equity[transition] - td3_equity[transition - 20]
    mtf_marl_loss = mtf_marl_equity[transition] - mtf_marl_equity[transition - 20]
    td3_transition_loss.append(td3_loss)
    mtf_marl_transition_loss.append(mtf_marl_loss)

# Expected: mean(mtf_marl_transition_loss) > mean(td3_transition_loss)
# i.e., MTF-MARL loses less during transitions
```

---

## 3. ABLATION STUDY DESIGN

Para validar científicamente las mejoras, se debe ejecutar un **Ablation Study** con 4 variantes:

### Variants

| Variant | Description | Purpose |
|---------|-------------|---------|
| **V1: TD3 Baseline** | Current implementation (single agent, M1 only) | Baseline performance |
| **V2: MTF No Coordination** | H1 + M1 agents, but M1 **doesn't** observe H1 actions | Test coordination impact |
| **V3: MTF Separate Buffers** | H1 + M1 agents, separate replay buffers | Test shared experience impact |
| **V4: MTF-MARL Full (ASEAC)** | H1 + M1 agents, shared buffer + joint critic | Full proposed architecture |

### Expected Results

```
Performance Ranking (Sharpe Ratio):
V4 (MTF-MARL) > V3 (Separate Buffers) > V2 (No Coordination) > V1 (TD3 Baseline)

Risk Control Ranking (DD Violations):
V4 (MTF-MARL) < V2 (No Coordination) < V3 (Separate Buffers) < V1 (TD3 Baseline)

Learning Speed Ranking (Episodes to Converge):
V4 (MTF-MARL) < V3 (Separate Buffers) < V1 (TD3 Baseline) < V2 (No Coordination)
```

**Key Insights**:
- If **V4 > V3**: Shared experience accelerates learning (validates ASEAC)
- If **V4 > V2**: Coordination reduces DD violations (validates hierarchical control)
- If **V4 > V1**: MTF-MARL is superior to single-agent TD3 (validates entire architecture)

### Ablation Study Execution

```python
# Run all 4 variants with same random seeds
results = {}
for variant in ['V1_TD3', 'V2_MTF_NoCoord', 'V3_MTF_SeparateBuffers', 'V4_MTF_MARL']:
    for seed in [42, 123, 456, 789, 1337]:  # 5 seeds for statistical significance
        agent = create_agent(variant, seed)
        metrics = train(agent, episodes=1000, eval_freq=50)
        results[variant][seed] = metrics

# Statistical testing
import scipy.stats as stats

# Compare V4 vs V1 (MTF-MARL vs TD3)
v4_sharpe = [results['V4_MTF_MARL'][s]['final_sharpe'] for s in seeds]
v1_sharpe = [results['V1_TD3'][s]['final_sharpe'] for s in seeds]

t_stat, p_value = stats.ttest_ind(v4_sharpe, v1_sharpe)
print(f"MTF-MARL vs TD3 Sharpe: t={t_stat:.3f}, p={p_value:.4f}")
# Expected: p < 0.05 → statistically significant improvement
```

---

## 4. COMPUTATIONAL COST ANALYSIS

**Critical Question**: ¿El MTF-MARL vale el costo computacional adicional?

### Training Cost

| Aspect | TD3 | MTF-MARL | Overhead |
|--------|-----|----------|----------|
| **Forward Passes per Step** | 2 (actor + critic) | 4 (H1_actor + M1_actor + joint_critic + target) | **2x** |
| **Backward Passes per Step** | 2 | 3 (H1, M1, joint_critic) | **1.5x** |
| **Memory (Replay Buffer)** | 1M transitions × 30D | 1M transitions × 48D (22+23+2+2+1) | **1.6x** |
| **Training Time per Episode** | ~60s (baseline) | ~90-120s | **1.5-2x slower** |
| **Total Training Time (1000 ep)** | ~16 hours | **24-32 hours** | **1.5-2x longer** |

**Verdict**: 
- If MTF-MARL converges in **500-800 episodes** vs TD3's **800-1200 episodes**:
  - MTF-MARL: 500 ep × 90s = **12.5 hours**
  - TD3: 800 ep × 60s = **13.3 hours**
  - **Result**: MTF-MARL is actually **faster** to converge despite higher cost per episode

### Inference Cost (Production)

| Aspect | TD3 | MTF-MARL | Overhead |
|--------|-----|----------|----------|
| **Forward Passes per Trade** | 1 (actor) | 2 (H1 + M1 actors) | **2x** |
| **Latency** | ~2-5ms | **~4-8ms** | **2x** |
| **CPU Usage** | Low (single model) | Medium (2 models) | **+50%** |

**Verdict**: 
- Latency increase (4-8ms) is **negligible** for forex trading (execution latency is 50-200ms)
- CPU overhead is acceptable for prop firm server (not high-frequency trading)

---

## 5. DECISION CRITERIA: When to Implement MTF-MARL

### 🚨 IMPLEMENT IMMEDIATELY if Quick Test shows:

1. **High DD Violation Rate** (>20% of episodes):
   - TD3 no logra controlar drawdown con CMDP penalties
   - Evidencia: Episodios terminan frecuentemente con reward = -1000 o -10000
   - **Action**: H1 risk manager necesario para prevención proactiva

2. **Miopic Trading Behavior**:
   - TD3 toma trades rentables a corto plazo (M1) pero destructivos a largo plazo
   - Evidencia: Alta frecuencia de trades (>30 per day), bajo Sharpe (<0.8)
   - **Action**: H1 trend alignment necesario para dirección estratégica

3. **Poor Regime Adaptation**:
   - Performance cae >10% durante transiciones de régimen
   - Evidencia: Equity curve con drawdowns pronunciados en cambios trend→range
   - **Action**: H1 macro awareness necesario para adaptación rápida

4. **Excessive Action Volatility**:
   - Agente cambia posiciones bruscamente sin consistencia
   - Evidencia: std(position_sizes) > 0.5, frecuentes reversals
   - **Action**: H1 smoothing de decisiones estratégicas

### ⏸️ POSTPONE MTF-MARL if Quick Test shows:

1. **Low DD Violation Rate** (<5% of episodes):
   - TD3 controla riesgo efectivamente con CMDP actual
   - **Action**: Proceder a full training (2000 episodes), validar en backtest

2. **Good Learning Curve**:
   - Sharpe aumenta consistentemente, DD disminuye over episodes
   - **Action**: TD3 funciona, MTF-MARL es optimización futura

3. **Acceptable Performance** (Sharpe >1.2, DD <8%):
   - TD3 cumple targets de prop firm
   - **Action**: Priorizar deployment, MTF-MARL como Phase 2

### 🛑 REJECT MTF-MARL if:

1. **Fundamental Data Issues**:
   - Features con alta correlación (multicolinealidad)
   - Regime predictions con baja confianza (<0.6)
   - **Action**: Arreglar data pipeline primero

2. **CMDP Config Issues**:
   - Penalties demasiado agresivos (reward promedio < -500)
   - Limits demasiado restrictivos (episodes terminan en <100 steps)
   - **Action**: Ajustar CMDP hyperparams antes de agregar complejidad

---

## 6. VALIDATION PROTOCOL

### Phase 1: Quick Test (100 episodes)

**Purpose**: Gather data for go/no-go decision

**Metrics to Track**:
```python
quick_test_metrics = {
    'dd_violation_rate': 0.0,  # % episodes with DD > 5% or 10%
    'final_sharpe': 0.0,
    'convergence_episode': 0,  # First episode with Sharpe > 1.0
    'avg_episode_reward': 0.0,
    'action_volatility': 0.0,  # std(position_sizes)
    'regime_adaptation_loss': 0.0,  # Avg loss during transitions
    'max_drawdown': 0.0,
    'time_in_dd_3pct': 0.0  # % time with DD > 3%
}
```

**Decision Tree**:
```
if dd_violation_rate > 0.20:
    decision = "IMPLEMENT MTF-MARL (High DD violations)"
elif avg_episode_reward < -100:
    decision = "FIX CMDP CONFIG (Penalties too harsh)"
elif final_sharpe < 0.8:
    decision = "IMPLEMENT MTF-MARL (Poor performance)"
elif action_volatility > 0.5:
    decision = "IMPLEMENT MTF-MARL (Unstable actions)"
elif final_sharpe > 1.2 and dd_violation_rate < 0.05:
    decision = "PROCEED TO FULL TRAINING (TD3 sufficient)"
else:
    decision = "RUN LONGER TEST (200 episodes for more data)"
```

### Phase 2: Ablation Study (if MTF-MARL approved)

**Variants**: V1 (TD3), V2 (MTF no coord), V3 (MTF separate buffers), V4 (MTF-MARL full)

**Runs**: 5 seeds × 4 variants = 20 training runs

**Duration**: 1000 episodes per run × 90s/ep = **25 hours per run** → **500 hours total** (~20 days on single GPU)

**Parallelization**: 4 GPUs → **5 days**

**Analysis**:
```python
# Statistical significance testing
from scipy.stats import f_oneway

# ANOVA: Are there significant differences between variants?
f_stat, p_value = f_oneway(
    v1_sharpes, v2_sharpes, v3_sharpes, v4_sharpes
)
print(f"ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
# Expected: p < 0.001 → variants are significantly different

# Post-hoc: Which variant is best?
import scikit_posthocs as sp
posthoc = sp.posthoc_tukey([v1_sharpes, v2_sharpes, v3_sharpes, v4_sharpes])
print(posthoc)
# Expected: V4 > V3 > V2 > V1 with p < 0.05
```

### Phase 3: Production Validation (if MTF-MARL selected)

**Test Set**: Last 3 months (Sept-Oct 2024) - held-out

**Metrics**:
- Out-of-sample Sharpe > 1.0
- Max DD < 7%
- Zero CMDP violations
- PropFirmSafetyShield compliance (100%)

**Stress Tests**:
1. **FRED Macro Shock**: Simulación de subida de tasas +2% súbita
2. **Drawdown Approach**: Portfolio al 4.5% DD → H1 debe reducir límite
3. **Regime Change**: Trend fuerte → range choppy → H1 debe adaptar
4. **High Volatility**: VIX spike a 40+ → H1 debe reducir exposición

**Success Criteria**: Todas las stress tests passed, 0 catastrophic failures

---

## 7. EXPECTED RESEARCH CONTRIBUTIONS

Si MTF-MARL demuestra superioridad sobre TD3, el proyecto puede contribuir a la literatura:

### Paper Outline: "Hierarchical Multi-Agent RL for Prop Firm Trading"

**Abstract**:
We propose MTF-MARL, a multi-timeframe multi-agent reinforcement learning architecture for forex trading with catastrophic drawdown constraints. Our system uses a hierarchical design: a high-level agent (H1) manages risk and trend direction on H4 timeframe, while a low-level agent (M1) optimizes execution on M1 timeframe. Using Shared Experience Actor-Critic (ASEAC), we achieve 30-40% faster convergence and 50-80% reduction in drawdown violations compared to single-agent TD3. We validate on 3 years of forex data (4 pairs) and demonstrate deployment to prop firm demo account with zero CMDP violations.

**Contributions**:
1. **Novel Architecture**: First application of hierarchical MARL to forex with explicit CMDP safety
2. **Shared Experience**: Empirical validation that ASEAC accelerates learning in financial domain
3. **Ablation Study**: Quantifies contribution of coordination vs shared buffer vs hierarchy
4. **Production Deployment**: Demonstrates real-world viability (not just backtest)

**Venues**:
- **NeurIPS** (ML track): Strong MARL methodology
- **ICML** (Finance ML workshop): Application focus
- **AAAI** (AI applications): Deployment + safety constraints
- **JMLR** (long paper): Complete empirical study

**Timeline**: Escribir paper solo si MTF-MARL supera TD3 con p < 0.05 en ablation study

---

## 8. RISK ASSESSMENT: What if MTF-MARL Fails?

### Potential Failure Modes

1. **No Convergence** (learning unstable):
   - **Cause**: Joint critic overfits, H1/M1 actions conflictive
   - **Mitigation**: Reduce learning rate, increase entropy regularization
   - **Fallback**: Revert to TD3 with tuned hyperparams

2. **Coordination Failure** (M1 ignores H1):
   - **Cause**: Weak coupling between agents, M1 reward doesn't depend on H1
   - **Mitigation**: Increase weight of joint reward, add explicit coordination penalty
   - **Fallback**: Use H1 as hard constraint (rule-based) instead of learned

3. **Worse Performance than TD3**:
   - **Cause**: Increased complexity doesn't justify overhead, data insufficient
   - **Mitigation**: Simplify architecture (remove M1, keep H1 as feature)
   - **Fallback**: Use ensemble of TD3 agents instead

4. **Computational Infeasibility**:
   - **Cause**: Training time >1 week, GPU memory insufficient
   - **Mitigation**: Reduce buffer size, use model parallelism
   - **Fallback**: TD3 with H4 features (hybrid approach)

### Fallback Plan: Enhanced TD3

If MTF-MARL fails or is infeasible, enhance TD3 with:

```python
# Add H4 features to observation space (no second agent)
observation_space_enhanced = np.concatenate([
    current_24d_state,  # M1 features
    h4_trend_strength,  # ADX on H4
    h4_regime,          # Regime on H4
    h4_macro_turbulence # VIX + yield curve aggregated
])  # 27D total

# Train TD3 with enhanced observation
# Simpler than MTF-MARL, no coordination complexity
```

**Expected**: 70% of MTF-MARL benefit with 20% of implementation cost

---

## 9. TIMELINE & RESOURCE ALLOCATION

### Scenario 1: Quick Test Shows MTF-MARL Needed

| Week | Task | Hours | Resources |
|------|------|-------|-----------|
| 1 | MTF-MARL design review & spec | 8h | 1 ML engineer |
| 2 | Implementation (env + agents) | 40h | 1 ML engineer |
| 3 | Training infrastructure | 24h | 1 ML engineer + 4 GPUs |
| 4 | Ablation study execution | 120h compute | 4 GPUs parallel |
| 5 | Analysis + paper draft | 16h | 1 ML engineer + 1 researcher |
| **Total** | **5 weeks** | **88h labor + 120h compute** | **~$2000 GPU cost** |

### Scenario 2: Quick Test Shows TD3 Sufficient

| Week | Task | Hours | Resources |
|------|------|-------|-----------|
| 1 | Full TD3 training (2000 ep) | 24h compute | 1 GPU |
| 2 | Validation + safety testing | 16h | 1 ML engineer |
| 3 | Demo deployment + monitoring | 8h | 1 ML engineer + MT5 account |
| **Total** | **3 weeks** | **24h labor + 24h compute** | **~$100 GPU cost** |

**Decision Point**: Si Quick Test muestra TD3 funciona, ahorramos **2 weeks + $1900** vs MTF-MARL

---

## 10. CONCLUSION

### Key Takeaways

1. **MTF-MARL is theoretically superior** (faster learning, better DD control, higher Sharpe)
2. **BUT requires validation**: Quick Test must show TD3 insufficient
3. **Cost-benefit**: 2x training time, but 30-40% faster convergence = net neutral
4. **Production viability**: Latency overhead (4-8ms) is acceptable for forex
5. **Scientific contribution**: Potential NeurIPS/ICML paper if ablation study succeeds

### Recommendation

**STEP 1**: Complete Quick Test (100 episodes TD3) **ASAP**

**STEP 2**: Analyze metrics:
- If `dd_violation_rate > 0.20` OR `final_sharpe < 0.8` → **Go MTF-MARL**
- If `dd_violation_rate < 0.05` AND `final_sharpe > 1.2` → **Go Full TD3**
- Else → **Run 200-episode test for more data**

**STEP 3**: If MTF-MARL approved:
- Week 1: Design + spec review
- Week 2-3: Implementation + training
- Week 4: Ablation study
- Week 5: Analysis + decision

**STEP 4**: Production deployment (TD3 or MTF-MARL winner)

---

**Status**: 📋 DECISION PENDING - Awaiting Quick Test results (ETA: 1-2 hours after Feature Engineering completes)

**Next Action**: Monitor Feature Engineering completion, then execute:
```powershell
poetry run python scripts\train_drl_agent.py --episodes 100 --eval-freq 10 --symbols EURUSD
```

Once Quick Test completes, analyze metrics and make go/no-go decision on MTF-MARL implementation.
