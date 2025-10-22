# 🧠 MTF-MARL ARCHITECTURE: Multi-Timeframe Multi-Agent Reinforcement Learning

**Status**: 📋 DESIGN PHASE  
**Target**: Post-validation enhancement after Quick Test success  
**Scientific Basis**: Albrecht et al. - Shared Experience Actor-Critic (ASEAC)

---

## 1. MOTIVACIÓN: Limitaciones del Sistema Actual (Single-Agent TD3)

### ❌ Problema con TD3 Único
El sistema actual (`ForexTradingEnv` + `TD3Agent`) opera en **un solo timeframe** (M1/M5) con una sola política:

```python
# Sistema Actual: Single Agent
observation_space = Box(24,)  # 24D state
action_space = Box(2,)        # [position_size, entry/exit]
```

**Limitaciones**:
1. **Miopia Temporal**: El agente M1 optimiza para recompensa inmediata (siguiente minuto) sin visión macro (H4/Diario).
2. **Conflicto Risk/Reward**: No existe separación entre decisión estratégica (trend direction) y táctica (execution timing).
3. **DD Control Reactivo**: La lógica CMDP actual solo reacciona cuando el DD ya ocurrió, no lo previene proactivamente.

### ✅ Solución: Arquitectura Jerárquica Multi-Agente

Inspirada en **Albrecht's ASEAC** (Shared Experience Actor-Critic), se propone un sistema de **dos agentes coordinados**:

| Agente | Timeframe | Rol | Acción | Frecuencia |
|--------|-----------|-----|--------|------------|
| **H1 (High-Level)** | H4/Diario | Risk Manager + Trend Director | `max_position_limit ∈ [-1.0, 1.0]` | Cada 4 horas |
| **M1 (Low-Level)** | M1/M5 | Execution Specialist | `entry_size, exit_timing` | Cada minuto |

**Coordinación**: La acción de M1 está **restringida** por la decisión de H1:

```python
# Constraint propagation
if H1.action = 0.5:  # Long bias, max 50% position
    M1.action_space = Box(low=0.0, high=0.5)  # Can't short or exceed limit
```

---

## 2. DISEÑO TÉCNICO: MTF-MARL System

### A. Arquitectura de Agentes

```
┌─────────────────────────────────────────────────────────────┐
│                     MARKET DATA STREAM                       │
│  OHLCV M1 → Feature Engineering → State Vector (24D)        │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
    ┌────▼─────┐         ┌───▼──────┐
    │ H1 Agent │         │ M1 Agent │
    │ (H4 TF)  │         │ (M1 TF)  │
    └────┬─────┘         └───┬──────┘
         │ max_position     │ entry_size
         │ trend_bias       │ exit_timing
         └─────────┬──────────┘
                   │
            ┌──────▼───────┐
            │ Joint Action │
            │  Validator   │
            │  (CMDP)      │
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │ MT5 Executor │
            └──────────────┘
```

### B. Observation Space: Compartido + Específico

Cada agente recibe una observación compuesta de:

#### 1️⃣ **Shared State** (común para ambos):
```python
shared_obs = [
    # Price features [0-2]
    normalized_price, returns, volatility,
    
    # Technicals [3-8]
    rsi, macd, atr, bb_width, adx, cci,
    
    # Macro [13-15]
    vix_norm, fed_rate_norm, yield_curve_norm,
    
    # Regime [10-12]
    regime_trend, regime_range, regime_transition
]  # 16 dimensions
```

#### 2️⃣ **Agent-Specific State**:

**H1 Agent** (Risk Manager):
```python
h1_specific = [
    current_equity,
    total_drawdown_pct,
    daily_drawdown_pct,
    portfolio_volatility_7d,
    macro_turbulence_index,  # VIX + yield curve
    h4_trend_strength        # ADX on H4 timeframe
]  # 6 dimensions

h1_obs = np.concatenate([shared_obs, h1_specific])  # 22D
```

**M1 Agent** (Execution Specialist):
```python
m1_specific = [
    current_position_size,
    cash_balance_norm,
    spread_cost_norm,
    m1_volume_ratio,
    m1_momentum,
    h1_max_position_limit,  # ⚠️ CRITICAL: Constrains M1 action
    h1_trend_bias           # ⚠️ CRITICAL: Guides M1 direction
]  # 7 dimensions

m1_obs = np.concatenate([shared_obs, m1_specific])  # 23D
```

**🔑 Key Innovation**: M1 observa las decisiones de H1 (`h1_max_position_limit`, `h1_trend_bias`), garantizando coordinación implícita.

### C. Action Space: Jerárquico

#### H1 Agent (Strategic):
```python
h1_action_space = Box(
    low=np.array([-1.0, -1.0]),
    high=np.array([1.0, 1.0]),
    dtype=np.float32
)
# [0] trend_bias: -1.0 (short) → 1.0 (long)
# [1] max_position_limit: 0.0 (neutral) → 1.0 (aggressive)
```

**Ejecución**: H1 actúa cada 240 minutos (4 horas):
- Si `trend_bias > 0.3`: Permite solo posiciones long a M1
- Si `max_position_limit = 0.2`: M1 no puede superar 20% del capital

#### M1 Agent (Tactical):
```python
m1_action_space = Box(
    low=np.array([0.0, 0.0]),
    high=np.array([h1_max_position_limit, 1.0]),  # ⚠️ Dynamic constraint
    dtype=np.float32
)
# [0] entry_size: cantidad a entrar (limitada por H1)
# [1] exit_confidence: 0.0 (hold) → 1.0 (exit immediately)
```

**Ejecución**: M1 actúa cada minuto, pero:
- `entry_size` está limitada por `h1_max_position_limit`
- Si `h1_trend_bias < -0.5` (strong short), M1 no puede tomar longs

### D. Reward Function: Cooperativa vs Individual

#### Joint Reward (CMDP Colectivo):
```python
def compute_joint_reward(h1_action, m1_action, state):
    # 1. Calcula posición neta
    net_position = m1_action[0] * h1_action[1]  # entry_size * max_limit
    
    # 2. Ejecuta trade y calcula PnL
    pnl = execute_trade(net_position)
    
    # 3. Calcula drawdown CONJUNTO
    equity = state.current_equity + pnl
    daily_dd = (state.daily_peak - equity) / state.daily_peak
    total_dd = (state.peak_equity - equity) / state.peak_equity
    
    # 4. CMDP Catastrófico (igual que antes)
    if daily_dd > 0.05:
        return -1000.0, True  # Penalty + terminate
    if total_dd > 0.10:
        return -10000.0, True
    
    # 5. Reward base (Sharpe-like)
    reward = pnl / state.portfolio_volatility
    
    # 6. Penalty suave si se acerca a límites
    if daily_dd > 0.03:
        reward -= 100 * (daily_dd - 0.03)  # Soft penalty
    
    return reward, False
```

**⚠️ CRITICAL**: Ambos agentes reciben la **misma recompensa** (`joint_reward`), incentivando cooperación.

#### Individual Rewards (Opcional - para debugging):
```python
# H1: Penaliza volatilidad excesiva de su max_position_limit
h1_individual_reward = -0.1 * np.std(h1_actions_last_24h)

# M1: Penaliza slippage y costos de transacción
m1_individual_reward = -0.05 * total_spread_cost
```

---

## 3. APRENDIZAJE: Shared Experience Actor-Critic (ASEAC)

### A. Problema del Multi-Agent RL Tradicional

En MARL clásico, cada agente tiene su propio:
- Replay Buffer (separado)
- Critic Network (separado)
- Actor Network (separado)

**Resultado**: Aprendizaje lento, exploración redundante, y no-stationarity (el entorno cambia desde la perspectiva de cada agente porque el otro agente también está aprendiendo).

### B. Solución de Albrecht: ASEAC

**Paper**: "Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning" (Albrecht & Ramamoorthy, NeurIPS 2018)

**Key Insights**:
1. **Shared Replay Buffer**: Todas las transiciones `(s, a_h1, a_m1, r, s')` se almacenan en un solo buffer.
2. **Joint Critic**: Un solo Critic que evalúa `Q(s, a_h1, a_m1)` - el valor de la acción conjunta.
3. **Individual Actors**: Cada agente mantiene su propio Actor, pero aprende del mismo Critic.

```python
class SharedExperienceBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = []
    
    def add(self, state, h1_action, m1_action, reward, next_state, done):
        # Almacena transición conjunta
        transition = (state, h1_action, m1_action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch  # Usado por AMBOS agentes

class JointCritic(nn.Module):
    def __init__(self, state_dim, h1_action_dim, m1_action_dim):
        super().__init__()
        total_input = state_dim + h1_action_dim + m1_action_dim
        self.q_network = nn.Sequential(
            nn.Linear(total_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Q(s, a_h1, a_m1)
        )
    
    def forward(self, state, h1_action, m1_action):
        x = torch.cat([state, h1_action, m1_action], dim=1)
        return self.q_network(x)
```

### C. Training Loop

```python
# 1. Environment step (cada minuto)
state = env.get_state()  # 24D shared features

# 2. Agents act
if step % 240 == 0:  # H1 acts every 4 hours
    h1_action = h1_agent.act(state_h1)  # Uses h1_obs (22D)
else:
    h1_action = h1_agent.current_action  # Keep previous

m1_action = m1_agent.act(state_m1, h1_action)  # Uses m1_obs (23D) + h1_constraint

# 3. Execute joint action
next_state, reward, done = env.step(h1_action, m1_action)

# 4. Store in SHARED buffer
shared_buffer.add(state, h1_action, m1_action, reward, next_state, done)

# 5. Update agents (every N steps)
if len(shared_buffer) > batch_size:
    batch = shared_buffer.sample(batch_size)
    
    # Update Joint Critic (shared)
    joint_critic_loss = compute_td_error(batch, joint_critic, h1_target_actor, m1_target_actor)
    joint_critic.update(joint_critic_loss)
    
    # Update H1 Actor
    h1_actor_loss = -joint_critic(state, h1_actor(state_h1), m1_action).mean()
    h1_actor.update(h1_actor_loss)
    
    # Update M1 Actor
    m1_actor_loss = -joint_critic(state, h1_action, m1_actor(state_m1)).mean()
    m1_actor.update(m1_actor_loss)
```

**🔑 Ventaja**: Cuando H1 toma una decisión macro (ej. "reduce posición por dato FRED negativo"), su experiencia se usa para entrenar M1 **inmediatamente**, acelerando el aprendizaje.

---

## 4. CMDP COLECTIVO: Safety Constraint

El CMDP actual se extiende para validar la **acción conjunta**:

```python
class JointCMDPValidator:
    def __init__(self, max_daily_dd=0.05, max_total_dd=0.10):
        self.max_daily_dd = max_daily_dd
        self.max_total_dd = max_total_dd
    
    def validate_joint_action(self, h1_action, m1_action, state):
        # 1. Calcula posición neta propuesta
        net_position = m1_action[0] * h1_action[1]  # entry_size * max_limit
        
        # 2. Simula PnL con worst-case scenario
        worst_case_pnl = net_position * state.atr * -2.0  # 2 ATR loss
        
        # 3. Proyecta drawdown
        projected_dd = abs(worst_case_pnl) / state.equity
        
        # 4. Rechaza acción si viola constraints
        if state.daily_dd + projected_dd > self.max_daily_dd:
            return False, "Daily DD limit would be breached"
        if state.total_dd + projected_dd > self.max_total_dd:
            return False, "Total DD limit would be breached"
        
        return True, "Action allowed"
```

**Uso en env.step()**:
```python
def step(self, h1_action, m1_action):
    # Validate BEFORE execution
    is_valid, msg = self.cmdp_validator.validate_joint_action(h1_action, m1_action, self.state)
    
    if not is_valid:
        logger.warning(f"⚠️ CMDP blocked action: {msg}")
        return self.state, -100.0, False, {}  # Penalty but don't terminate
    
    # Proceed with execution...
```

---

## 5. ROADMAP DE IMPLEMENTACIÓN

### Phase 1: Foundation (Week 1 - Post Quick Test)
- [ ] **Task 1.1**: Crear `underdog/rl/multi_agent_env.py`
  - Extender `ForexTradingEnv` para aceptar acciones de 2 agentes
  - Implementar `step(h1_action, m1_action)`
  - Agregar H4 timeframe aggregation para H1 observations

- [ ] **Task 1.2**: Crear `underdog/rl/hierarchical_agents.py`
  - Clase `H1Agent` (risk manager): Actor 22D → 2D, Critic joint
  - Clase `M1Agent` (execution): Actor 23D → 2D (constrained), Critic joint
  - Clase `JointCritic`: Evalúa Q(s, a_h1, a_m1)

- [ ] **Task 1.3**: Implementar `SharedExperienceBuffer`
  - Almacenar transiciones conjuntas `(s, a_h1, a_m1, r, s')`
  - Sampling balanceado (garantizar diversidad de H1 actions)

### Phase 2: Training Infrastructure (Week 2)
- [ ] **Task 2.1**: Script `scripts/train_mtf_marl_agent.py`
  - Bucle de entrenamiento con coordinación H1/M1
  - Logging separado para H1 vs M1 actions
  - Checkpoint saving de ambos agentes

- [ ] **Task 2.2**: Métricas de Coordinación
  - Track `h1_action_stability` (cambios por hora)
  - Track `m1_constraint_violations` (intentos de superar límite H1)
  - Track `joint_reward_decomposition` (cuánto contribuye cada agente)

- [ ] **Task 2.3**: Visualización
  - Plot de `h1_max_position_limit` vs tiempo (debería ser smooth)
  - Plot de `m1_entry_size` vs `h1_limit` (M1 debería respetar límite)
  - Plot de DD colectivo vs límites CMDP

### Phase 3: Validation (Week 3)
- [ ] **Task 3.1**: Quick Test MTF-MARL (100 episodes)
  - Comparar Sharpe vs TD3 single-agent
  - Verificar coordinación: M1 no viola límites de H1
  - Confirmar DD control: 0 violaciones catastróficas

- [ ] **Task 3.2**: Ablation Study
  - **Baseline**: TD3 single-agent (actual)
  - **Variant 1**: MTF-MARL sin shared buffer (2 buffers separados)
  - **Variant 2**: MTF-MARL con shared buffer (ASEAC completo)
  - Comparar velocidad de convergencia

- [ ] **Task 3.3**: Stress Testing
  - Test con FRED macro shocks (interés súbito +2%)
  - Test con drawdown approach (DD al 4.5% → H1 debería reducir límite)
  - Test con cambio de régimen (trend → range)

### Phase 4: Production (Week 4)
- [ ] **Task 4.1**: Integración con MT5
  - H1 actualiza cada 4H via MT5 API
  - M1 ejecuta trades cada minuto
  - Logging de decisiones jerárquicas

- [ ] **Task 4.2**: Backtesting Completo
  - 3 años de datos (2022-2024)
  - 4 símbolos (EURUSD, GBPUSD, USDJPY, XAUUSD)
  - Target: Sharpe > 1.5, DD < 7%

- [ ] **Task 4.3**: Demo Live
  - 24h paper trading con MTF-MARL
  - Validar que H1 previene DD proactivamente
  - Validar que M1 optimiza execution

---

## 6. VENTAJAS CIENTÍFICAS vs TD3 Actual

| Métrica | TD3 Single-Agent | MTF-MARL (ASEAC) |
|---------|------------------|------------------|
| **Miopia Temporal** | ❌ Solo M1 (short-term) | ✅ H1 (long-term) + M1 (short-term) |
| **DD Control** | ⚠️ Reactivo (penaliza después) | ✅ Proactivo (H1 previene antes) |
| **Aprendizaje** | Lento (1 buffer, 1 agente) | ✅ Acelerado (shared buffer, 2 agentes) |
| **Coordinación** | N/A (solo 1 agente) | ✅ Implícita (M1 observa H1) |
| **Prop Firm Ready** | ⚠️ Necesita ajustes | ✅ Diseñado para cumplir reglas |

---

## 7. REFERENCIAS

### Papers Clave:
1. **Albrecht & Ramamoorthy (NeurIPS 2018)**  
   *"Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning"*  
   → Shared buffer + Joint Critic

2. **Heess et al. (ICML 2017)**  
   *"Emergence of Locomotion Behaviours in Rich Environments"*  
   → Hierarchical RL for continuous control

3. **Vezhnevets et al. (ICML 2017)**  
   *"FeUdal Networks for Hierarchical Reinforcement Learning"*  
   → Manager-Worker architecture (similar a H1-M1)

### Código de Referencia:
- **OpenAI Baselines**: Implementación de Shared Buffer  
  https://github.com/openai/baselines/blob/master/baselines/her/her.py

- **PyMARL**: Framework para MARL  
  https://github.com/oxwhirl/pymarl

---

## 8. NEXT STEPS (Immediate)

1. **BLOQUEO ACTUAL**: Esperar Quick Test de TD3 single-agent (validación base)
2. **PREPARACIÓN**: Revisar implementación actual de `ForexTradingEnv` y `TD3Agent` para identificar puntos de extensión
3. **DISEÑO DETALLADO**: Crear spec de `multi_agent_env.py` con interface de 2 agentes
4. **PROTOTIPO**: Implementar versión simplificada (H1 regla fija, solo M1 aprende) como proof-of-concept

**Timeline**: Iniciar desarrollo MTF-MARL solo si Quick Test TD3 tiene éxito (< 10% DD violations, learning curve positiva).

---

## 9. DECISION CRITERIA: ¿Cuándo implementar MTF-MARL?

### ✅ Implementar si Quick Test muestra:
- ❌ **DD violations > 20%**: TD3 no aprende a controlar riesgo → H1 necesario
- ❌ **Miopic behavior**: Agente toma trades que son rentables a corto plazo pero destruyen equity a largo plazo
- ❌ **Alta volatilidad de actions**: Agente cambia posiciones constantemente sin estrategia macro

### ⏸️ POSTPONER si Quick Test muestra:
- ✅ **DD violations < 5%**: TD3 funciona bien, no urgente
- ✅ **Learning curve clara**: Recompensa aumenta, DD disminuye
- ✅ **Sharpe > 1.0**: Performance ya es competitiva

**Recomendación**: Si TD3 funciona razonablemente bien, **primero completar full training (2000 episodes)** y evaluar en backtest. MTF-MARL es una mejora avanzada que requiere semanas de desarrollo adicional.

Si TD3 falla en controlar DD o muestra comportamiento miope, **priorizar MTF-MARL inmediatamente**.

---

**Status**: 📋 DESIGN COMPLETE - Awaiting Quick Test results for go/no-go decision.
