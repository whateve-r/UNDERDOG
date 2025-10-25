# 📊 RECOMENDACIONES DEL CONSULTOR - MTF-MARL ARCHITECTURE

**Fecha:** 23 de Octubre, 2025  
**Consultor:** Especialista en Deep RL para Trading  
**Proyecto:** UNDERDOG - Arquitectura Multi-Timeframe Multi-Agent RL  
**Estado Actual:** TD3 Single-Agent funcional (post 13 bug fixes)

---

## 🎯 EXECUTIVE SUMMARY

El consultor **VALIDA** la arquitectura MTF-MARL de dos niveles y proporciona referencias científicas concretas que respaldan cada decisión arquitectónica. Las recomendaciones se alinean perfectamente con la literatura de investigación más reciente (2024-2025).

**Recomendación Principal:** Implementar **A3C (sin lock) como Meta-Agente Coordinador** sobre 4 agentes TD3 locales (uno por par: EURUSD, GBPUSD, USDJPY, USDCHF).

**Papers Clave Validados:**
1. **2405.19982v1.pdf** - "DRL for Forex... Multi-Agent Asynchronous Distribution"
2. **ALA2017_Gupta.pdf** - "Cooperative Multi-Agent Control Using Actor-Critic"
3. **3745133.3745185.pdf** - "TD3 for Stock Trading" (Turbulence Index + 50+ indicators)
4. **new+Multi-Agent+Reinforcement+Learning...** - VDN + MAPPO (cooperative reward)

---

## 🏗️ 1. VALIDACIÓN ARQUITECTURA MARL (CTDE)

### **Modelo Confirmado: CTDE (Centralized Training, Decentralized Execution)**

```
┌─────────────────────────────────────────────────────────────────┐
│                  NIVEL 2: META-AGENTE (A3C)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Meta-State (12-15D):                                      │  │
│  │  • DD Global                                               │  │
│  │  • Balance Total                                           │  │
│  │  • Turbulence Index Global (promedio 4 pares)             │  │
│  │  • Posiciones Agregadas (4 pares)                         │  │
│  │  • Penalizaciones CMDP Agregadas                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           ↓                                      │
│  Meta-Action ∈ ℝ⁴: [risk_limit_EUR, risk_limit_GBP,            │
│                      risk_limit_JPY, risk_limit_CHF]            │
│                     (modula exposición máxima)                   │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│             NIVEL 1: AGENTES LOCALES (4× TD3)                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────┐│
│  │ TD3_EURUSD   │  │ TD3_GBPUSD   │  │ TD3_USDJPY   │  │ ... ││
│  │              │  │              │  │              │  │     ││
│  │ State (26D+):│  │ State (26D+):│  │ State (26D+):│  │     ││
│  │ • 24D base   │  │ • 24D base   │  │ • 24D base   │  │     ││
│  │ • Turb Local │  │ • Turb Local │  │ • Turb Local │  │     ││
│  │ • DXY/Corr   │  │ • DXY/Corr   │  │ • DXY/Corr   │  │     ││
│  │              │  │              │  │              │  │     ││
│  │ Action ∈[-1,1]│ │ Action ∈[-1,1]│ │ Action ∈[-1,1]│ │     ││
│  │ (clipped by  │  │ (clipped by  │  │ (clipped by  │  │     ││
│  │ Meta-Action) │  │ Meta-Action) │  │ Meta-Action) │  │     ││
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────┘│
└─────────────────────────────────────────────────────────────────┘
```

### **Validación Científica**

| **Componente** | **Algoritmo** | **Paper de Referencia** | **Razón Clave** |
|----------------|---------------|------------------------|-----------------|
| **Meta-Agente (Nivel 2)** | **A3C sin lock** | `2405.19982v1.pdf` | Superior a PPO para multi-currency. Entrenamiento asíncrono y paralelo esencial para 4 símbolos. |
| **Modelo de Coordinación** | **CTDE** | `ALA2017_Gupta.pdf` | Entrenamiento Centralizado (Meta-Agente actualiza política global) + Ejecución Descentralizada (4 TD3 actúan localmente). |
| **Mecanismo de Recompensa** | **Cooperativo** | `new+Multi-Agent+RL...` | Recompensa = Suma de recompensas individuales. Simplifica coordinación y optimiza portafolio global. |
| **Agentes Locales** | **TD3** | Implementado ✅ | Agentes de ejecución descentralizada. Mantener arquitectura actual (24D→1D). |

**Conclusión Arquitectónica:**
- ✅ TD3 actual = **Nivel 1 (Agente Local de Ejecución)**
- 🆕 A3C Meta-Agente = **Nivel 2 (Coordinador Centralizado)**
- 🆕 Meta-Action = **Modulador de Riesgo** (controla límites de exposición de cada agente local)

---

## 📊 2. INGENIERÍA DE FEATURES (STATE AUGMENTATION)

### **A. Turbulence Index (PRIORIDAD ALTA)**

**Paper:** `3745133.3745185.pdf` - TD3 para Stock Trading

**Concepto:**
- Mide volatilidad dinámica y condiciones anormales del mercado
- Red neuronal aprende a cerrar/reducir posiciones cuando turbulencia es alta

**Implementación:**

#### **2.1 Turbulence Index Local (Nivel 1 - TD3)**

```python
# En underdog/rl/environments.py - _get_observation()

def _calculate_turbulence_local(self, window: int = 20) -> float:
    """
    Calculate local turbulence index for this symbol
    
    Formula: σ(log_returns) over last N bars
    High turbulence → market stress → reduce position
    """
    if len(self.returns_history) < window:
        return 0.0
    
    recent_returns = np.array(self.returns_history[-window:])
    turbulence = np.std(recent_returns)  # Standard deviation of log returns
    
    # Normalize to [0, 1] range (0 = calm, 1 = extreme turbulence)
    # Assume 3-sigma as extreme
    turbulence_norm = np.clip(turbulence / 0.03, 0.0, 1.0)
    
    return turbulence_norm
```

**Integración en State Vector (24D → 25D):**
```python
state = np.array([
    # ... existing 24 features ...
    turbulence_local,          # [24] 🆕 LOCAL TURBULENCE
], dtype=np.float32)
```

#### **2.2 Turbulence Index Global (Nivel 2 - A3C Meta-Agent)**

```python
# En (NUEVO) underdog/rl/meta_agent.py

def _calculate_turbulence_global(self, agent_states: List[np.ndarray]) -> float:
    """
    Calculate global turbulence index across all 4 symbols
    
    Args:
        agent_states: List of 4 state vectors (one per TD3 agent)
    
    Returns:
        Global turbulence (average of local turbulences)
    """
    # Extract local turbulence from each agent's state (index 24)
    local_turbulences = [state[24] for state in agent_states]
    
    # Global turbulence = average (or max for conservative approach)
    turbulence_global = np.mean(local_turbulences)
    
    return turbulence_global
```

**Integración en Meta-State (12-15D):**
```python
meta_state = np.array([
    total_dd_pct,              # [0] Global DD
    total_balance_norm,        # [1] Portfolio balance
    turbulence_global,         # [2] 🆕 GLOBAL TURBULENCE
    # ... otros features meta ...
], dtype=np.float32)
```

### **B. State Augmentation - Features Inter-Mercado (PRIORIDAD MEDIA)**

**Paper:** `3745133.3745185.pdf` - Usa 50+ indicadores

**Recomendación:** Expandir state_dim de 24D a **30-40D**

**Features Adicionales Sugeridos:**

#### **2.3 Dollar Index (DXY) - Feature Inter-Mercado**

```python
# En environments.py

def _get_dxy_value(self) -> float:
    """
    Get current US Dollar Index (DXY) value
    
    DXY = weighted average of USD vs basket (EUR, GBP, JPY, CHF, CAD, SEK)
    High DXY → USD strengthening → impacts all USD pairs
    """
    # TODO: Fetch from database or calculate from major pairs
    # Placeholder: Synthetic DXY from EURUSD inverse
    
    eurusd_price = self.current_price  # Assuming EURUSD
    dxy_synthetic = 1.0 / eurusd_price  # Inverse relationship
    
    # Normalize to [0, 1] range
    dxy_norm = np.clip((dxy_synthetic - 0.9) / 0.2, 0.0, 1.0)
    
    return dxy_norm
```

#### **2.4 Cross-Pair Correlation**

```python
def _calculate_cross_correlation(self, other_symbol: str = "GBPUSD", window: int = 50) -> float:
    """
    Calculate correlation with another major pair (e.g., EURUSD vs GBPUSD)
    
    High correlation → pairs move together → portfolio diversification risk
    """
    # TODO: Fetch other symbol's price history from DB
    # Calculate rolling correlation coefficient
    
    # Placeholder: Return 0.7 (typical EURUSD/GBPUSD correlation)
    return 0.7
```

**State Vector Actualizado (24D → 28D):**
```python
state = np.array([
    # ... existing 24 features ...
    turbulence_local,          # [24] 🆕 LOCAL TURBULENCE
    dxy_norm,                  # [25] 🆕 DOLLAR INDEX
    cross_correlation,         # [26] 🆕 CROSS-PAIR CORRELATION
    vix_proxy,                 # [27] 🆕 VOLATILITY INDEX (proxy)
], dtype=np.float32)
```

---

## 🎁 3. FUNCIÓN DE RECOMPENSA AVANZADA

### **A. Recompensa Basada en Sharpe Ratio**

**Paper:** `3745133.3745185.pdf` (referencia a Rodinos et al.)

**Fórmula Recomendada:**
```
R_final = Retorno - λ × Penalización_CMDP + μ × Sharpe_Episodio
```

**Implementación:**

```python
# En environments.py - Al final del episodio (done=True)

def _calculate_final_reward(self) -> float:
    """
    Calculate episode-level reward combining return, DD penalty, and Sharpe
    
    Formula: R = Return - λ×DD_Penalty + μ×Sharpe
    
    Hyperparameters:
        λ = 0.5 (DD penalty weight)
        μ = 0.3 (Sharpe weight)
    """
    # 1. Simple return
    total_return = (self.equity - self.initial_balance_snapshot) / self.initial_balance_snapshot
    
    # 2. CMDP penalty (already applied step-by-step)
    total_cmdp_penalty = 0.0
    if self.daily_dd_ratio > self.config.max_daily_dd_pct:
        total_cmdp_penalty += 1000.0
    if self.total_dd_ratio > self.config.max_total_dd_pct:
        total_cmdp_penalty += 10000.0
    
    # 3. Sharpe Ratio (calculated in _get_info())
    info = self._get_info()
    sharpe_ratio = info.get('sharpe_ratio', 0.0)
    
    # Combine with weights
    lambda_dd = 0.5
    mu_sharpe = 0.3
    
    final_reward = total_return - lambda_dd * total_cmdp_penalty + mu_sharpe * sharpe_ratio
    
    return final_reward
```

**Modificación en `step()`:**
```python
def step(self, action):
    # ... existing code ...
    
    # Calculate step reward
    reward = self._calculate_reward()
    
    # If episode ended, add final reward component
    if terminated:
        final_bonus = self._calculate_final_reward()
        reward += final_bonus
    
    return obs, reward, terminated, truncated, info
```

### **B. Reward Clipping (Penalizar Pérdidas Persistentes)**

**Paper:** `3745133.3745185.pdf`

**Concepto:**
- Pequeñas pérdidas constantes erosionan capital gradualmente
- Clipping asimétrico penaliza indecisión/acumulación de spread

**Implementación:**

```python
def _calculate_reward(self) -> float:
    """Calculate step-wise reward with asymmetric clipping"""
    
    # ... existing reward calculation ...
    
    # Base reward (equity change)
    reward = (self.equity - self.equity_history[-1]) / self.initial_balance_snapshot
    
    # 🆕 ASYMMETRIC CLIPPING: Penalize persistent small losses
    if reward <= -0.01:  # Small loss
        reward = -0.05   # Amplify penalty (5x)
    
    # Apply CMDP penalties
    if self.daily_dd_ratio > self.config.max_daily_dd_pct:
        reward -= 1000.0
    
    if self.total_dd_ratio > self.config.max_total_dd_pct:
        reward -= 10000.0
    
    return reward
```

**Justificación:**
- Evita que el agente adopte estrategias "lentamente perdedoras"
- Fuerza exploración de acciones más decisivas
- Reduce DD gradual (más peligroso que DD catastrófico en Prop Firms)

---

## 🗺️ 4. PLAN DE IMPLEMENTACIÓN (PRIORIZADO)

### **FASE 1: VALIDAR TD3 ACTUAL** ⏰ ESTA SEMANA (CRÍTICO)

**Objetivo:** Decidir si TD3 single-agent es suficiente ANTES de invertir en MARL

**Tareas:**
1. ✅ Completar Quick Test (100 episodios) - **EN PROGRESO**
2. ⏳ Analizar métricas clave:
   - DD violation rate (target: <5%)
   - Sharpe ratio (target: >0.5)
   - Win rate (target: >40%)
3. **Decision Point:**
   - **Si DD < 5% Y Sharpe > 0.5:** → Proceder con TD3 full training (2000 ep)
   - **Si DD > 10% O Sharpe < 0:** → **IMPLEMENTAR MTF-MARL**

**No iniciar MARL hasta confirmar que TD3 no es suficiente.**

### **FASE 2: MEJORAR TD3 ACTUAL** ⏰ 3-5 DÍAS (SI QUICK TEST ES PROMETEDOR)

**Antes de MARL, implementar mejoras de bajo costo:**

| # | Tarea | Componente | Esfuerzo | Prioridad |
|---|-------|------------|----------|-----------|
| 2.1 | **Turbulence Index Local** | `environments.py` | 2-3 horas | 🔴 ALTA |
| 2.2 | **Reward con Sharpe** | `environments.py` | 2 horas | 🔴 ALTA |
| 2.3 | **Reward Clipping** | `environments.py` | 1 hora | 🟡 MEDIA |
| 2.4 | **DXY Feature** | `environments.py` | 3-4 horas | 🟡 MEDIA |
| 2.5 | **State 24D→28D** | `environments.py` + `agents.py` | 2 horas | 🟢 BAJA |

**Resultado Esperado:**
- TD3 mejorado con features avanzadas
- 2nd Quick Test (100 ep) con nuevo state/reward
- **Si mejora significativa:** Continuar con TD3
- **Si mejora marginal:** Proceder a MARL

### **FASE 3: IMPLEMENTAR MTF-MARL** ⏰ 2-3 SEMANAS (SI TD3 NO ES SUFICIENTE)

**Decisión Confirmada:** Implementar A3C Meta-Agente sobre 4× TD3 Locales

#### **3.1 Crear MultiAssetEnv** (Entorno de Coordinación)

**Archivo:** `underdog/rl/multi_asset_env.py` (NUEVO)

```python
import gymnasium as gym
from typing import Dict, List, Tuple
import numpy as np

class MultiAssetEnv(gym.Env):
    """
    Multi-Asset Forex Trading Environment for MARL
    
    Coordinates 4 ForexTradingEnv instances (EURUSD, GBPUSD, USDJPY, USDCHF)
    Provides Meta-State to A3C coordinator
    Applies Meta-Actions (risk limits) to local agents
    """
    
    def __init__(
        self,
        symbols: List[str] = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
        initial_balance: float = 100000.0,
        config: Dict = None
    ):
        super().__init__()
        
        self.symbols = symbols
        self.num_agents = len(symbols)
        
        # Create 4 local ForexTradingEnv instances
        self.local_envs = [
            ForexTradingEnv(
                config=config,
                symbol=symbol,
                initial_balance=initial_balance / self.num_agents  # Split capital
            )
            for symbol in symbols
        ]
        
        # Meta-State space (12-15D)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),  # Meta-state dimension
            dtype=np.float32
        )
        
        # Meta-Action space (4D - one risk limit per agent)
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),  # Risk limit for each agent [0, 1]
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset all 4 local environments"""
        super().reset(seed=seed)
        
        local_states = []
        for env in self.local_envs:
            state, info = env.reset()
            local_states.append(state)
        
        # Construct Meta-State
        meta_state = self._build_meta_state(local_states)
        
        return meta_state, {}
    
    def step(self, meta_action: np.ndarray):
        """
        Execute one coordinated step
        
        Args:
            meta_action: [risk_limit_EUR, risk_limit_GBP, risk_limit_JPY, risk_limit_CHF]
                        Each value in [0, 1] represents max position size allowed
        
        Returns:
            meta_state, total_reward, terminated, truncated, info
        """
        # 1. Apply Meta-Action to each local agent (risk limits)
        for i, env in enumerate(self.local_envs):
            env.config.max_position_size = float(meta_action[i])
        
        # 2. Let each local agent take its own action (TD3)
        local_states = []
        local_rewards = []
        local_dones = []
        
        for env in self.local_envs:
            # Local agent selects action (TD3 policy)
            local_action = env.agent.select_action(env._get_observation())
            
            # Execute in local environment
            state, reward, done, truncated, info = env.step(local_action)
            
            local_states.append(state)
            local_rewards.append(reward)
            local_dones.append(done or truncated)
        
        # 3. Aggregate rewards (cooperative MARL)
        total_reward = sum(local_rewards)
        
        # 4. Build Meta-State
        meta_state = self._build_meta_state(local_states)
        
        # 5. Termination condition (any agent breached DD or all done)
        terminated = any(local_dones)
        
        # 6. Info dict
        info = {
            'local_rewards': local_rewards,
            'local_dds': [env.total_dd_ratio for env in self.local_envs],
            'global_dd': self._calculate_global_dd(),
        }
        
        return meta_state, total_reward, terminated, False, info
    
    def _build_meta_state(self, local_states: List[np.ndarray]) -> np.ndarray:
        """
        Build Meta-State from 4 local states
        
        Meta-State (15D):
            [0] Global DD (%)
            [1] Total Balance (normalized)
            [2] Turbulence Global (average of 4 local turbulences)
            [3-6] Local DD ratios (4 agents)
            [7-10] Local position sizes (4 agents)
            [11-14] Local balance ratios (4 agents)
        """
        # Extract features from local states
        turbulences = [state[24] for state in local_states]  # Index 24 = turbulence_local
        
        # Calculate global metrics
        global_dd = self._calculate_global_dd()
        total_balance = sum(env.balance for env in self.local_envs)
        total_balance_norm = total_balance / (self.num_agents * 100000.0)
        turbulence_global = np.mean(turbulences)
        
        # Local metrics
        local_dds = [env.total_dd_ratio for env in self.local_envs]
        local_positions = [env.position_size for env in self.local_envs]
        local_balances = [env.balance / 25000.0 for env in self.local_envs]  # Normalize by 1/4 initial
        
        meta_state = np.array([
            global_dd,           # [0]
            total_balance_norm,  # [1]
            turbulence_global,   # [2]
            *local_dds,          # [3-6]
            *local_positions,    # [7-10]
            *local_balances,     # [11-14]
        ], dtype=np.float32)
        
        return meta_state
    
    def _calculate_global_dd(self) -> float:
        """Calculate global portfolio drawdown"""
        total_balance = sum(env.balance for env in self.local_envs)
        initial_balance = self.num_agents * 100000.0
        peak_balance = max(
            sum(env.peak_balance for env in self.local_envs),
            initial_balance
        )
        
        global_dd = (peak_balance - total_balance) / peak_balance if peak_balance > 0 else 0.0
        
        return global_dd
```

**Esfuerzo:** 2-3 días (implementación + debugging)

#### **3.2 Implementar A3C Meta-Agente**

**Archivo:** `underdog/rl/meta_agent.py` (NUEVO)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class A3CMetaAgent(nn.Module):
    """
    A3C Meta-Agent for coordinating 4 TD3 local agents
    
    Architecture:
        Meta-State (15D) → Shared Network → Actor (4D) + Critic (1D)
    
    Meta-Action:
        4D vector controlling risk limits for each local agent
    """
    
    def __init__(
        self,
        meta_state_dim: int = 15,
        meta_action_dim: int = 4,
        hidden_dim: int = 128,
        lr: float = 3e-4
    ):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(meta_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, meta_action_dim),
            nn.Sigmoid()  # Output in [0, 1] for risk limits
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, meta_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            (meta_action_probs, state_value)
        """
        shared_features = self.shared(meta_state)
        
        meta_action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        
        return meta_action_probs, state_value
    
    def select_meta_action(self, meta_state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select Meta-Action (risk limits for 4 agents)
        
        Args:
            meta_state: Meta-State vector (15D)
            explore: Add noise for exploration
        
        Returns:
            meta_action: [risk_limit_EUR, risk_limit_GBP, risk_limit_JPY, risk_limit_CHF]
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(meta_state).unsqueeze(0)
            meta_action_probs, _ = self.forward(state_tensor)
            meta_action = meta_action_probs.squeeze(0).numpy()
        
        # Add exploration noise
        if explore:
            noise = np.random.normal(0, 0.1, size=meta_action.shape)
            meta_action = np.clip(meta_action + noise, 0.0, 1.0)
        
        return meta_action
```

**Esfuerzo:** 3-4 días (implementación + integración con MultiAssetEnv)

#### **3.3 Training Loop para MARL**

**Archivo:** `scripts/train_marl_agent.py` (NUEVO)

```python
# Training loop que coordina:
# 1. Meta-Agente (A3C) selecciona risk limits
# 2. 4× TD3 locales ejecutan trades
# 3. Meta-Agente aprende de recompensa agregada
```

**Esfuerzo:** 2-3 días (adaptación de train_drl_agent.py)

### **FASE 4: TESTING & VALIDATION** ⏰ 1 SEMANA

1. **Unit Tests:** MultiAssetEnv, A3CMetaAgent
2. **Quick Test MARL:** 100 episodios con 4 símbolos
3. **Comparación:** TD3 single-agent vs MARL
4. **Decision:** Elegir arquitectura final para production

---

## 📊 5. CRONOGRAMA REVISADO

```
SEMANA 1 (ACTUAL):
├─ [✅] Completar Quick Test TD3 (100 ep)
├─ [⏳] Analizar resultados
└─ [⏳] DECISION POINT: TD3 vs MARL

SEMANA 2 (SI TD3 PROMETEDOR):
├─ [🔴] Implementar Turbulence Index
├─ [🔴] Reward con Sharpe Ratio
├─ [🟡] Reward Clipping
├─ [🟡] DXY Feature
├─ [⏳] 2nd Quick Test (100 ep)
└─ [⏳] DECISION: Continuar TD3 OR Pivot MARL

SEMANA 3-4 (SI MARL NECESARIO):
├─ [🆕] MultiAssetEnv (2-3 días)
├─ [🆕] A3CMetaAgent (3-4 días)
├─ [🆕] Training Loop MARL (2-3 días)
└─ [🆕] Quick Test MARL (100 ep, 4 symbols)

SEMANA 5-6:
├─ [⏳] Full Training (2000 ep)
├─ [⏳] Hyperparameter Tuning
├─ [⏳] Paper Trading (30 días)
└─ [⏳] FTMO Demo Challenge
```

---

## 🎯 6. RECOMENDACIONES FINALES

### **DO (HACER AHORA)**

1. ✅ **Esperar Quick Test** (100 ep) - NO tomar decisiones sin datos
2. 🔴 **Si DD > 10%:** Implementar Turbulence Index ANTES que MARL
3. 🔴 **Si Sharpe < 0:** Implementar Reward con Sharpe ANTES que MARL
4. 🟡 **Documentar papers:** Crear referencias bibliográficas en docs/

### **DON'T (NO HACER AÚN)**

1. ❌ **NO implementar MARL** hasta confirmar que TD3 single-agent NO es suficiente
2. ❌ **NO sobre-ingenierizar:** MARL añade ~3 semanas de desarrollo
3. ❌ **NO ignorar Quick Test:** Es la métrica más importante

### **MAYBE (CONSIDERAR)**

1. 🟡 **Híbrido:** Usar TD3 mejorado (con Turbulence + Sharpe) en 1 símbolo
2. 🟡 **Escalado gradual:** Si TD3 funciona, añadir 2do símbolo (sin MARL)
3. 🟡 **Benchmark:** Comparar TD3 single vs 4× TD3 independientes (sin coordinación)

---

## 📚 7. REFERENCIAS CIENTÍFICAS

### **Papers Clave**

1. **2405.19982v1.pdf** - "Deep Reinforcement Learning for Forex Trading with Multi-Agent Asynchronous Distribution"
   - Propone A3C sin lock para multi-currency
   - Validación empírica: A3C > PPO en Forex
   - **Aplicación:** Meta-Agente Coordinador

2. **ALA2017_Gupta.pdf** - "Cooperative Multi-Agent Control Using Deep Reinforcement Learning"
   - Modelo CTDE (Centralized Training, Decentralized Execution)
   - Actor-Critic multi-agente
   - **Aplicación:** Arquitectura de coordinación

3. **3745133.3745185.pdf** - "Deep Reinforcement Learning based Trading Agent for Stock Market"
   - Turbulence Index para detección de estrés
   - 50+ indicadores técnicos
   - Reward Clipping asimétrico
   - **Aplicación:** Feature engineering + Reward shaping

4. **new+Multi-Agent+Reinforcement+Learning...** - VDN + MAPPO para HFT
   - Value Decomposition Networks
   - Recompensa cooperativa (suma de individuales)
   - **Aplicación:** Diseño de recompensa global

### **Integración con UNDERDOG**

| **Paper** | **Concepto** | **Módulo UNDERDOG** | **Estado** |
|-----------|-------------|---------------------|-----------|
| 2405.19982v1 | A3C Multi-Currency | `meta_agent.py` | 🔴 NO IMPLEMENTADO |
| ALA2017_Gupta | CTDE | `multi_asset_env.py` | 🔴 NO IMPLEMENTADO |
| 3745133.3745185 | Turbulence Index | `environments.py` | 🟡 DISEÑADO |
| 3745133.3745185 | Reward Sharpe | `environments.py` | 🟡 DISEÑADO |
| 3745133.3745185 | Reward Clipping | `environments.py` | 🟡 DISEÑADO |

---

## ✅ 8. CHECKLIST DE IMPLEMENTACIÓN

### **Fase 1: Validación (ESTA SEMANA)**
- [ ] Completar Quick Test TD3 (100 ep)
- [ ] Analizar DD violation rate
- [ ] Analizar Sharpe ratio evolution
- [ ] Analizar Win rate
- [ ] **DECISION POINT:** TD3 suficiente? (Sí/No)

### **Fase 2: Mejoras TD3 (SI PROMETEDOR)**
- [ ] Implementar `_calculate_turbulence_local()` en `environments.py`
- [ ] Añadir turbulence_local al state vector (24D→25D)
- [ ] Implementar `_calculate_final_reward()` con Sharpe
- [ ] Implementar reward clipping en `_calculate_reward()`
- [ ] Actualizar `state_dim=25` en `agents.py` config
- [ ] 2nd Quick Test (100 ep) con mejoras
- [ ] **DECISION POINT:** Mejora significativa? (Sí/No)

### **Fase 3: MARL (SI TD3 NO SUFICIENTE)**
- [ ] Crear `underdog/rl/multi_asset_env.py`
- [ ] Implementar `MultiAssetEnv` class (500+ líneas)
- [ ] Crear `underdog/rl/meta_agent.py`
- [ ] Implementar `A3CMetaAgent` class (200+ líneas)
- [ ] Crear `scripts/train_marl_agent.py`
- [ ] Adaptar training loop para MARL
- [ ] Unit tests para MultiAssetEnv
- [ ] Unit tests para A3CMetaAgent
- [ ] Quick Test MARL (100 ep, 4 símbolos)
- [ ] Comparación benchmark: TD3 vs MARL

### **Fase 4: Production (FINAL)**
- [ ] Full training (2000 ep) con arquitectura elegida
- [ ] Hyperparameter tuning (Optuna)
- [ ] Paper trading validation (30 días)
- [ ] FTMO demo challenge
- [ ] Live funded account

---

**FIN DE RECOMENDACIONES DEL CONSULTOR**

**Next Action:** ⏰ Esperar resultados del Quick Test (100 ep) antes de proceder.
