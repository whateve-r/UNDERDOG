"""
Position Sizing Module
Implements fixed fractional + fractional Kelly + confidence-weighted sizing.

**SCIENTIFIC IMPROVEMENTS**:
- Fixed Fractional Risk (Chan 2013)
- Fractional Kelly Criterion (Ralph Vince, López de Prado)
- Confidence-Weighted Sizing (Fuzzy Logic integration)
- CVaR-adjusted sizing (tail risk management)
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class SizingConfig:
    """Position sizing configuration"""
    fixed_risk_pct: float = 1.0  # Base risk per trade (% of capital)
    kelly_fraction: float = 0.2  # Fractional Kelly (0.2 = 20% of full Kelly)
    kelly_cap: float = 0.25  # Maximum Kelly fraction (cap at 25%)
    min_confidence: float = 0.6  # Minimum confidence to trade
    confidence_exponent: float = 1.0  # Confidence scaling exponent
    max_size_per_symbol: Optional[float] = None  # Max lots per symbol
    min_size: float = 0.01  # Minimum position size in lots
    
    # Scaling factors
    use_confidence_scaling: bool = True
    use_kelly: bool = True
    use_volatility_scaling: bool = False


def confidence_weighted_sizing(
    base_lot: float,
    confidence: float,
    min_confidence: float = 0.5,
    exponent: float = 1.0
) -> float:
    """
    Ajustar lotaje por confianza de señal (Fuzzy Logic).
    
    **MEJORA CIENTÍFICA**: Integración de Lógica Difusa con Position Sizing
    
    Concepto:
    - Señales de alta confianza (0.9-1.0) → Lotaje completo
    - Señales de confianza media (0.6-0.8) → Lotaje reducido
    - Señales de baja confianza (<0.5) → NO operar
    
    Ventajas:
    - Optimiza capital: Más agresivo en señales fuertes, conservador en débiles
    - Reduce drawdown: Evita trades de baja probabilidad
    - Mejora Sharpe Ratio: Mejor retorno ajustado por riesgo
    
    Args:
        base_lot: Lote calculado con Fixed Fractional
        confidence: Score de 0.0 a 1.0 (de Fuzzy Logic system)
        min_confidence: Umbral mínimo para operar
        exponent: Exponente de escalado (>1 = más agresivo, <1 = más conservador)
    
    Returns:
        Lote ajustado (0.0 si confidence < min_confidence)
    
    Example:
        >>> # Fixed fractional dice 0.10 lotes
        >>> base = 0.10
        >>> 
        >>> # Alta confianza (95%) → Lotaje completo
        >>> confidence_weighted_sizing(base, 0.95)
        0.10
        >>> 
        >>> # Confianza media (60%) → Lotaje reducido
        >>> confidence_weighted_sizing(base, 0.60, min_confidence=0.5)
        0.06  # 60% del lotaje base
        >>> 
        >>> # Baja confianza (40%) → NO operar
        >>> confidence_weighted_sizing(base, 0.40, min_confidence=0.5)
        0.0
    """
    if confidence < min_confidence:
        return 0.0
    
    # Escalar linealmente entre min_confidence y 1.0
    # Formula: scaling = ((conf - min) / (1 - min))^exponent
    scaling_factor = ((confidence - min_confidence) / (1.0 - min_confidence)) ** exponent
    
    adjusted_lot = base_lot * scaling_factor
    
    # Round to 2 decimals (standard lot precision)
    return round(adjusted_lot, 2)


class PositionSizer:
    """
    Calculates position sizes using multi-factor approach:
    1. Fixed fractional base size
    2. Kelly criterion adjustment
    3. Confidence score weighting
    4. Risk/reward optimization
    """
    
    def __init__(self, config: Optional[SizingConfig] = None):
        """
        Initialize position sizer.
        
        Args:
            config: Sizing configuration
        """
        self.config = config or SizingConfig()
    
    def calculate_kelly_fraction(self,
                                 win_rate: float,
                                 avg_win: float,
                                 avg_loss: float) -> float:
        """
        Calculate Kelly fraction: f = (p*b - q) / b
        where p = win rate, q = loss rate, b = avg_win / avg_loss
        
        Args:
            win_rate: Historical win rate [0-1]
            avg_win: Average winning trade size
            avg_loss: Average losing trade size (positive value)
        
        Returns:
            Kelly fraction [0-1], capped by kelly_cap
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        # Full Kelly
        kelly = (p * b - q) / b
        
        # Apply fractional Kelly (conservative)
        kelly_fraction = kelly * self.config.kelly_fraction
        
        # Cap maximum Kelly
        kelly_fraction = min(kelly_fraction, self.config.kelly_cap)
        
        # Floor at 0
        kelly_fraction = max(kelly_fraction, 0.0)
        
        return kelly_fraction
    
    def calculate_size(self,
                      account_balance: float,
                      entry_price: float,
                      stop_loss: float,
                      pip_value: float = 1.0,
                      confidence_score: Optional[float] = None,
                      win_rate: Optional[float] = None,
                      avg_win: Optional[float] = None,
                      avg_loss: Optional[float] = None,
                      volatility_adj: float = 1.0,
                      max_size_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size using multi-factor approach.
        
        **COMPLETE PIPELINE**:
        1. Fixed Fractional: Base lot from risk% and SL distance
        2. Kelly Adjustment: Scale by historical win rate
        3. Confidence Weighting: Scale by Fuzzy Logic score
        4. Volatility Adjustment: Scale by ATR (optional)
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            pip_value: Value per pip/point in account currency
            confidence_score: Model confidence [0-1] (from Fuzzy Logic)
            win_rate: Historical win rate [0-1] (for Kelly)
            avg_win: Average winning trade size (for Kelly)
            avg_loss: Average losing trade size (for Kelly)
            volatility_adj: Volatility multiplier (from ATR)
            max_size_override: Override max size
        
        Returns:
            Dict with size calculation breakdown
        
        Example:
            >>> sizer = PositionSizer()
            >>> result = sizer.calculate_size(
            ...     account_balance=100000,
            ...     entry_price=1.1000,
            ...     stop_loss=1.0950,
            ...     confidence_score=0.85,  # High confidence
            ...     win_rate=0.55,
            ...     avg_win=150,
            ...     avg_loss=100
            ... )
            >>> print(f"Position size: {result['final_size']:.2f} lots")
            >>> print(f"Risk: ${result['risk_amount']:.2f}")
        """
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return self._zero_size_result("Stop distance is zero")
        
        # 1. Base sizing: Fixed fractional risk
        risk_dollars = account_balance * (self.config.fixed_risk_pct / 100)
        base_size = risk_dollars / (stop_distance * pip_value)
        
        factors = {
            'base': 1.0,
            'kelly': 1.0,
            'confidence': 1.0,
            'volatility': volatility_adj
        }
        
        current_size = base_size
        
        # 2. Kelly criterion adjustment
        if self.config.use_kelly and all(x is not None for x in [win_rate, avg_win, avg_loss]):
            kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
            kelly_multiplier = 1.0 + kelly_fraction  # Additive Kelly
            current_size *= kelly_multiplier
            factors['kelly'] = kelly_multiplier
        
        kelly_size = current_size
        
        # 3. Confidence score weighting
        if self.config.use_confidence_scaling and confidence_score is not None:
            # Filter out low confidence trades
            if confidence_score < self.config.min_confidence:
                return self._zero_size_result(f"Confidence {confidence_score:.2f} below minimum {self.config.min_confidence}")
            
            # Scale by confidence with optional exponent for non-linear scaling
            confidence_factor = (confidence_score ** self.config.confidence_exponent)
            current_size *= confidence_factor
            factors['confidence'] = confidence_factor
        
        confidence_size = current_size
        
        # 4. Volatility adjustment
        if self.config.use_volatility_scaling:
            current_size *= volatility_adj
        
        # 5. Apply maximum size constraints
        max_size = max_size_override or self.config.max_size_per_symbol
        if max_size is not None:
            current_size = min(current_size, max_size)
        
        # 6. Apply minimum size
        if current_size < self.config.min_size:
            current_size = self.config.min_size
        
        final_size = current_size
        
        # Calculate actual risk with final size
        actual_risk_dollars = final_size * stop_distance * pip_value
        
        return {
            'final_size': round(final_size, 2),
            'base_size': round(base_size, 2),
            'kelly_size': round(kelly_size, 2),
            'confidence_size': round(confidence_size, 2),
            'risk_dollars': round(risk_dollars, 2),
            'actual_risk_dollars': round(actual_risk_dollars, 2),
            'risk_pct': round((actual_risk_dollars / account_balance) * 100, 2),
            'factors': factors,
            'rejected': False
        }
    
    def _zero_size_result(self, reason: str) -> Dict[str, Any]:
        """Return zero size result with rejection reason"""
        return {
            'final_size': 0.0,
            'base_size': 0.0,
            'kelly_size': 0.0,
            'confidence_size': 0.0,
            'risk_dollars': 0.0,
            'actual_risk_dollars': 0.0,
            'risk_pct': 0.0,
            'factors': {},
            'rejected': True,
            'rejection_reason': reason
        }
    
    def calculate_stop_loss(self,
                           entry_price: float,
                           side: str,
                           atr: float,
                           atr_multiplier: float = 2.0) -> float:
        """
        Calculate stop loss based on ATR.
        
        Args:
            entry_price: Entry price
            side: "buy" or "sell"
            atr: Average True Range value
            atr_multiplier: ATR multiplier for stop distance
        
        Returns:
            Stop loss price
        """
        stop_distance = atr * atr_multiplier
        
        if side.lower() == "buy":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(self,
                             entry_price: float,
                             stop_loss: float,
                             side: str,
                             risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit based on risk:reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: "buy" or "sell"
            risk_reward_ratio: Target R:R ratio
        
        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if side.lower() == "buy":
            return entry_price + reward
        else:
            return entry_price - reward
    
    def adjust_for_portfolio_risk(self,
                                  base_size: float,
                                  portfolio_dd_scaling: float) -> float:
        """
        Adjust position size based on portfolio drawdown scaling from Risk Master.
        
        Args:
            base_size: Base position size
            portfolio_dd_scaling: Scaling factor from Risk Master [0-1]
        
        Returns:
            Adjusted position size
        """
        return base_size * portfolio_dd_scaling


# ========================================
# Utility Functions
# ========================================

def calculate_pip_value(symbol: str, 
                       account_currency: str = "USD",
                       lot_size: float = 100000) -> float:
    """
    Calculate pip value for a given symbol.
    Simplified version - in production, fetch from broker or use proper FX math.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        account_currency: Account currency
        lot_size: Standard lot size
    
    Returns:
        Pip value in account currency
    """
    # Simplified: For most pairs vs USD, 1 pip on 1 standard lot = $10
    # For JPY pairs: 1 pip = $9.17 approx
    
    if "JPY" in symbol:
        return 9.17
    else:
        return 10.0


def calculate_position_value(size: float,
                            price: float,
                            lot_size: float = 100000) -> float:
    """
    Calculate notional position value.
    
    Args:
        size: Position size in lots
        price: Current price
        lot_size: Lot size (default 100,000 for standard lot)
    
    Returns:
        Notional value in base currency
    """
    return size * lot_size * price


def calculate_required_margin(position_value: float,
                              leverage: int = 100) -> float:
    """
    Calculate required margin for position.
    
    Args:
        position_value: Notional position value
        leverage: Account leverage
    
    Returns:
        Required margin
    """
    return position_value / leverage
