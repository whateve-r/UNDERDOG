"""
Conditional Value at Risk (CVaR) - Expected Shortfall
Métrica de riesgo de cola para optimización robusta.

CVaR (también llamado Expected Shortfall o ES) mide la pérdida promedio
esperada en los peores escenarios (tail risk).

Ventajas sobre VaR y Sharpe:
- Captura el riesgo de cola completo (no solo un percentil)
- Coherent risk measure (cumple axiomas matemáticos)
- Mejor para optimización (penaliza drawdowns extremos)
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable, Any
from scipy import stats


def calculate_var(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Calcular Value at Risk (VaR).
    
    VaR es el percentil de pérdidas al nivel de confianza dado.
    
    Args:
        returns: Array de returns (pueden ser negativos para pérdidas)
        confidence_level: Nivel de confianza (0.95 = peor 5%)
    
    Returns:
        VaR (valor positivo = pérdida)
    
    Example:
        >>> returns = np.array([0.02, -0.01, 0.03, -0.05, 0.01, -0.02])
        >>> var_95 = calculate_var(returns, 0.95)
        >>> print(f"En el 95% de los días, pérdida no excederá {var_95:.2%}")
    """
    # Percentil de pérdidas (cola izquierda)
    percentile = (1 - confidence_level) * 100
    var = np.percentile(returns, percentile)
    
    # Retornar como valor positivo (magnitud de pérdida)
    return abs(var)


def calculate_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Calcular Conditional Value at Risk (CVaR / Expected Shortfall).
    
    CVaR es la pérdida PROMEDIO esperada en el peor (1-α)% de escenarios.
    Es una medida de riesgo más robusta que VaR.
    
    Matemáticamente:
        CVaR_α = E[L | L ≥ VaR_α]
    
    Donde L son las pérdidas y α es el nivel de confianza.
    
    Args:
        returns: Array de returns diarios/por-trade
        confidence_level: Nivel de confianza (0.95 = peor 5%)
    
    Returns:
        CVaR (pérdida promedio en cola, valor positivo)
    
    Example:
        >>> returns = pd.Series([...])  # Returns históricos
        >>> cvar_95 = calculate_cvar(returns.values, 0.95)
        >>> print(f"En el peor 5% de días, pérdida promedio: {cvar_95:.2%}")
        >>> # Interpretación: "Si entramos en drawdown severo, esperamos perder ~cvar_95%"
    """
    # Calcular VaR primero
    var = -calculate_var(returns, confidence_level)  # Negativo para threshold
    
    # CVaR: Promedio de returns peores que VaR
    tail_returns = returns[returns <= var]
    
    if len(tail_returns) == 0:
        # No hay pérdidas en la cola (returns muy buenos)
        return 0.0
    
    cvar = tail_returns.mean()
    
    # Retornar como valor positivo
    return abs(cvar)


def calculate_cvar_percentile(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calcular CVaR con percentiles adicionales.
    
    Returns:
        Dict con VaR, CVaR, y distribución de tail losses
    """
    var = calculate_var(returns, confidence_level)
    cvar = calculate_cvar(returns, confidence_level)
    
    # Tail losses (peores que VaR)
    var_threshold = -var
    tail_losses = returns[returns <= var_threshold]
    
    if len(tail_losses) == 0:
        return {
            'var': var,
            'cvar': cvar,
            'tail_count': 0,
            'tail_pct': 0.0,
            'worst_loss': 0.0
        }
    
    return {
        'var': float(var),
        'cvar': float(cvar),
        'tail_count': len(tail_losses),
        'tail_pct': (len(tail_losses) / len(returns)) * 100,
        'worst_loss': float(abs(tail_losses.min())),
        'tail_mean': float(abs(tail_losses.mean())),
        'tail_std': float(abs(tail_losses.std()))
    }


def calculate_modified_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    adjust_for_skewness: bool = True
) -> float:
    """
    CVaR modificado para distribuciones no normales.
    
    Los returns financieros suelen tener:
    - Asimetría negativa (skew < 0): Más pérdidas extremas que ganancias
    - Curtosis elevada (fat tails): Eventos extremos más frecuentes
    
    Cornish-Fisher ajusta VaR por skewness y kurtosis.
    
    Args:
        returns: Array de returns
        confidence_level: Nivel de confianza
        adjust_for_skewness: Si ajustar por momentos superiores
    
    Returns:
        CVaR ajustado
    """
    if not adjust_for_skewness:
        return calculate_cvar(returns, confidence_level)
    
    # Calcular momentos
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
    
    # Z-score normal para confidence level
    z = stats.norm.ppf(1 - confidence_level)
    
    # Cornish-Fisher adjustment
    z_cf = (
        z 
        + (z**2 - 1) * skew / 6
        + (z**3 - 3*z) * kurt / 24
        - (2*z**3 - 5*z) * (skew**2) / 36
    )
    
    # VaR ajustado
    var_cf = -(mean + std * z_cf)
    
    # CVaR: Promedio de tail losses
    tail_returns = returns[returns <= -var_cf]
    
    if len(tail_returns) == 0:
        return 0.0
    
    cvar_cf = abs(tail_returns.mean())
    
    return cvar_cf


def optimize_by_cvar(
    strategy_fn: Callable,
    param_grid: Dict[str, list],
    data: pd.DataFrame,
    max_cvar: float = 0.02,  # Máximo 2% CVaR
    confidence_level: float = 0.95
) -> Tuple[Dict, pd.DataFrame]:
    """
    Optimizar estrategia MINIMIZANDO CVaR.
    
    Más robusto que maximizar Sharpe Ratio porque:
    - Penaliza drawdowns extremos (tail risk)
    - Optimiza para worst-case scenarios
    - Cumple con requisitos de Prop Firms (control de riesgo)
    
    Args:
        strategy_fn: Función de estrategia que retorna results dict
        param_grid: Dict de parámetros a optimizar
        data: DataFrame de market data
        max_cvar: CVaR máximo aceptable (constraint)
        confidence_level: Nivel de confianza
    
    Returns:
        Tuple de (best_params, results_df)
    
    Example:
        >>> def my_strategy(data, rsi_period, rsi_threshold):
        ...     # Implementación
        ...     return {'returns': returns_array, 'sharpe': 1.5}
        >>> 
        >>> param_grid = {
        ...     'rsi_period': [10, 14, 20],
        ...     'rsi_threshold': [30, 35, 40]
        ... }
        >>> 
        >>> best_params, results = optimize_by_cvar(
        ...     my_strategy,
        ...     param_grid,
        ...     data,
        ...     max_cvar=0.02
        ... )
        >>> print(f"Mejores parámetros: {best_params}")
        >>> print(f"CVaR: {results.loc[0, 'cvar']:.2%}")
    """
    from itertools import product
    
    # Generar todas las combinaciones de parámetros
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"[CVaR Optimization] Testing {len(param_combinations)} parameter combinations...")
    
    results = []
    best_cvar = float('inf')
    best_params = None
    
    for i, param_combo in enumerate(param_combinations):
        # Crear dict de parámetros
        params = dict(zip(param_names, param_combo))
        
        # Ejecutar backtest
        try:
            backtest_results = strategy_fn(data, **params)
            returns = backtest_results['returns']
            
            # Calcular CVaR
            cvar = calculate_cvar(returns, confidence_level)
            cvar_details = calculate_cvar_percentile(returns, confidence_level)
            
            # Métricas adicionales
            sharpe = backtest_results.get('sharpe', 0.0)
            max_dd = backtest_results.get('max_drawdown', 0.0)
            
            # Guardar resultado
            result = {
                **params,
                'cvar': cvar,
                'var': cvar_details['var'],
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'worst_loss': cvar_details['worst_loss'],
                'tail_count': cvar_details['tail_count']
            }
            results.append(result)
            
            # Check if mejor (minimizar CVaR)
            if cvar < best_cvar and cvar <= max_cvar:
                best_cvar = cvar
                best_params = params
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Tested {i+1}/{len(param_combinations)} combinations...")
        
        except Exception as e:
            print(f"  ERROR with params {params}: {e}")
            continue
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cvar')  # Ordenar por CVaR (mejor primero)
    
    print(f"\n[CVaR Optimization] Complete!")
    print(f"Best params: {best_params}")
    print(f"Best CVaR: {best_cvar:.4f} ({best_cvar*100:.2f}%)")
    
    if best_params is None:
        print(f"WARNING: No parameters found with CVaR <= {max_cvar:.2%}")
        # Return least bad option
        best_params = results_df.iloc[0][param_names].to_dict()
    
    return best_params, results_df


def calculate_cvar_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95
) -> float:
    """
    CVaR Ratio = (Mean Return - Risk Free Rate) / CVaR
    
    Similar al Sharpe Ratio pero usando CVaR en lugar de std.
    
    Interpretación:
    - Alto CVaR Ratio: Buenos returns con bajo tail risk
    - Bajo CVaR Ratio: Returns mediocres o alto tail risk
    
    Args:
        returns: Array de returns
        risk_free_rate: Tasa libre de riesgo (anualizada)
        confidence_level: Nivel de confianza para CVaR
    
    Returns:
        CVaR Ratio
    """
    mean_return = np.mean(returns)
    cvar = calculate_cvar(returns, confidence_level)
    
    if cvar == 0:
        return 0.0
    
    # Ajustar risk-free rate a frecuencia de returns
    # Asumiendo returns diarios
    daily_rf = risk_free_rate / 252
    
    cvar_ratio = (mean_return - daily_rf) / cvar
    
    return cvar_ratio


def calculate_risk_contribution(
    returns: np.ndarray,
    strategy_returns: Dict[str, np.ndarray],
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calcular contribución de cada estrategia al CVaR del portafolio.
    
    Útil para:
    - Identificar estrategias que contribuyen más al tail risk
    - Rebalancear portafolio para reducir CVaR
    
    Args:
        returns: Returns totales del portafolio
        strategy_returns: Dict mapping strategy_id → returns array
        confidence_level: Nivel de confianza
    
    Returns:
        Dict con % de contribución al CVaR de cada estrategia
    """
    portfolio_cvar = calculate_cvar(returns, confidence_level)
    
    contributions = {}
    total_contribution = 0.0
    
    for strategy_id, strat_returns in strategy_returns.items():
        # CVaR marginal: CVaR si eliminamos esta estrategia
        other_returns = returns - strat_returns
        other_cvar = calculate_cvar(other_returns, confidence_level)
        
        # Contribución = CVaR_portfolio - CVaR_without_strategy
        contribution = portfolio_cvar - other_cvar
        contributions[strategy_id] = contribution
        total_contribution += abs(contribution)
    
    # Normalizar a porcentajes
    if total_contribution > 0:
        contributions = {
            k: (v / total_contribution) * 100
            for k, v in contributions.items()
        }
    
    return contributions


# ====================================
# EJEMPLO DE USO
# ====================================

def example_cvar_analysis():
    """Ejemplo de análisis CVaR"""
    
    # Simular returns de estrategia
    np.random.seed(42)
    
    # Estrategia con tail risk (skew negativo)
    normal_returns = np.random.normal(0.001, 0.02, 200)
    crash_returns = np.random.normal(-0.05, 0.01, 10)  # Crash events
    returns = np.concatenate([normal_returns, crash_returns])
    np.random.shuffle(returns)
    
    print("="*60)
    print(" CVaR ANALYSIS - TAIL RISK ASSESSMENT")
    print("="*60)
    
    # Calcular VaR y CVaR
    var_95 = calculate_var(returns, 0.95)
    cvar_95 = calculate_cvar(returns, 0.95)
    
    print(f"\nRisk Metrics (95% confidence):")
    print(f"  VaR (95%):  {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"  CVaR (95%): {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"\nInterpretación:")
    print(f"  - En el 95% de los días, pérdida no excederá {var_95*100:.2f}%")
    print(f"  - En el peor 5% de días, pérdida promedio será {cvar_95*100:.2f}%")
    
    # Detalles de tail
    cvar_details = calculate_cvar_percentile(returns, 0.95)
    print(f"\nTail Loss Distribution:")
    print(f"  Tail events: {cvar_details['tail_count']} ({cvar_details['tail_pct']:.1f}%)")
    print(f"  Worst loss: {cvar_details['worst_loss']*100:.2f}%")
    print(f"  Tail mean: {cvar_details['tail_mean']*100:.2f}%")
    print(f"  Tail std: {cvar_details['tail_std']*100:.2f}%")
    
    # CVaR modificado (ajuste por skewness)
    cvar_modified = calculate_modified_cvar(returns, 0.95, adjust_for_skewness=True)
    print(f"\nCVaR (Cornish-Fisher adjusted): {cvar_modified:.4f} ({cvar_modified*100:.2f}%)")
    
    # CVaR Ratio
    cvar_ratio = calculate_cvar_ratio(returns, risk_free_rate=0.02)
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    
    print(f"\nRisk-Adjusted Returns:")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  CVaR Ratio:   {cvar_ratio:.3f}")
    
    # Comparación con estrategia sin tail risk
    print("\n" + "="*60)
    print(" COMPARISON: Strategy with vs without tail risk")
    print("="*60)
    
    # Estrategia sin tail risk
    safe_returns = np.random.normal(0.0008, 0.015, 210)  # Menor mean, menor vol
    
    safe_var = calculate_var(safe_returns, 0.95)
    safe_cvar = calculate_cvar(safe_returns, 0.95)
    safe_sharpe = (np.mean(safe_returns) / np.std(safe_returns)) * np.sqrt(252)
    safe_cvar_ratio = calculate_cvar_ratio(safe_returns, 0.02)
    
    print(f"\nRisky Strategy (con tail events):")
    print(f"  Mean Return: {np.mean(returns)*252:.2%}")
    print(f"  Sharpe:      {sharpe:.3f}")
    print(f"  CVaR (95%):  {cvar_95*100:.2f}%")
    print(f"  CVaR Ratio:  {cvar_ratio:.3f}")
    
    print(f"\nSafe Strategy (sin tail events):")
    print(f"  Mean Return: {np.mean(safe_returns)*252:.2%}")
    print(f"  Sharpe:      {safe_sharpe:.3f}")
    print(f"  CVaR (95%):  {safe_cvar*100:.2f}%")
    print(f"  CVaR Ratio:  {safe_cvar_ratio:.3f}")
    
    print(f"\n💡 INSIGHT:")
    print(f"   La estrategia 'risky' tiene mejor Sharpe ({sharpe:.2f} vs {safe_sharpe:.2f})")
    print(f"   PERO peor CVaR ({cvar_95*100:.2f}% vs {safe_cvar*100:.2f}%)")
    print(f"   → Para Prop Firms, 'safe' es mejor (menor tail risk)")
    
    print("="*60)


if __name__ == "__main__":
    example_cvar_analysis()
