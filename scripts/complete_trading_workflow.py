"""
Complete Multi-Strategy Trading Workflow
End-to-end demonstration of UNDERDOG system integration.

Workflow:
1. Data → Feature Engineering (hash-based versioning)
2. Features → ML Training (MLflow experiment tracking)
3. Regime Detection → Strategy Gating (HMM-based)
4. Signal Generation → Risk Management (multi-layer)
5. Position Sizing → Order Execution (MT5)
6. Validation → WFO + Monte Carlo
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Feature Engineering
from underdog.strategies.ml_strategies.feature_engineering import (
    FeatureEngineer, FeatureConfig
)

# ML Training
from underdog.ml.training.train_pipeline import (
    MLTrainingPipeline, TrainingConfig
)

# Regime Detection
from underdog.ml.models.regime_classifier import (
    HMMRegimeClassifier, RegimeConfig, RegimeType
)

# Risk Management
from underdog.risk_management.risk_master import RiskMaster, DrawdownLimits, ExposureLimits
from underdog.risk_management.position_sizing import PositionSizer, SizingConfig

# Strategy Coordination
from underdog.strategies.strategy_matrix import StrategyMatrix, StrategyConfig
from underdog.strategies.fuzzy_logic.mamdani_inference import ConfidenceScorer

# Validation
from underdog.backtesting.validation.wfo import WalkForwardOptimizer, WFOConfig
from underdog.backtesting.validation.monte_carlo import MonteCarloSimulator, MonteCarloConfig


def create_synthetic_market_data(n_bars=2000, seed=42):
    """Create synthetic multi-regime market data"""
    np.random.seed(seed)
    
    # Generate regime-switching data
    dates = pd.date_range(start='2020-01-01', periods=n_bars, freq='D')
    
    prices = []
    current_price = 100.0
    
    # Define regimes: (drift, volatility, duration)
    regimes = [
        (0.0005, 0.01, 400),   # Bull: positive drift, low vol
        (-0.0003, 0.015, 300),  # Bear: negative drift, medium vol
        (0.0, 0.005, 400),      # Sideways: no drift, very low vol
        (0.0001, 0.025, 300),   # High vol: small drift, high vol
        (0.0002, 0.008, 600)    # Low vol bull: positive drift, low vol
    ]
    
    regime_idx = 0
    regime_counter = 0
    
    for i in range(n_bars):
        # Switch regime
        if regime_counter >= regimes[regime_idx][2]:
            regime_idx = (regime_idx + 1) % len(regimes)
            regime_counter = 0
        
        drift, vol, _ = regimes[regime_idx]
        
        # Generate return
        ret = np.random.normal(drift, vol)
        current_price *= (1 + ret)
        
        # OHLC
        high = current_price * (1 + abs(np.random.normal(0, vol)))
        low = current_price * (1 - abs(np.random.normal(0, vol)))
        open_price = low + (high - low) * np.random.random()
        volume = np.random.randint(1000000, 10000000)
        
        prices.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
        
        regime_counter += 1
    
    df = pd.DataFrame(prices, index=dates)
    return df


def step1_feature_engineering(data: pd.DataFrame):
    """Step 1: Feature Engineering with Versioning"""
    print("\n" + "="*80)
    print("STEP 1: FEATURE ENGINEERING")
    print("="*80)
    
    config = FeatureConfig(
        sma_periods=[20, 50, 200],
        ema_periods=[12, 26],
        momentum_periods=[5, 10, 20],
        lag_periods=[1, 2, 3, 5],
        prediction_horizon=1,
        target_type="classification",
        random_seed=42
    )
    
    engineer = FeatureEngineer(config)
    features = engineer.transform(data)
    
    print(f"\n✓ Features Generated: {len(engineer.feature_names)} features")
    print(f"✓ Samples: {len(features)}")
    print(f"✓ Version: {engineer.metadata.config_hash}_{engineer.metadata.data_hash}")
    
    return features, engineer


def step2_regime_detection(data: pd.DataFrame):
    """Step 2: HMM Regime Detection"""
    print("\n" + "="*80)
    print("STEP 2: REGIME DETECTION (HMM)")
    print("="*80)
    
    config = RegimeConfig(
        n_states=4,
        train_window=200,
        min_regime_duration=5
    )
    
    classifier = HMMRegimeClassifier(config)
    classifier.fit(data['close'])
    
    # Predict regimes
    regimes = classifier.predict(data['close'])
    
    # Current state
    current_state = classifier.predict_current(data['close'])
    
    print(f"\n✓ Current Regime: {current_state.regime.value.upper()}")
    print(f"✓ Confidence: {current_state.confidence:.2%}")
    print(f"✓ Duration: {current_state.duration} bars")
    
    print("\n✓ Regime Distribution:")
    print(regimes['regime'].value_counts())
    
    return classifier, regimes


def step3_ml_training(features: pd.DataFrame):
    """Step 3: ML Model Training with MLflow"""
    print("\n" + "="*80)
    print("STEP 3: ML MODEL TRAINING (MLflow)")
    print("="*80)
    
    # Prepare data
    feature_cols = [col for col in features.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
    
    X = features[feature_cols].values
    y = features['target'].values
    
    # Train/Val/Test split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Configure training
    config = TrainingConfig(
        model_name="underdog_multiregime",
        model_type="rf",  # Random Forest for speed
        epochs=10,
        experiment_name="underdog_complete_workflow"
    )
    
    # Train
    pipeline = MLTrainingPipeline(config)
    results = pipeline.train(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        run_name="multi_regime_classifier"
    )
    
    print(f"\n✓ Model Trained: {config.model_name}")
    print(f"✓ Val Loss: {results['val_metrics']['val_loss']:.4f}")
    if 'test_metrics' in results:
        print(f"✓ Test Loss: {results['test_metrics']['test_loss']:.4f}")
    
    return pipeline, results


def step4_strategy_gating(classifier: HMMRegimeClassifier):
    """Step 4: Strategy Gating Based on Regime"""
    print("\n" + "="*80)
    print("STEP 4: STRATEGY GATING (Regime-Based)")
    print("="*80)
    
    strategies = ['trend', 'mean_reversion', 'breakout', 'ml_predictive']
    
    print("\nStrategy Activation Status:")
    for strategy in strategies:
        is_active = classifier.get_strategy_gate(strategy)
        status = "✓ ACTIVE" if is_active else "✗ INACTIVE"
        print(f"  {strategy.upper():20s}: {status}")
    
    return strategies


def step5_risk_management(data: pd.DataFrame):
    """Step 5: Multi-Layer Risk Management"""
    print("\n" + "="*80)
    print("STEP 5: RISK MANAGEMENT")
    print("="*80)
    
    # Risk Master
    dd_limits = DrawdownLimits(
        daily_max=0.02,
        weekly_max=0.05,
        monthly_max=0.10
    )
    
    exposure_limits = ExposureLimits(
        max_position_pct=0.05,
        max_total_exposure=0.20,
        max_correlated_exposure=0.10
    )
    
    risk_master = RiskMaster(
        initial_capital=100000.0,
        dd_limits=dd_limits,
        exposure_limits=exposure_limits
    )
    
    # Position Sizer
    sizing_config = SizingConfig(
        base_risk_pct=0.01,
        kelly_fraction=0.25,
        use_confidence=True,
        max_leverage=2.0
    )
    
    position_sizer = PositionSizer(sizing_config)
    
    # Simulate trade check
    sample_trade = {
        'symbol': 'EURUSD',
        'side': 'long',
        'entry_price': data['close'].iloc[-1],
        'stop_loss': data['close'].iloc[-1] * 0.98,
        'confidence': 0.75
    }
    
    # Pre-trade validation
    is_allowed = risk_master.pre_trade_check(
        symbol=sample_trade['symbol'],
        direction=sample_trade['side'],
        notional_value=10000.0
    )
    
    if is_allowed:
        # Calculate position size
        size = position_sizer.calculate_size(
            account_balance=risk_master.current_capital,
            entry_price=sample_trade['entry_price'],
            stop_loss=sample_trade['stop_loss'],
            confidence=sample_trade['confidence']
        )
        
        print(f"\n✓ Trade Approved")
        print(f"  Symbol: {sample_trade['symbol']}")
        print(f"  Entry: {sample_trade['entry_price']:.5f}")
        print(f"  Stop Loss: {sample_trade['stop_loss']:.5f}")
        print(f"  Position Size: {size['position_size']:.2f} units")
        print(f"  Risk Amount: ${size['risk_amount']:.2f}")
        print(f"  Confidence: {sample_trade['confidence']:.2%}")
    else:
        print(f"\n✗ Trade Rejected by Risk Master")
    
    return risk_master, position_sizer


def step6_walk_forward_optimization(data: pd.DataFrame):
    """Step 6: Walk-Forward Optimization"""
    print("\n" + "="*80)
    print("STEP 6: WALK-FORWARD OPTIMIZATION")
    print("="*80)
    
    # Simple SMA crossover strategy for demo
    def sma_strategy(prices, short_period, long_period):
        sma_short = prices.rolling(window=short_period).mean()
        sma_long = prices.rolling(window=long_period).mean()
        
        signals = pd.Series(0, index=prices.index)
        signals[sma_short > sma_long] = 1
        signals[sma_short < sma_long] = -1
        
        # Calculate returns
        returns = prices.pct_change()
        strategy_returns = signals.shift(1) * returns
        
        return strategy_returns.fillna(0)
    
    # Parameter grid
    param_grid = {
        'short_period': [10, 20, 30],
        'long_period': [50, 100, 150]
    }
    
    # WFO configuration
    config = WFOConfig(
        n_folds=3,
        train_ratio=0.7,
        anchored=False
    )
    
    optimizer = WalkForwardOptimizer(config)
    
    print("\n[Running WFO... this may take a moment]")
    results = optimizer.run(
        prices=data['close'],
        strategy_func=sma_strategy,
        param_grid=param_grid
    )
    
    print(f"\n✓ WFO Complete")
    print(f"  Total Folds: {len(results.fold_results)}")
    print(f"  Avg IS Sharpe: {results.summary['avg_is_sharpe']:.3f}")
    print(f"  Avg OOS Sharpe: {results.summary['avg_oos_sharpe']:.3f}")
    print(f"  Best Params: {results.best_params}")
    
    return results


def step7_monte_carlo_validation(data: pd.DataFrame):
    """Step 7: Monte Carlo Validation"""
    print("\n" + "="*80)
    print("STEP 7: MONTE CARLO VALIDATION")
    print("="*80)
    
    # Generate sample trades
    returns = data['close'].pct_change().dropna()
    
    # Simulate 100 trades
    np.random.seed(42)
    trades = pd.DataFrame({
        'pnl': np.random.choice(returns.values, size=100, replace=True) * 10000,
        'entry_time': pd.date_range(start=data.index[0], periods=100, freq='D'),
        'exit_time': pd.date_range(start=data.index[0], periods=100, freq='D') + timedelta(days=1)
    })
    
    # Monte Carlo config
    config = MonteCarloConfig(
        n_simulations=1000,
        resample_with_replacement=True,
        add_slippage=True,
        slippage_mean=0.0001,
        slippage_std=0.0001
    )
    
    simulator = MonteCarloSimulator(config)
    
    print("\n[Running Monte Carlo... 1000 simulations]")
    results = simulator.run(trades)
    
    print(f"\n✓ Monte Carlo Complete")
    print(f"  Total Return - Mean: ${results.summary['total_return_mean']:.2f}")
    print(f"  Total Return - 5th %ile: ${results.summary['total_return_p5']:.2f}")
    print(f"  Total Return - 95th %ile: ${results.summary['total_return_p95']:.2f}")
    print(f"  Max Drawdown - Mean: {results.summary['max_dd_mean']:.2%}")
    print(f"  VaR (5%): ${results.summary['var_5']:.2f}")
    print(f"  CVaR (5%): ${results.summary['cvar_5']:.2f}")
    
    return results


def main():
    """Complete workflow execution"""
    print("\n" + "="*80)
    print(" UNDERDOG - COMPLETE MULTI-STRATEGY TRADING WORKFLOW")
    print("="*80)
    print("\nThis workflow demonstrates:")
    print("  1. Feature Engineering with hash-based versioning")
    print("  2. HMM-based regime detection")
    print("  3. ML model training with MLflow")
    print("  4. Regime-based strategy gating")
    print("  5. Multi-layer risk management")
    print("  6. Walk-Forward Optimization")
    print("  7. Monte Carlo validation")
    
    # Generate data
    print("\n[Generating synthetic multi-regime market data...]")
    data = create_synthetic_market_data(n_bars=2000, seed=42)
    print(f"✓ Generated {len(data)} bars of OHLCV data")
    
    # Execute workflow
    features, engineer = step1_feature_engineering(data)
    classifier, regimes = step2_regime_detection(data)
    pipeline, ml_results = step3_ml_training(features)
    strategies = step4_strategy_gating(classifier)
    risk_master, position_sizer = step5_risk_management(data)
    wfo_results = step6_walk_forward_optimization(data)
    mc_results = step7_monte_carlo_validation(data)
    
    # Final summary
    print("\n" + "="*80)
    print(" WORKFLOW COMPLETE - SUMMARY")
    print("="*80)
    print("\n✓ All components integrated successfully")
    print("\nKey Metrics:")
    print(f"  Features: {len(engineer.feature_names)}")
    print(f"  Current Regime: {classifier.current_state.regime.value}")
    print(f"  Model Val Loss: {ml_results['val_metrics']['val_loss']:.4f}")
    print(f"  WFO Avg OOS Sharpe: {wfo_results.summary['avg_oos_sharpe']:.3f}")
    print(f"  Monte Carlo VaR (5%): ${mc_results.summary['var_5']:.2f}")
    print(f"  Risk Master Capital: ${risk_master.current_capital:,.2f}")
    
    print("\n" + "="*80)
    print(" Next Steps:")
    print("="*80)
    print("  1. Connect to MT5 using mt5_connector.py")
    print("  2. Deploy to Cloud/VPS for 24/7 operation")
    print("  3. Enable Prometheus monitoring")
    print("  4. Run unit tests (tests/)")
    print("  5. Review logs and adjust parameters")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
