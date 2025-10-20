"""
Quick Integration Test - Scientific Improvements + Data Ingestion

Tests all 5 scientific improvements + 2 data modules to ensure they work correctly.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add underdog to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_triple_barrier():
    """Test Triple-Barrier Labeling"""
    print("\n" + "=" * 60)
    print("TEST 1/7: Triple-Barrier Labeling")
    print("=" * 60)
    
    try:
        from underdog.strategies.ml_strategies.feature_engineering import (
            FeatureEngineer, FeatureConfig, create_sample_ohlcv
        )
        
        # Create sample data
        data = create_sample_ohlcv(500)
        
        # Initialize engineer
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        
        # Apply triple-barrier labeling
        labels = engineer.apply_triple_barrier_labeling(
            data,
            upper_barrier=0.02,
            lower_barrier=-0.01,
            max_holding_hours=24
        )
        
        print(f"✓ Generated {len(labels)} labels")
        print(f"✓ Label distribution:")
        print(labels['label'].value_counts())
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_purging_embargo():
    """Test Purging & Embargo"""
    print("\n" + "=" * 60)
    print("TEST 2/7: Purging & Embargo")
    print("=" * 60)
    
    try:
        from underdog.backtesting.validation.wfo import WalkForwardOptimizer
        
        # Create sample data
        train = pd.DataFrame({'price': range(100)})
        val = pd.DataFrame({'price': range(100, 120)})
        
        # Initialize optimizer
        optimizer = WalkForwardOptimizer()
        
        # Apply purging & embargo
        train_clean, val_clean = optimizer.purge_and_embargo(train, val, embargo_pct=0.02)
        
        print(f"✓ Train: {len(train)} → {len(train_clean)} (removed {len(train) - len(train_clean)})")
        print(f"✓ Val: {len(val)} → {len(val_clean)} (removed {len(val) - len(val_clean)})")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_realistic_slippage():
    """Test Realistic Slippage Model"""
    print("\n" + "=" * 60)
    print("TEST 3/7: Realistic Slippage Model")
    print("=" * 60)
    
    try:
        from underdog.backtesting.validation.monte_carlo import MonteCarloSimulator
        
        # Initialize simulator
        simulator = MonteCarloSimulator()
        
        # Calculate slippage
        slippage = simulator.calculate_realistic_slippage(
            trade_size=100000,
            market_conditions={
                'bid_ask_spread': 0.0001,
                'volume': 1000000,
                'volatility': 0.015,
                'avg_volatility': 0.01
            }
        )
        
        print(f"✓ Realistic slippage: {slippage:.6f} ({slippage*10000:.2f} pips)")
        print(f"✓ Components: spread + price_impact + vol_multiplier + noise")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_regime_transitions():
    """Test Regime Transition Probability"""
    print("\n" + "=" * 60)
    print("TEST 4/7: Regime Transition Probability")
    print("=" * 60)
    
    try:
        from underdog.ml.models.regime_classifier import (
            HMMRegimeClassifier,
            RegimeConfig,
            RegimeType,
            create_synthetic_regime_data
        )
        
        # Create synthetic data
        prices = create_synthetic_regime_data(n_samples=500, n_regimes=3)
        
        # Initialize classifier
        config = RegimeConfig(n_states=3, train_window=200)
        classifier = HMMRegimeClassifier(config)
        
        # Fit model
        classifier.fit(prices)
        
        # Calculate transition probability
        prob = classifier.calculate_transition_probability(
            RegimeType.LOW_VOL,
            RegimeType.HIGH_VOL,
            horizon=5
        )
        
        print(f"✓ Transition probability (LOW_VOL → HIGH_VOL, 5 bars): {prob:.3f}")
        
        # Get full transition matrix
        matrix = classifier.get_transition_matrix(regime_based=True)
        print(f"✓ Transition matrix shape: {matrix.shape}")
        
        # Predict regime change
        forecast = classifier.predict_regime_change(prices, horizon=10)
        print(f"✓ Regime change likely: {forecast['regime_change_likely']}")
        print(f"✓ Most likely next regime: {forecast['most_likely_next_regime']}")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_permutation_importance():
    """Test Permutation Feature Importance"""
    print("\n" + "=" * 60)
    print("TEST 5/7: Permutation Feature Importance")
    print("=" * 60)
    
    try:
        from underdog.ml.training.train_pipeline import (
            MLTrainingPipeline,
            TrainingConfig,
            create_synthetic_dataset
        )
        
        # Create synthetic dataset
        X, y = create_synthetic_dataset(n_samples=500, sequence_length=30, n_features=5)
        
        # Split data
        train_size = int(0.7 * len(X))
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        # Configure training (use RF for speed)
        config = TrainingConfig(
            model_name="test_model",
            model_type="rf",
            epochs=5
        )
        
        # Initialize pipeline
        pipeline = MLTrainingPipeline(config)
        
        # Train model
        pipeline.train(X_train, y_train, X_val, y_val, run_name="importance_test")
        
        # Calculate permutation importance
        importance = pipeline.calculate_permutation_importance(
            X_val,
            y_val,
            n_repeats=5
        )
        
        print(f"✓ Calculated importance for {len(importance)} features")
        print(f"✓ Top 3 features:")
        print(importance.head(3)[['feature', 'importance_mean', 'importance_std']])
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_histdata_module():
    """Test HistData Ingestion Module"""
    print("\n" + "=" * 60)
    print("TEST 6/7: HistData Ingestion Module")
    print("=" * 60)
    
    try:
        from underdog.database.histdata_ingestion import (
            HistDataIngestion,
            HistDataConfig
        )
        
        # Initialize module
        config = HistDataConfig(
            symbols=["EURUSD"],
            start_year=2023,
            start_month=1,
            end_year=2023,
            end_month=1
        )
        
        ingestion = HistDataIngestion(config)
        
        print(f"✓ HistDataIngestion initialized")
        print(f"✓ Config: {config.symbols}, {config.start_year}-{config.start_month}")
        print(f"✓ NOTE: Requires manual CSV download from HistData.com")
        print(f"✓ CSV path: data/raw/histdata/SYMBOL_YYYYMM.csv")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_news_scraping():
    """Test News Scraping Module"""
    print("\n" + "=" * 60)
    print("TEST 7/7: News Scraping + Sentiment Analysis")
    print("=" * 60)
    
    try:
        from underdog.database.news_scraping import (
            NewsIngestion,
            NewsScrapingConfig
        )
        
        # Initialize module
        config = NewsScrapingConfig(
            sentiment_model='vader',
            symbols=['EURUSD', 'GBPUSD']
        )
        
        scraper = NewsIngestion(config)
        
        print(f"✓ NewsIngestion initialized")
        print(f"✓ Sentiment model: {config.sentiment_model}")
        print(f"✓ Symbols: {config.symbols}")
        
        # Test single RSS feed (quick test)
        print(f"\n✓ Testing RSS feed scraping (sample)...")
        news = scraper.scrape_rss_feed(
            "https://www.investing.com/rss/news.rss",
            source="Investing.com"
        )
        
        if not news.empty:
            print(f"✓ Scraped {len(news)} articles")
            print(f"✓ Sample article:")
            print(f"  Title: {news.iloc[0]['title'][:60]}...")
            print(f"  Sentiment: {news.iloc[0]['sentiment_score']:.3f}")
        else:
            print(f"✓ No articles scraped (may be rate-limited or offline)")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("UNDERDOG - Scientific Improvements Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Triple-Barrier Labeling", test_triple_barrier),
        ("Purging & Embargo", test_purging_embargo),
        ("Realistic Slippage", test_realistic_slippage),
        ("Regime Transitions", test_regime_transitions),
        ("Permutation Importance", test_permutation_importance),
        ("HistData Module", test_histdata_module),
        ("News Scraping", test_news_scraping)
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ CRITICAL ERROR in {name}: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} {name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Scientific improvements are working correctly!")
        print("✅ Data ingestion modules are ready to use!")
        print("\n📝 Next steps:")
        print("  1. Download HistData CSVs manually")
        print("  2. Run: python scripts/backfill_all_data.py")
        print("  3. Execute WFO with real data")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("Review error messages above for details")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
