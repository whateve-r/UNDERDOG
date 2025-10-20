"""
Pytest Configuration and Shared Fixtures
Provides reusable test fixtures for the entire test suite.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    
    n_samples = 500
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    close = 100.0
    prices = []
    
    for _ in range(n_samples):
        ret = np.random.normal(0.0005, 0.02)
        close *= (1 + ret)
        
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        volume = np.random.randint(1000000, 10000000)
        
        prices.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(prices, index=dates)
    return df


@pytest.fixture
def sample_price_series(sample_ohlcv_data):
    """Generate sample price series from OHLCV data"""
    return sample_ohlcv_data['close']


@pytest.fixture
def sample_tick_data():
    """Generate sample tick data for ZeroMQ message testing"""
    return {
        'symbol': 'EURUSD',
        'bid': 1.09850,
        'ask': 1.09853,
        'time': datetime.now().isoformat(),
        'volume': 100
    }


@pytest.fixture
def sample_trade_data():
    """Generate sample trade data for backtesting"""
    np.random.seed(42)
    
    trades = []
    entry_time = datetime(2020, 1, 1)
    
    for i in range(100):
        pnl = np.random.normal(100, 500)
        duration = np.random.randint(1, 10)
        
        trades.append({
            'trade_id': i,
            'pnl': pnl,
            'entry_time': entry_time,
            'exit_time': entry_time + timedelta(days=duration),
            'symbol': 'EURUSD',
            'side': 'long' if i % 2 == 0 else 'short',
            'entry_price': 1.1000 + np.random.normal(0, 0.001),
            'exit_price': 1.1000 + np.random.normal(0, 0.002)
        })
        
        entry_time += timedelta(days=duration + 1)
    
    return pd.DataFrame(trades)


@pytest.fixture
def risk_master_config():
    """Configuration for Risk Master testing"""
    from underdog.risk_management.risk_master import DrawdownLimits, ExposureLimits
    
    return {
        'initial_capital': 100000.0,
        'dd_limits': DrawdownLimits(
            daily_max=0.02,
            weekly_max=0.05,
            monthly_max=0.10
        ),
        'exposure_limits': ExposureLimits(
            max_position_pct=0.05,
            max_total_exposure=0.20,
            max_correlated_exposure=0.10
        )
    }


@pytest.fixture
def position_sizer_config():
    """Configuration for Position Sizer testing"""
    from underdog.risk_management.position_sizing import SizingConfig
    
    return SizingConfig(
        base_risk_pct=0.01,
        kelly_fraction=0.25,
        use_confidence=True,
        max_leverage=2.0
    )


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for file I/O tests"""
    return tmp_path
