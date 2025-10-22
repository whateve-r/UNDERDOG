"""
Regime-Switching Classification for Market States

Implements Gaussian Mixture Model (GMM) + XGBoost classifier for
identifying market regimes: Trend / Range / Transition

Paper: arXiv:2510.03236v1 - Regime-Switching Methods for Volatility Prediction

Features:
- ATR (Average True Range) - volatility proxy
- Normalized Volume
- ADX (Average Directional Index) - trend strength
- Bollinger Band Width - volatility bands
- RSI - momentum

Regimes:
- TREND: High ADX (>25), moderate volatility
- RANGE: Low ADX (<20), low volatility, RSI oscillating
- TRANSITION: High volatility, erratic price action
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RegimeType = Literal['trend', 'range', 'transition']


@dataclass
class RegimeConfig:
    """Configuration for regime classification"""
    
    n_regimes: int = 3  # Trend, Range, Transition
    gmm_covariance_type: str = 'full'  # 'full', 'tied', 'diag', 'spherical'
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 3
    
    # Feature windows
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    rsi_period: int = 14
    volume_ma_period: int = 20


class RegimeSwitchingModel:
    """
    Market Regime Classifier using GMM + XGBoost
    
    Training Flow:
    1. Extract features from historical data
    2. GMM clustering (unsupervised) → soft labels
    3. Manual labeling of clusters (trend/range/transition)
    4. Train XGBoost classifier with soft labels
    
    Inference Flow:
    1. Extract features from current bar
    2. XGBoost predicts regime
    3. Return regime + confidence score
    
    Usage:
        # Training
        model = RegimeSwitchingModel()
        model.fit(historical_data)
        model.save('models/regime_classifier.pkl')
        
        # Inference
        model = RegimeSwitchingModel.load('models/regime_classifier.pkl')
        regime, confidence = model.predict(current_features)
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        
        self.gmm: Optional[GaussianMixture] = None
        self.xgb: Optional[XGBClassifier] = None
        self.scaler = StandardScaler()
        
        self.regime_labels: Dict[int, RegimeType] = {}  # Cluster ID → regime name
        self.is_fitted = False
        
        logger.info(f"RegimeSwitchingModel initialized with {self.config.n_regimes} regimes")
    
    def fit(self, df: pd.DataFrame, regime_column: Optional[str] = None):
        """
        Train regime classifier.
        
        Args:
            df: Historical OHLCV data with indicators
            regime_column: Optional manual labels for supervised training
        
        Expected columns:
        - open, high, low, close, volume
        - atr, adx, bb_width, rsi (or will be computed)
        """
        logger.info(f"Training regime classifier on {len(df)} bars")
        
        # Extract features
        features = self._extract_features(df)
        
        # Drop NaN rows (from indicator warm-up)
        features = features.dropna()
        logger.info(f"Features extracted: {len(features)} valid bars")
        
        # Normalize features
        X = self.scaler.fit_transform(features)
        
        if regime_column and regime_column in df.columns:
            # Supervised learning with provided labels
            y = df.loc[features.index, regime_column]
            self._train_supervised(X, y)
        else:
            # Unsupervised learning with GMM
            self._train_unsupervised(X, df.loc[features.index])
        
        self.is_fitted = True
        logger.info("✓ Regime classifier trained successfully")
    
    def _train_unsupervised(self, X: np.ndarray, df: pd.DataFrame):
        """Train using GMM clustering + manual labeling"""
        
        # Step 1: GMM clustering
        logger.info("Fitting GMM for soft clustering...")
        self.gmm = GaussianMixture(
            n_components=self.config.n_regimes,
            covariance_type=self.config.gmm_covariance_type,
            random_state=42
        )
        cluster_labels = self.gmm.fit_predict(X)
        
        # Step 2: Analyze clusters and assign regime labels
        self._label_clusters(cluster_labels, df)
        
        # Step 3: Train XGBoost with GMM soft labels
        logger.info("Training XGBoost classifier...")
        self.xgb = XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            random_state=42
        )
        
        # Map cluster labels to regime indices
        y = np.array([self._regime_to_index(self.regime_labels[c]) for c in cluster_labels])
        self.xgb.fit(X, y)
        
        # Log feature importance
        self._log_feature_importance()
    
    def _train_supervised(self, X: np.ndarray, y: pd.Series):
        """Train using provided regime labels"""
        
        logger.info("Training XGBoost with supervised labels...")
        
        # Convert regime names to indices
        y_indices = y.map(self._regime_to_index)
        
        self.xgb = XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            random_state=42
        )
        self.xgb.fit(X, y_indices)
        
        self._log_feature_importance()
    
    def _label_clusters(self, cluster_labels: np.ndarray, df: pd.DataFrame):
        """
        Assign regime names to GMM clusters based on characteristics.
        
        Heuristics:
        - TREND: High ADX (>25), medium volatility
        - RANGE: Low ADX (<20), low volatility
        - TRANSITION: High volatility, low ADX
        """
        logger.info("Analyzing clusters for regime labeling...")
        
        for cluster_id in range(self.config.n_regimes):
            mask = cluster_labels == cluster_id
            cluster_data = df[mask]
            
            # Compute cluster statistics
            avg_adx = cluster_data['adx'].mean() if 'adx' in cluster_data.columns else 20
            avg_atr = cluster_data['atr'].mean() if 'atr' in cluster_data.columns else 0
            avg_bb_width = cluster_data['bb_width'].mean() if 'bb_width' in cluster_data.columns else 0
            
            # Labeling logic
            if avg_adx > 25:
                regime = 'trend'
            elif avg_adx < 20 and avg_atr < df['atr'].median():
                regime = 'range'
            else:
                regime = 'transition'
            
            self.regime_labels[cluster_id] = regime
            
            logger.info(
                f"Cluster {cluster_id} → {regime.upper()} "
                f"(ADX: {avg_adx:.1f}, ATR: {avg_atr:.5f}, BB: {avg_bb_width:.5f})"
            )
    
    def predict(
        self,
        features: pd.DataFrame
    ) -> tuple[RegimeType, float]:
        """
        Predict regime for current market state.
        
        Args:
            features: Single row DataFrame with indicator values
        
        Returns:
            (regime, confidence)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Normalize features
        X = self.scaler.transform(features)
        
        # XGBoost prediction
        y_pred = self.xgb.predict(X)[0]
        y_proba = self.xgb.predict_proba(X)[0]
        
        regime = self._index_to_regime(y_pred)
        confidence = y_proba[y_pred]
        
        return regime, confidence
    
    def predict_batch(self, features: pd.DataFrame) -> pd.Series:
        """Predict regime for multiple bars"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.scaler.transform(features)
        y_pred = self.xgb.predict(X)
        
        return pd.Series(
            [self._index_to_regime(idx) for idx in y_pred],
            index=features.index
        )
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract regime classification features.
        
        Returns:
            DataFrame with columns: atr, adx, bb_width, rsi, volume_ratio
        """
        features = pd.DataFrame(index=df.index)
        
        # ATR (volatility)
        if 'atr' not in df.columns:
            features['atr'] = self._compute_atr(df, self.config.atr_period)
        else:
            features['atr'] = df['atr']
        
        # ADX (trend strength)
        if 'adx' not in df.columns:
            features['adx'] = self._compute_adx(df, self.config.adx_period)
        else:
            features['adx'] = df['adx']
        
        # Bollinger Band Width (volatility)
        if 'bb_width' not in df.columns:
            features['bb_width'] = self._compute_bb_width(df, self.config.bb_period)
        else:
            features['bb_width'] = df['bb_width']
        
        # RSI (momentum)
        if 'rsi' not in df.columns:
            features['rsi'] = self._compute_rsi(df, self.config.rsi_period)
        else:
            features['rsi'] = df['rsi']
        
        # Volume ratio
        volume_ma = df['volume'].rolling(self.config.volume_ma_period).mean()
        features['volume_ratio'] = df['volume'] / volume_ma
        
        return features
    
    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _compute_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average Directional Index"""
        # Simplified ADX (full implementation would use +DI/-DI)
        high = df['high']
        low = df['low']
        
        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)
        
        atr = self._compute_atr(df, period)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _compute_bb_width(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Bollinger Band Width"""
        close = df['close']
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        bb_width = (upper - lower) / sma
        
        return bb_width
    
    def _compute_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Relative Strength Index"""
        delta = df['close'].diff()
        
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _regime_to_index(self, regime: RegimeType) -> int:
        """Map regime name to index"""
        mapping = {'trend': 0, 'range': 1, 'transition': 2}
        return mapping[regime]
    
    def _index_to_regime(self, index: int) -> RegimeType:
        """Map index to regime name"""
        mapping = {0: 'trend', 1: 'range', 2: 'transition'}
        return mapping[index]
    
    def _log_feature_importance(self):
        """Log XGBoost feature importance"""
        if self.xgb is None:
            return
        
        feature_names = ['atr', 'adx', 'bb_width', 'rsi', 'volume_ratio']
        importances = self.xgb.feature_importances_
        
        logger.info("Feature Importances:")
        for name, importance in zip(feature_names, importances):
            logger.info(f"  {name:15s}: {importance:.4f}")
    
    def save(self, path: str):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'config': self.config,
            'gmm': self.gmm,
            'xgb': self.xgb,
            'scaler': self.scaler,
            'regime_labels': self.regime_labels,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'RegimeSwitchingModel':
        """Load model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(model_data['config'])
        model.gmm = model_data['gmm']
        model.xgb = model_data['xgb']
        model.scaler = model_data['scaler']
        model.regime_labels = model_data['regime_labels']
        model.is_fitted = model_data['is_fitted']
        
        logger.info(f"✓ Model loaded from {path}")
        return model


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_bars = 10000
    
    df = pd.DataFrame({
        'open': np.random.randn(n_bars).cumsum() + 1.1000,
        'high': np.random.randn(n_bars).cumsum() + 1.1010,
        'low': np.random.randn(n_bars).cumsum() + 1.0990,
        'close': np.random.randn(n_bars).cumsum() + 1.1000,
        'volume': np.random.randint(1000, 10000, n_bars)
    })
    
    # Train model
    model = RegimeSwitchingModel()
    model.fit(df)
    
    # Test prediction
    test_features = model._extract_features(df.tail(1))
    regime, confidence = model.predict(test_features)
    
    print(f"Predicted Regime: {regime} (confidence: {confidence:.2f})")
    
    # Save model
    model.save('models/regime_classifier_test.pkl')
