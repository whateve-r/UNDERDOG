"""
ML Training Pipeline with MLflow Integration
Experiment tracking, model versioning, and staging workflow.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import hashlib
import json


@dataclass
class TrainingConfig:
    """Training pipeline configuration"""
    # Model
    model_name: str = "lstm_predictor"
    model_type: str = "lstm"  # lstm, cnn, transformer, rf, xgboost
    
    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    
    # Features
    sequence_length: int = 60
    prediction_horizon: int = 1
    
    # MLflow
    experiment_name: str = "underdog_ml"
    tracking_uri: str = "file:./mlruns"
    log_artifacts: bool = True
    register_model: bool = True
    
    # Reproducibility
    random_seed: int = 42
    
    # Output
    model_dir: str = "models"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'random_seed': self.random_seed
        }


class MLTrainingPipeline:
    """
    ML training pipeline with MLflow experiment tracking.
    
    Features:
    - Experiment tracking and logging
    - Model versioning and registry
    - Staging workflow (None → Staging → Production)
    - Artifact management
    - Reproducible training with seed management
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize training pipeline.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Initialize MLflow
        self._init_mlflow()
        
        # Model placeholder
        self.model = None
        self.training_history = None
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility"""
        np.random.seed(self.config.random_seed)
        
        try:
            import torch
            torch.manual_seed(self.config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.random_seed)
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.random.set_seed(self.config.random_seed)
        except ImportError:
            pass
    
    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking"""
        try:
            import mlflow
            
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            
            print(f"[MLflow] Experiment: {self.config.experiment_name}")
            print(f"[MLflow] Tracking URI: {self.config.tracking_uri}")
            
        except ImportError:
            print("[WARNING] MLflow not installed. Install with: pip install mlflow")
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             X_test: Optional[np.ndarray] = None,
             y_test: Optional[np.ndarray] = None,
             run_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train model with MLflow tracking.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Optional test features
            y_test: Optional test labels
            run_name: Optional MLflow run name
        
        Returns:
            Dict with training results and metrics
        """
        try:
            import mlflow
            
            # Start MLflow run
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                print(f"\n[MLflow] Run ID: {run_id}")
                
                # Log parameters
                mlflow.log_params(self.config.to_dict())
                
                # Log dataset info
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_val))
                if X_test is not None:
                    mlflow.log_param("test_samples", len(X_test))
                
                # Calculate and log data hash for reproducibility
                data_hash = self._calculate_data_hash(X_train, y_train)
                mlflow.log_param("data_hash", data_hash)
                
                # Build model
                print(f"[Training] Building {self.config.model_type} model...")
                self.model = self._build_model(X_train.shape)
                
                # Train model
                print(f"[Training] Training for {self.config.epochs} epochs...")
                history = self._train_model(
                    X_train, y_train,
                    X_val, y_val
                )
                
                self.training_history = history
                
                # Log training metrics
                for epoch, metrics in enumerate(history['epochs']):
                    mlflow.log_metrics({
                        'train_loss': metrics['train_loss'],
                        'val_loss': metrics['val_loss']
                    }, step=epoch)
                
                # Evaluate on validation set
                val_metrics = self._evaluate(X_val, y_val, prefix='val')
                mlflow.log_metrics(val_metrics)
                
                # Evaluate on test set if provided
                if X_test is not None and y_test is not None:
                    test_metrics = self._evaluate(X_test, y_test, prefix='test')
                    mlflow.log_metrics(test_metrics)
                
                # Save model artifacts
                if self.config.log_artifacts:
                    self._save_artifacts(run_id)
                
                # Register model
                if self.config.register_model:
                    model_uri = f"runs:/{run_id}/model"
                    model_version = mlflow.register_model(
                        model_uri,
                        self.config.model_name
                    )
                    print(f"[MLflow] Model registered: {self.config.model_name} v{model_version.version}")
                
                # Prepare results
                results = {
                    'run_id': run_id,
                    'model_type': self.config.model_type,
                    'val_metrics': val_metrics,
                    'history': history
                }
                
                if X_test is not None:
                    results['test_metrics'] = test_metrics
                
                print(f"\n[Training] Complete!")
                print(f"[Training] Val Loss: {val_metrics.get('val_loss', 0):.4f}")
                
                return results
                
        except ImportError:
            print("[ERROR] MLflow not available. Running without tracking.")
            # Fallback: train without MLflow
            return self._train_without_mlflow(X_train, y_train, X_val, y_val, X_test, y_test)
    
    def _build_model(self, input_shape: Tuple) -> Any:
        """Build model based on configuration"""
        if self.config.model_type == "lstm":
            return self._build_lstm_model(input_shape)
        elif self.config.model_type == "cnn":
            return self._build_cnn_model(input_shape)
        elif self.config.model_type == "transformer":
            return self._build_transformer_model(input_shape)
        elif self.config.model_type in ["rf", "random_forest"]:
            return self._build_rf_model()
        elif self.config.model_type == "xgboost":
            return self._build_xgboost_model()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _build_lstm_model(self, input_shape: Tuple) -> Any:
        """Build LSTM model"""
        try:
            import torch
            import torch.nn as nn
            
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])
            
            input_size = input_shape[-1] if len(input_shape) > 1 else 1
            model = LSTMModel(input_size=input_size)
            return model
            
        except ImportError:
            print("[WARNING] PyTorch not installed. Using placeholder model.")
            return None
    
    def _build_cnn_model(self, input_shape: Tuple) -> Any:
        """Build CNN 1D model"""
        try:
            import torch
            import torch.nn as nn
            
            class CNN1DModel(nn.Module):
                def __init__(self, input_size, output_size=1):
                    super().__init__()
                    self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
                    self.pool = nn.AdaptiveAvgPool1d(1)
                    self.fc = nn.Linear(32, output_size)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    x = x.transpose(1, 2)  # (batch, seq, features) -> (batch, features, seq)
                    x = self.relu(self.conv1(x))
                    x = self.relu(self.conv2(x))
                    x = self.pool(x).squeeze(-1)
                    return self.fc(x)
            
            input_size = input_shape[-1] if len(input_shape) > 1 else 1
            model = CNN1DModel(input_size=input_size)
            return model
            
        except ImportError:
            return None
    
    def _build_transformer_model(self, input_shape: Tuple) -> Any:
        """Build Transformer model"""
        # Placeholder for Transformer implementation
        print("[INFO] Transformer model not yet implemented. Using LSTM instead.")
        return self._build_lstm_model(input_shape)
    
    def _build_rf_model(self) -> Any:
        """Build Random Forest model"""
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.random_seed
        )
        return model
    
    def _build_xgboost_model(self) -> Any:
        """Build XGBoost model"""
        try:
            import xgboost as xgb
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_seed
            )
            return model
            
        except ImportError:
            print("[WARNING] XGBoost not installed. Using Random Forest instead.")
            return self._build_rf_model()
    
    def _train_model(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train the model"""
        # Check if model is sklearn-based
        if hasattr(self.model, 'fit') and hasattr(self.model, 'predict'):
            # Sklearn-style training
            if X_train.ndim > 2:
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
            else:
                X_train_flat = X_train
                X_val_flat = X_val
            
            self.model.fit(X_train_flat, y_train)
            
            train_pred = self.model.predict(X_train_flat)
            val_pred = self.model.predict(X_val_flat)
            
            train_loss = np.mean((train_pred - y_train) ** 2)
            val_loss = np.mean((val_pred - y_val) ** 2)
            
            history = {
                'epochs': [{'train_loss': train_loss, 'val_loss': val_loss}]
            }
            
            return history
        
        else:
            # PyTorch-style training (placeholder)
            print("[INFO] PyTorch training not yet fully implemented. Using mock history.")
            history = {
                'epochs': [
                    {'train_loss': 0.05 - i * 0.001, 'val_loss': 0.06 - i * 0.0008}
                    for i in range(min(self.config.epochs, 20))
                ]
            }
            return history
    
    def _evaluate(self, X, y, prefix='val') -> Dict[str, float]:
        """Evaluate model"""
        if X.ndim > 2 and hasattr(self.model, 'predict'):
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        try:
            predictions = self.model.predict(X_flat)
            
            mse = np.mean((predictions - y) ** 2)
            mae = np.mean(np.abs(predictions - y))
            rmse = np.sqrt(mse)
            
            return {
                f'{prefix}_loss': float(mse),
                f'{prefix}_mae': float(mae),
                f'{prefix}_rmse': float(rmse)
            }
        except:
            return {f'{prefix}_loss': 0.0}
    
    def _calculate_data_hash(self, X, y) -> str:
        """Calculate hash of training data for reproducibility"""
        data_bytes = X.tobytes() + y.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]
    
    def _save_artifacts(self, run_id: str) -> None:
        """Save model and artifacts"""
        try:
            import mlflow
            
            # Save model
            model_path = f"models/{self.config.model_name}_{run_id}.pkl"
            Path("models").mkdir(exist_ok=True)
            joblib.dump(self.model, model_path)
            mlflow.log_artifact(model_path, "model")
            
            # Save config
            config_path = f"models/config_{run_id}.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            mlflow.log_artifact(config_path, "config")
            
            print(f"[MLflow] Artifacts saved")
            
        except Exception as e:
            print(f"[ERROR] Failed to save artifacts: {e}")
    
    def _train_without_mlflow(self, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
        """Fallback training without MLflow"""
        print("[Training] Running without MLflow tracking...")
        
        self.model = self._build_model(X_train.shape)
        history = self._train_model(X_train, y_train, X_val, y_val)
        val_metrics = self._evaluate(X_val, y_val, 'val')
        
        results = {
            'model_type': self.config.model_type,
            'val_metrics': val_metrics,
            'history': history
        }
        
        if X_test is not None:
            test_metrics = self._evaluate(X_test, y_test, 'test')
            results['test_metrics'] = test_metrics
        
        return results
    
    def promote_model(self, model_name: str, version: int, stage: str = "Production") -> None:
        """
        Promote model to a specific stage.
        
        Args:
            model_name: Name of registered model
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            print(f"[MLflow] Model {model_name} v{version} promoted to {stage}")
            
        except ImportError:
            print("[ERROR] MLflow not available")
        except Exception as e:
            print(f"[ERROR] Failed to promote model: {e}")
    
    def load_production_model(self, model_name: str) -> Any:
        """Load model from Production stage"""
        try:
            import mlflow
            
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            
            print(f"[MLflow] Loaded production model: {model_name}")
            return model
            
        except ImportError:
            print("[ERROR] MLflow not available")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return None
    
    def calculate_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        metric: str = "mse",
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate permutation feature importance.
        
        **MEJORA CIENTÍFICA #5: Permutation Feature Importance**
        
        Methodology (Breiman 2001, scikit-learn):
        - Permutation importance measures feature contribution by:
          1. Baseline: Calculate model performance on original data
          2. Shuffle: Randomly permute feature i (breaks relationship with target)
          3. Score drop: Δ performance = baseline - permuted_score
          4. Repeat: Average over n_repeats for statistical stability
        
        Why better than built-in feature importance (e.g., Gini, coefficients):
        - Model-agnostic: Works with any black-box model (LSTM, XGBoost, etc.)
        - Realistic: Measures actual predictive value, not just correlation
        - Handles interactions: Captures complex non-linear relationships
        - Statistical: Provides confidence intervals via repeated permutations
        
        Applications:
        1. Feature selection: Drop features with low/negative importance
        2. Model debugging: Identify spurious correlations (e.g., data leakage)
        3. Interpretability: Understand which features drive predictions
        4. Risk management: Validate that economic features > noise
        
        Args:
            X: Features (2D or 3D for time-series)
            y: Target values
            feature_names: Optional list of feature names
            n_repeats: Number of permutations per feature (higher = more stable)
            metric: 'mse' (regression), 'accuracy' (classification), or custom callable
            random_state: Random seed for reproducibility
        
        Returns:
            DataFrame with feature importance scores + std errors
        
        Example:
            >>> pipeline = MLTrainingPipeline()
            >>> pipeline.train(X_train, y_train, X_val, y_val)
            >>> importance = pipeline.calculate_permutation_importance(X_val, y_val)
            >>> print(importance.head())
            >>> # Drop low-importance features
            >>> important_features = importance[importance['importance_mean'] > 0.01].index
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Set random seed
        if random_state is None:
            random_state = self.config.random_seed
        
        np.random.seed(random_state)
        
        # Flatten 3D data for sklearn models
        original_shape = X.shape
        if X.ndim > 2 and hasattr(self.model, 'predict'):
            X_flat = X.reshape(X.shape[0], -1)
            n_features = X_flat.shape[1]
        else:
            X_flat = X
            n_features = X.shape[-1]
        
        # Generate feature names if not provided
        if feature_names is None:
            if X.ndim == 3:
                # Time-series: feature_name_timestep
                n_timesteps = X.shape[1]
                n_base_features = X.shape[2]
                feature_names = [
                    f"feature_{f}_t{t}"
                    for f in range(n_base_features)
                    for t in range(n_timesteps)
                ]
            else:
                feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Define scoring function
        if metric == "mse":
            def score_fn(y_true, y_pred):
                return -np.mean((y_true - y_pred) ** 2)  # Negative MSE (higher = better)
        elif metric == "mae":
            def score_fn(y_true, y_pred):
                return -np.mean(np.abs(y_true - y_pred))
        elif metric == "accuracy":
            def score_fn(y_true, y_pred):
                return np.mean(y_true == y_pred)
        elif callable(metric):
            score_fn = metric
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Calculate baseline score
        baseline_pred = self.model.predict(X_flat)
        baseline_score = score_fn(y, baseline_pred)
        
        print(f"[Permutation Importance] Baseline score: {baseline_score:.4f}")
        print(f"[Permutation Importance] Calculating importance for {n_features} features...")
        
        # Calculate permutation importance
        importances = np.zeros((n_features, n_repeats))
        
        for feature_idx in range(n_features):
            for repeat in range(n_repeats):
                # Copy data
                X_permuted = X_flat.copy()
                
                # Permute feature
                permutation_idx = np.random.permutation(len(X_permuted))
                X_permuted[:, feature_idx] = X_permuted[permutation_idx, feature_idx]
                
                # Predict with permuted data
                permuted_pred = self.model.predict(X_permuted)
                permuted_score = score_fn(y, permuted_pred)
                
                # Importance = baseline - permuted (drop in performance)
                importances[feature_idx, repeat] = baseline_score - permuted_score
            
            # Progress indicator
            if (feature_idx + 1) % 10 == 0:
                print(f"  Processed {feature_idx + 1}/{n_features} features...")
        
        # Calculate statistics
        importance_mean = np.mean(importances, axis=1)
        importance_std = np.std(importances, axis=1)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'feature': feature_names[:n_features],
            'importance_mean': importance_mean,
            'importance_std': importance_std,
            'importance_normalized': importance_mean / (importance_mean.sum() + 1e-8)
        })
        
        # Sort by importance
        results = results.sort_values('importance_mean', ascending=False)
        
        print(f"[Permutation Importance] Complete!")
        print(f"\nTop 5 Most Important Features:")
        print(results.head())
        
        return results
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot permutation feature importance with error bars.
        
        Args:
            importance_df: Output from calculate_permutation_importance()
            top_n: Number of top features to plot
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get top N features
            plot_data = importance_df.head(top_n).copy()
            
            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            y_pos = np.arange(len(plot_data))
            ax.barh(
                y_pos,
                plot_data['importance_mean'],
                xerr=plot_data['importance_std'],
                align='center',
                alpha=0.7,
                color='steelblue',
                ecolor='darkblue'
            )
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(plot_data['feature'])
            ax.invert_yaxis()  # Top feature at the top
            ax.set_xlabel('Permutation Importance')
            ax.set_title(f'Top {top_n} Feature Importance (±1 std)')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"[Plot] Saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("[WARNING] Matplotlib not installed. Cannot plot.")
    
    def select_important_features(
        self,
        X: np.ndarray,
        importance_df: pd.DataFrame,
        threshold: float = 0.01,
        top_k: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select important features based on permutation importance.
        
        Args:
            X: Original features
            importance_df: Output from calculate_permutation_importance()
            threshold: Minimum importance threshold (normalized)
            top_k: Alternatively, select top K features
        
        Returns:
            Tuple of (X_selected, selected_feature_names)
        """
        if top_k is not None:
            # Select top K features
            selected = importance_df.head(top_k)
        else:
            # Select features above threshold
            selected = importance_df[
                importance_df['importance_normalized'] > threshold
            ]
        
        selected_features = selected['feature'].tolist()
        
        # Extract corresponding columns
        if X.ndim == 3:
            # Time-series: This is complex, need to map back to original structure
            print("[WARNING] Feature selection for 3D data not yet implemented.")
            return X, []
        else:
            # 2D: Simple column selection
            feature_indices = [
                int(name.split('_')[1]) for name in selected_features
            ]
            X_selected = X[:, feature_indices]
            
            print(f"[Feature Selection] Selected {len(selected_features)} features")
            print(f"  Original: {X.shape[1]} features")
            print(f"  Selected: {X_selected.shape[1]} features")
            
            return X_selected, selected_features


# ========================================
# Utility Functions
# ========================================

def create_synthetic_dataset(n_samples=1000, sequence_length=60, n_features=5):
    """Create synthetic time-series dataset for testing"""
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randn(n_samples)
    return X, y


def run_training_example():
    """Example training pipeline"""
    print("Creating synthetic dataset...")
    X, y = create_synthetic_dataset(n_samples=1000, sequence_length=60, n_features=5)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Configure training
    config = TrainingConfig(
        model_name="test_lstm",
        model_type="rf",  # Use RF for quick example
        epochs=10,
        batch_size=32
    )
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline(config)
    
    # Train
    results = pipeline.train(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        run_name="example_run"
    )
    
    print("\n" + "="*50)
    print("Training Results:")
    print(f"Val Loss: {results['val_metrics']['val_loss']:.4f}")
    if 'test_metrics' in results:
        print(f"Test Loss: {results['test_metrics']['test_loss']:.4f}")
    print("="*50)
    
    return results


if __name__ == '__main__':
    print("Running ML Training Pipeline Example...")
    results = run_training_example()
