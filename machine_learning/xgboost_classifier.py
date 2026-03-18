"""
CrossLayerGuardian XGBoost Classifier
Implements gradient boosting for anomaly detection in cross-layer correlated events
Based on dissertation specifications for first-stage classification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class XGBoostAnomalyClassifier:
    """
    XGBoost-based anomaly detection for cross-layer correlation events
    First stage of the two-stage ensemble system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # XGBoost hyperparameters from dissertation
        self.xgb_params = {
            'max_depth': config.get('xgb_max_depth', 6),
            'learning_rate': config.get('xgb_learning_rate', 0.1),
            'n_estimators': config.get('xgb_n_estimators', 50),
            'subsample': config.get('xgb_subsample', 0.8),
            'colsample_bytree': config.get('xgb_colsample_bytree', 0.8),
            'reg_alpha': config.get('xgb_reg_alpha', 0.1),
            'reg_lambda': config.get('xgb_reg_lambda', 1.0),
            'random_state': config.get('random_state', 42),
            'n_jobs': config.get('n_jobs', -1),
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.training_history = []
        
        # Performance tracking
        self.last_training_time = None
        self.prediction_times = []
        self.model_version = "1.0.0"
        
        # Model persistence
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info(f"XGBoostAnomalyClassifier initialized with params: {self.xgb_params}")
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              validation_split: float = 0.2,
              use_cross_validation: bool = True,
              save_model: bool = True) -> Dict[str, float]:
        """
        Train XGBoost classifier with cross-validation
        
        Args:
            X: Feature matrix (n_samples, 127)
            y: Labels (0=normal, 1=anomaly)
            validation_split: Fraction for validation
            use_cross_validation: Whether to use cross-validation
            save_model: Whether to save trained model
            
        Returns:
            Training metrics dictionary
        """
        start_time = time.time()
        logger.info(f"Starting XGBoost training with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Validate input dimensions
        if X.shape[1] != 127:
            raise ValueError(f"Expected 127 features, got {X.shape[1]}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded, 
            test_size=validation_split, 
            random_state=self.xgb_params['random_state'],
            stratify=y_encoded
        )
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.xgb_params)
        
        # Training with early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Cross-validation if requested
        cv_scores = None
        if use_cross_validation:
            cv_scores = cross_val_score(
                self.model, X_scaled, y_encoded, 
                cv=5, scoring='f1', n_jobs=self.xgb_params['n_jobs']
            )
            logger.info(f"Cross-validation F1 scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Generate predictions for validation set
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.0,
            'training_time': time.time() - start_time,
            'best_iteration': getattr(self.model, 'best_iteration', self.xgb_params['n_estimators'])
        }
        
        if cv_scores is not None:
            metrics['cv_f1_mean'] = cv_scores.mean()
            metrics['cv_f1_std'] = cv_scores.std()
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Update training history
        self.training_history.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        })
        
        self.last_training_time = time.time()
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        logger.info(f"XGBoost training completed in {metrics['training_time']:.2f}s")
        logger.info(f"Validation metrics - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray, return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix (n_samples, 127)
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Predictions and optionally probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        start_time = time.time()
        
        # Validate input dimensions
        if X.shape[1] != 127:
            raise ValueError(f"Expected 127 features, got {X.shape[1]}")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode predictions
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        
        # Track prediction time
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        # Keep only recent prediction times (last 1000)
        if len(self.prediction_times) > 1000:
            self.prediction_times = self.prediction_times[-1000:]
        
        if return_probabilities:
            return predictions_decoded, probabilities
        else:
            return predictions_decoded
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix (n_samples, 127)
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Validate input dimensions
        if X.shape[1] != 127:
            raise ValueError(f"Expected 127 features, got {X.shape[1]}")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)
    
    def get_confidence_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get confidence scores for predictions
        Higher values indicate higher confidence
        
        Args:
            X: Feature matrix (n_samples, 127)
            
        Returns:
            Confidence scores (0-1)
        """
        probabilities = self.predict_proba(X)
        
        # Confidence is the maximum probability across classes
        confidence_scores = np.max(probabilities, axis=1)
        
        return confidence_scores
    
    def get_feature_importance(self, top_k: Optional[int] = None) -> Dict[int, float]:
        """
        Get feature importance scores
        
        Args:
            top_k: Return only top k important features
            
        Returns:
            Dictionary mapping feature indices to importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained. Feature importance not available.")
        
        importance_dict = {i: importance for i, importance in enumerate(self.feature_importance)}
        
        if top_k is not None:
            # Sort by importance and take top k
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            importance_dict = dict(sorted_features[:top_k])
        
        return importance_dict
    
    def hyperparameter_tuning(self, 
                            X: np.ndarray, 
                            y: np.ndarray,
                            param_grid: Optional[Dict[str, List]] = None,
                            cv_folds: int = 5,
                            scoring_metric: str = 'f1_weighted') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X: Feature matrix
            y: Labels
            param_grid: Parameter grid for search
            cv_folds: Number of cross-validation folds
            scoring_metric: Scoring metric for optimization
            
        Returns:
            Best parameters and tuning results
        """
        logger.info("Starting hyperparameter tuning...")
        
        if param_grid is None:
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [50, 100, 150],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        
        # Encode labels and scale features
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize base model
        base_model = xgb.XGBClassifier(
            random_state=self.xgb_params['random_state'],
            n_jobs=self.xgb_params['n_jobs'],
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring=scoring_metric,
            n_jobs=self.xgb_params['n_jobs'],
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_scaled, y_encoded)
        tuning_time = time.time() - start_time
        
        # Update model parameters with best found
        self.xgb_params.update(grid_search.best_params_)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'tuning_time': tuning_time,
            'total_fits': len(grid_search.cv_results_['mean_test_score'])
        }
        
        logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f}s")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")
        
        return results
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save trained model to disk
        
        Args:
            filename: Custom filename (optional)
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"xgboost_model_v{self.model_version}_{timestamp}.pkl"
        
        model_path = self.model_dir / filename
        
        # Save model components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'xgb_params': self.xgb_params,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, model_path)
        
        # Save metadata
        metadata = {
            'model_path': str(model_path),
            'model_version': self.model_version,
            'save_timestamp': time.time(),
            'xgb_params': self.xgb_params,
            'last_training_time': self.last_training_time
        }
        
        metadata_path = self.model_dir / f"{filename.replace('.pkl', '_metadata.json')}"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from disk
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Success status
        """
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.xgb_params = model_data['xgb_params']
            self.feature_importance = model_data.get('feature_importance')
            self.model_version = model_data.get('model_version', '1.0.0')
            self.training_history = model_data.get('training_history', [])
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Performance metrics dictionary
        """
        metrics = {
            'model_version': self.model_version,
            'is_trained': self.model is not None,
            'last_training_time': self.last_training_time,
            'training_history_length': len(self.training_history),
            'xgb_params': self.xgb_params
        }
        
        if self.training_history:
            latest_training = self.training_history[-1]
            metrics.update({
                'latest_accuracy': latest_training['metrics'].get('accuracy', 0),
                'latest_f1_score': latest_training['metrics'].get('f1_score', 0),
                'latest_roc_auc': latest_training['metrics'].get('roc_auc', 0),
                'latest_training_time': latest_training['metrics'].get('training_time', 0)
            })
        
        if self.prediction_times:
            metrics.update({
                'avg_prediction_time': np.mean(self.prediction_times),
                'prediction_count': len(self.prediction_times)
            })
        
        if self.feature_importance is not None:
            metrics['feature_importance_available'] = True
            metrics['top_5_features'] = dict(list(self.get_feature_importance(top_k=5).items()))
        
        return metrics
    
    def update_model_incremental(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, float]:
        """
        Update model incrementally with new data
        Note: XGBoost doesn't support true incremental learning,
        so this retrains with combined data
        
        Args:
            X_new: New feature data
            y_new: New labels
            
        Returns:
            Updated training metrics
        """
        if self.model is None:
            logger.warning("No existing model found. Performing full training.")
            return self.train(X_new, y_new)
        
        logger.info(f"Performing incremental update with {X_new.shape[0]} new samples")
        
        # For now, we retrain with new data
        # In production, consider using online learning algorithms
        return self.train(X_new, y_new)
    
    def reset_model(self):
        """Reset model to untrained state"""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.training_history = []
        self.prediction_times = []
        self.last_training_time = None
        logger.info("Model reset to untrained state")

if __name__ == "__main__":
    # Test XGBoost classifier
    config = {
        'xgb_max_depth': 6,
        'xgb_learning_rate': 0.1,
        'xgb_n_estimators': 50,
        'xgb_subsample': 0.8,
        'model_dir': 'test_models',
        'random_state': 42
    }
    
    classifier = XGBoostAnomalyClassifier(config)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 127)  # 127 features
    y = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])  # 80% normal, 20% anomaly
    
    # Train model
    metrics = classifier.train(X, y)
    print(f"Training metrics: {metrics}")
    
    # Test predictions
    X_test = np.random.randn(100, 127)
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    confidence = classifier.get_confidence_score(X_test)
    
    print(f"Test predictions: {np.bincount(predictions)}")
    print(f"Average confidence: {np.mean(confidence):.4f}")
    
    # Feature importance
    importance = classifier.get_feature_importance(top_k=10)
    print(f"Top 10 important features: {importance}")
    
    # Performance metrics
    perf_metrics = classifier.get_performance_metrics()
    print(f"Performance metrics: {perf_metrics}")