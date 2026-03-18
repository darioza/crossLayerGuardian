"""
CrossLayerGuardian Ensemble Coordinator
Implements two-stage ensemble system combining XGBoost and MLP
Manages weight updates, confidence-based decisions, and final classification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import time
import logging
from pathlib import Path
import json
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .xgboost_classifier import XGBoostAnomalyClassifier
from .mlp_classifier import MLPAnomalyClassifier
from .feature_extractor import CrossLayerFeatureExtractor, CorrelatedEventGroup

logger = logging.getLogger(__name__)

@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results"""
    final_prediction: int
    confidence_score: float
    xgboost_prediction: int
    xgboost_probability: np.ndarray
    xgboost_confidence: float
    mlp_prediction: int
    mlp_probability: np.ndarray
    mlp_confidence: float
    ensemble_weights: Tuple[float, float]
    decision_method: str
    processing_time: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class EnsembleMetrics:
    """Container for ensemble performance metrics"""
    total_predictions: int = 0
    correct_predictions: int = 0
    xgboost_agreements: int = 0
    mlp_agreements: int = 0
    high_confidence_predictions: int = 0
    avg_processing_time: float = 0.0
    avg_confidence: float = 0.0
    weight_history: List[Tuple[float, float]] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)

class EnsembleCoordinator:
    """
    Two-stage ensemble coordinator for anomaly detection
    Combines XGBoost (first stage) and MLP (second stage) with adaptive weighting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Ensemble parameters from dissertation
        self.alpha = config.get('ensemble_alpha', 0.3)  # Weight update rate
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.agreement_threshold = config.get('agreement_threshold', 0.8)
        self.min_samples_for_update = config.get('min_samples_for_update', 100)
        
        # Initial weights (equal)
        self.xgboost_weight = 0.5
        self.mlp_weight = 0.5
        
        # Decision strategies
        self.decision_strategy = config.get('decision_strategy', 'confidence_weighted')
        self.fallback_strategy = config.get('fallback_strategy', 'conservative')
        
        # Initialize classifiers
        self.xgboost_classifier = XGBoostAnomalyClassifier(config)
        self.mlp_classifier = MLPAnomalyClassifier(config)
        self.feature_extractor = CrossLayerFeatureExtractor(config) if config.get('auto_extract_features', True) else None
        
        # Performance tracking
        self.metrics = EnsembleMetrics()
        self.prediction_history = deque(maxlen=config.get('history_size', 10000))
        self.recent_accuracies = deque(maxlen=config.get('accuracy_window', 1000))
        
        # Threading for parallel predictions
        self.max_workers = config.get('max_workers', 2)
        self.enable_parallel = config.get('enable_parallel_prediction', True)
        
        # Model state
        self.is_trained = False
        self.training_timestamp = None
        
        # Monitoring
        self.monitor_weights = config.get('monitor_weights', True)
        self.weight_adaptation_enabled = config.get('weight_adaptation', True)
        
        logger.info(f"EnsembleCoordinator initialized with α={self.alpha}, strategy={self.decision_strategy}")
    
    def train_ensemble(self, 
                      X: np.ndarray, 
                      y: np.ndarray,
                      train_xgboost: bool = True,
                      train_mlp: bool = True,
                      cross_validate: bool = True) -> Dict[str, Any]:
        """
        Train both ensemble components
        
        Args:
            X: Feature matrix (n_samples, 127) or raw events for feature extraction
            y: Labels
            train_xgboost: Whether to train XGBoost component
            train_mlp: Whether to train MLP component
            cross_validate: Whether to use cross-validation
            
        Returns:
            Combined training metrics
        """
        start_time = time.time()
        logger.info(f"Training ensemble with {X.shape[0] if isinstance(X, np.ndarray) else len(X)} samples")
        
        # Extract features if needed
        if self.feature_extractor and not isinstance(X, np.ndarray):
            logger.info("Extracting features from raw events")
            X = np.array([self.feature_extractor.extract_features(events) for events in X])
        
        # Validate dimensions
        if X.shape[1] != 127:
            raise ValueError(f"Expected 127 features, got {X.shape[1]}")
        
        training_metrics = {
            'training_start_time': start_time,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
        
        # Train XGBoost
        if train_xgboost:
            logger.info("Training XGBoost classifier...")
            xgb_metrics = self.xgboost_classifier.train(X, y, use_cross_validation=cross_validate)
            training_metrics['xgboost'] = xgb_metrics
            logger.info(f"XGBoost training completed: F1={xgb_metrics.get('f1_score', 0):.4f}")
        
        # Train MLP
        if train_mlp:
            logger.info("Training MLP classifier...")
            mlp_metrics = self.mlp_classifier.train(X, y, plot_history=False)
            training_metrics['mlp'] = mlp_metrics
            logger.info(f"MLP training completed: F1={mlp_metrics.get('f1_score', 0):.4f}")
        
        # Update ensemble state
        self.is_trained = (
            (train_xgboost and self.xgboost_classifier.model is not None) and
            (train_mlp and self.mlp_classifier.model is not None)
        )
        self.training_timestamp = time.time()
        
        # Calculate total training time
        training_metrics['total_training_time'] = time.time() - start_time
        training_metrics['ensemble_trained'] = self.is_trained
        
        # Reset ensemble metrics
        self.metrics = EnsembleMetrics()
        self.prediction_history.clear()
        self.recent_accuracies.clear()
        
        logger.info(f"Ensemble training completed in {training_metrics['total_training_time']:.2f}s")
        
        return training_metrics
    
    def predict(self, 
                X: Union[np.ndarray, List[CorrelatedEventGroup]], 
                return_detailed: bool = False) -> Union[np.ndarray, List[EnsemblePrediction]]:
        """
        Make ensemble predictions
        
        Args:
            X: Feature matrix or raw events
            return_detailed: Whether to return detailed prediction objects
            
        Returns:
            Predictions or detailed prediction objects
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train_ensemble() first.")
        
        start_time = time.time()
        
        # Extract features if needed
        if self.feature_extractor and not isinstance(X, np.ndarray):
            if isinstance(X, list) and len(X) > 0 and isinstance(X[0], CorrelatedEventGroup):
                X_features = np.array([self.feature_extractor.extract_features([event_group]) for event_group in X])
            else:
                X_features = np.array([self.feature_extractor.extract_features(events) for events in X])
        else:
            X_features = X
        
        # Validate dimensions
        if X_features.shape[1] != 127:
            raise ValueError(f"Expected 127 features, got {X_features.shape[1]}")
        
        # Make predictions based on strategy
        if self.enable_parallel and X_features.shape[0] > 1:
            predictions = self._predict_parallel(X_features)
        else:
            predictions = self._predict_sequential(X_features)
        
        # Update metrics
        processing_time = time.time() - start_time
        self.metrics.total_predictions += len(predictions)
        self.metrics.avg_processing_time = (
            (self.metrics.avg_processing_time * (self.metrics.total_predictions - len(predictions)) + 
             processing_time) / self.metrics.total_predictions
        )
        
        # Update prediction history
        for pred in predictions:
            self.prediction_history.append(pred)
        
        if return_detailed:
            return predictions
        else:
            return np.array([pred.final_prediction for pred in predictions])
    
    def _predict_sequential(self, X: np.ndarray) -> List[EnsemblePrediction]:
        """Sequential prediction processing"""
        predictions = []
        
        for i in range(X.shape[0]):
            sample = X[i:i+1]
            pred = self._predict_single(sample)
            predictions.append(pred)
        
        return predictions
    
    def _predict_parallel(self, X: np.ndarray) -> List[EnsemblePrediction]:
        """Parallel prediction processing"""
        predictions = [None] * X.shape[0]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit prediction tasks
            future_to_index = {
                executor.submit(self._predict_single, X[i:i+1]): i 
                for i in range(X.shape[0])
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    predictions[index] = future.result()
                except Exception as e:
                    logger.error(f"Prediction failed for sample {index}: {e}")
                    # Create fallback prediction
                    predictions[index] = EnsemblePrediction(
                        final_prediction=0,  # Conservative fallback
                        confidence_score=0.0,
                        xgboost_prediction=0,
                        xgboost_probability=np.array([1.0, 0.0]),
                        xgboost_confidence=0.0,
                        mlp_prediction=0,
                        mlp_probability=np.array([1.0, 0.0]),
                        mlp_confidence=0.0,
                        ensemble_weights=(self.xgboost_weight, self.mlp_weight),
                        decision_method='fallback',
                        processing_time=0.0
                    )
        
        return predictions
    
    def _predict_single(self, X_sample: np.ndarray) -> EnsemblePrediction:
        """Make prediction for a single sample"""
        start_time = time.time()
        
        # Get XGBoost predictions
        xgb_pred, xgb_proba = self.xgboost_classifier.predict(X_sample, return_probabilities=True)
        xgb_confidence = self.xgboost_classifier.get_confidence_score(X_sample)[0]
        
        # Get MLP predictions
        mlp_pred, mlp_proba = self.mlp_classifier.predict(X_sample, return_probabilities=True)
        mlp_confidence = self.mlp_classifier.get_confidence_score(X_sample)[0]
        
        # Make ensemble decision
        final_pred, confidence, decision_method = self._make_ensemble_decision(
            xgb_pred[0], xgb_proba[0], xgb_confidence,
            mlp_pred[0], mlp_proba[0], mlp_confidence
        )
        
        processing_time = time.time() - start_time
        
        prediction = EnsemblePrediction(
            final_prediction=final_pred,
            confidence_score=confidence,
            xgboost_prediction=xgb_pred[0],
            xgboost_probability=xgb_proba[0],
            xgboost_confidence=xgb_confidence,
            mlp_prediction=mlp_pred[0],
            mlp_probability=mlp_proba[0],
            mlp_confidence=mlp_confidence,
            ensemble_weights=(self.xgboost_weight, self.mlp_weight),
            decision_method=decision_method,
            processing_time=processing_time
        )
        
        return prediction
    
    def _make_ensemble_decision(self, 
                               xgb_pred: int, xgb_proba: np.ndarray, xgb_conf: float,
                               mlp_pred: int, mlp_proba: np.ndarray, mlp_conf: float) -> Tuple[int, float, str]:
        """
        Make final ensemble decision based on strategy
        
        Returns:
            (final_prediction, confidence_score, decision_method)
        """
        
        if self.decision_strategy == 'confidence_weighted':
            # Weight predictions by confidence and ensemble weights
            if xgb_pred == mlp_pred:
                # Agreement case
                final_pred = xgb_pred
                confidence = (xgb_conf * self.xgboost_weight + mlp_conf * self.mlp_weight)
                method = 'agreement'
            else:
                # Disagreement case - use confidence-weighted decision
                xgb_weighted_conf = xgb_conf * self.xgboost_weight
                mlp_weighted_conf = mlp_conf * self.mlp_weight
                
                if xgb_weighted_conf > mlp_weighted_conf:
                    final_pred = xgb_pred
                    confidence = xgb_weighted_conf
                    method = 'xgboost_confident'
                else:
                    final_pred = mlp_pred
                    confidence = mlp_weighted_conf
                    method = 'mlp_confident'
        
        elif self.decision_strategy == 'majority_vote':
            # Simple majority vote (tie-breaking by confidence)
            if xgb_pred == mlp_pred:
                final_pred = xgb_pred
                confidence = max(xgb_conf, mlp_conf)
                method = 'majority_agreement'
            else:
                # Tie-breaking by highest confidence
                if xgb_conf > mlp_conf:
                    final_pred = xgb_pred
                    confidence = xgb_conf
                    method = 'majority_xgboost'
                else:
                    final_pred = mlp_pred
                    confidence = mlp_conf
                    method = 'majority_mlp'
        
        elif self.decision_strategy == 'weighted_average':
            # Weighted average of probabilities
            weighted_proba = (
                xgb_proba * self.xgboost_weight + 
                mlp_proba * self.mlp_weight
            )
            final_pred = np.argmax(weighted_proba)
            confidence = np.max(weighted_proba)
            method = 'weighted_average'
        
        elif self.decision_strategy == 'conservative':
            # Conservative approach - prefer normal class in disagreement
            if xgb_pred == mlp_pred:
                final_pred = xgb_pred
                confidence = min(xgb_conf, mlp_conf)
                method = 'conservative_agreement'
            else:
                # In disagreement, choose normal class (0) if present
                if 0 in [xgb_pred, mlp_pred]:
                    final_pred = 0
                    confidence = min(xgb_conf, mlp_conf)
                    method = 'conservative_normal'
                else:
                    # Both predict anomaly - use higher confidence
                    if xgb_conf > mlp_conf:
                        final_pred = xgb_pred
                        confidence = xgb_conf
                        method = 'conservative_xgboost'
                    else:
                        final_pred = mlp_pred
                        confidence = mlp_conf
                        method = 'conservative_mlp'
        
        else:
            # Fallback to confidence-weighted
            logger.warning(f"Unknown decision strategy: {self.decision_strategy}, using confidence_weighted")
            return self._make_ensemble_decision(xgb_pred, xgb_proba, xgb_conf, mlp_pred, mlp_proba, mlp_conf)
        
        return final_pred, confidence, method
    
    def update_weights(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Update ensemble weights based on recent performance
        Uses adaptive weighting with α parameter from dissertation
        
        Args:
            X: Recent samples for evaluation
            y_true: True labels
            
        Returns:
            Weight update metrics
        """
        if not self.weight_adaptation_enabled:
            return {'weights_updated': False}
        
        if len(y_true) < self.min_samples_for_update:
            logger.debug(f"Insufficient samples for weight update: {len(y_true)} < {self.min_samples_for_update}")
            return {'weights_updated': False}
        
        logger.info(f"Updating ensemble weights with {len(y_true)} samples")
        
        # Get individual classifier predictions
        xgb_predictions = self.xgboost_classifier.predict(X)
        mlp_predictions = self.mlp_classifier.predict(X)
        
        # Calculate individual accuracies
        xgb_accuracy = np.mean(xgb_predictions == y_true)
        mlp_accuracy = np.mean(mlp_predictions == y_true)
        
        # Store current weights for comparison
        old_xgb_weight = self.xgboost_weight
        old_mlp_weight = self.mlp_weight
        
        # Adaptive weight update using exponential moving average
        # Higher accuracy gets higher weight
        total_accuracy = xgb_accuracy + mlp_accuracy
        if total_accuracy > 0:
            target_xgb_weight = xgb_accuracy / total_accuracy
            target_mlp_weight = mlp_accuracy / total_accuracy
            
            # Apply learning rate (alpha) for smooth updates
            self.xgboost_weight = (1 - self.alpha) * self.xgboost_weight + self.alpha * target_xgb_weight
            self.mlp_weight = (1 - self.alpha) * self.mlp_weight + self.alpha * target_mlp_weight
            
            # Normalize weights to sum to 1
            total_weight = self.xgboost_weight + self.mlp_weight
            self.xgboost_weight /= total_weight
            self.mlp_weight /= total_weight
        
        # Store weight history
        self.metrics.weight_history.append((self.xgboost_weight, self.mlp_weight))
        self.metrics.accuracy_history.append((xgb_accuracy + mlp_accuracy) / 2)
        
        # Calculate update metrics
        weight_change = abs(self.xgboost_weight - old_xgb_weight) + abs(self.mlp_weight - old_mlp_weight)
        
        update_metrics = {
            'weights_updated': True,
            'old_weights': (old_xgb_weight, old_mlp_weight),
            'new_weights': (self.xgboost_weight, self.mlp_weight),
            'weight_change': weight_change,
            'xgb_accuracy': xgb_accuracy,
            'mlp_accuracy': mlp_accuracy,
            'samples_used': len(y_true)
        }
        
        logger.info(f"Weights updated: XGB={self.xgboost_weight:.3f}, MLP={self.mlp_weight:.3f} "
                   f"(change: {weight_change:.3f})")
        
        return update_metrics
    
    def evaluate_ensemble(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive ensemble evaluation
        
        Args:
            X: Test features
            y_true: True labels
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(X, return_detailed=True)
        
        # Extract predictions
        final_predictions = np.array([pred.final_prediction for pred in predictions])
        xgb_predictions = np.array([pred.xgboost_prediction for pred in predictions])
        mlp_predictions = np.array([pred.mlp_prediction for pred in predictions])
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        
        ensemble_metrics = {
            'ensemble_accuracy': accuracy_score(y_true, final_predictions),
            'ensemble_precision': precision_score(y_true, final_predictions, average='weighted', zero_division=0),
            'ensemble_recall': recall_score(y_true, final_predictions, average='weighted', zero_division=0),
            'ensemble_f1': f1_score(y_true, final_predictions, average='weighted', zero_division=0),
            
            'xgboost_accuracy': accuracy_score(y_true, xgb_predictions),
            'xgboost_f1': f1_score(y_true, xgb_predictions, average='weighted', zero_division=0),
            
            'mlp_accuracy': accuracy_score(y_true, mlp_predictions),
            'mlp_f1': f1_score(y_true, mlp_predictions, average='weighted', zero_division=0),
            
            'agreement_rate': np.mean(xgb_predictions == mlp_predictions),
            'avg_confidence': np.mean([pred.confidence_score for pred in predictions]),
            'avg_processing_time': np.mean([pred.processing_time for pred in predictions]),
            
            'current_weights': (self.xgboost_weight, self.mlp_weight),
            'n_samples': len(y_true)
        }
        
        # Decision method distribution
        method_counts = defaultdict(int)
        for pred in predictions:
            method_counts[pred.decision_method] += 1
        
        ensemble_metrics['decision_methods'] = dict(method_counts)
        
        # Confidence distribution
        confidences = [pred.confidence_score for pred in predictions]
        ensemble_metrics['confidence_stats'] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'high_confidence_ratio': np.mean(np.array(confidences) > self.confidence_threshold)
        }
        
        logger.info(f"Ensemble evaluation: Accuracy={ensemble_metrics['ensemble_accuracy']:.4f}, "
                   f"F1={ensemble_metrics['ensemble_f1']:.4f}, Agreement={ensemble_metrics['agreement_rate']:.4f}")
        
        return ensemble_metrics
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current ensemble status and metrics"""
        status = {
            'is_trained': self.is_trained,
            'training_timestamp': self.training_timestamp,
            'current_weights': (self.xgboost_weight, self.mlp_weight),
            'decision_strategy': self.decision_strategy,
            'weight_adaptation_enabled': self.weight_adaptation_enabled,
            'alpha': self.alpha,
            'confidence_threshold': self.confidence_threshold,
            'total_predictions': self.metrics.total_predictions,
            'avg_processing_time': self.metrics.avg_processing_time,
            'prediction_history_size': len(self.prediction_history)
        }
        
        if self.metrics.weight_history:
            status['weight_evolution'] = self.metrics.weight_history[-10:]  # Last 10 updates
        
        if self.metrics.accuracy_history:
            status['recent_accuracy'] = np.mean(self.metrics.accuracy_history[-100:])  # Last 100 updates
        
        return status
    
    def save_ensemble(self, base_filename: str) -> Dict[str, str]:
        """
        Save entire ensemble to disk
        
        Args:
            base_filename: Base filename for saving
            
        Returns:
            Paths to saved components
        """
        paths = {}
        
        # Save individual classifiers
        xgb_path = self.xgboost_classifier.save_model(f"{base_filename}_xgboost.pkl")
        mlp_path = self.mlp_classifier.save_model(f"{base_filename}_mlp")
        
        paths['xgboost'] = xgb_path
        paths['mlp'] = mlp_path
        
        # Save ensemble metadata
        ensemble_data = {
            'xgboost_weight': self.xgboost_weight,
            'mlp_weight': self.mlp_weight,
            'alpha': self.alpha,
            'decision_strategy': self.decision_strategy,
            'confidence_threshold': self.confidence_threshold,
            'training_timestamp': self.training_timestamp,
            'config': self.config,
            'metrics': {
                'total_predictions': self.metrics.total_predictions,
                'weight_history': self.metrics.weight_history,
                'accuracy_history': self.metrics.accuracy_history
            }
        }
        
        ensemble_path = Path(self.config.get('model_dir', 'models')) / f"{base_filename}_ensemble.json"
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_data, f, indent=2)
        
        paths['ensemble'] = str(ensemble_path)
        
        logger.info(f"Ensemble saved with components: {list(paths.keys())}")
        
        return paths
    
    def load_ensemble(self, base_filename: str) -> bool:
        """
        Load entire ensemble from disk
        
        Args:
            base_filename: Base filename for loading
            
        Returns:
            Success status
        """
        try:
            # Load individual classifiers
            xgb_path = f"{base_filename}_xgboost.pkl"
            mlp_path = f"{base_filename}_mlp.h5"
            
            xgb_success = self.xgboost_classifier.load_model(xgb_path)
            mlp_success = self.mlp_classifier.load_model(mlp_path)
            
            if not (xgb_success and mlp_success):
                logger.error("Failed to load individual classifiers")
                return False
            
            # Load ensemble metadata
            ensemble_path = Path(self.config.get('model_dir', 'models')) / f"{base_filename}_ensemble.json"
            
            if ensemble_path.exists():
                with open(ensemble_path, 'r') as f:
                    ensemble_data = json.load(f)
                
                self.xgboost_weight = ensemble_data['xgboost_weight']
                self.mlp_weight = ensemble_data['mlp_weight']
                self.alpha = ensemble_data.get('alpha', self.alpha)
                self.decision_strategy = ensemble_data.get('decision_strategy', self.decision_strategy)
                self.confidence_threshold = ensemble_data.get('confidence_threshold', self.confidence_threshold)
                self.training_timestamp = ensemble_data.get('training_timestamp')
                
                # Restore metrics
                if 'metrics' in ensemble_data:
                    metrics_data = ensemble_data['metrics']
                    self.metrics.total_predictions = metrics_data.get('total_predictions', 0)
                    self.metrics.weight_history = metrics_data.get('weight_history', [])
                    self.metrics.accuracy_history = metrics_data.get('accuracy_history', [])
                
                self.is_trained = True
                
                logger.info(f"Ensemble loaded successfully with weights: XGB={self.xgboost_weight:.3f}, MLP={self.mlp_weight:.3f}")
                
                return True
            else:
                logger.warning(f"Ensemble metadata not found at {ensemble_path}")
                self.is_trained = True  # Models loaded successfully
                return True
                
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            return False

if __name__ == "__main__":
    # Test ensemble coordinator
    config = {
        'ensemble_alpha': 0.3,
        'confidence_threshold': 0.7,
        'decision_strategy': 'confidence_weighted',
        'enable_parallel_prediction': True,
        'max_workers': 2,
        'model_dir': 'test_models',
        'random_state': 42
    }
    
    coordinator = EnsembleCoordinator(config)
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(1000, 127)
    y_train = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
    
    X_test = np.random.randn(200, 127)
    y_test = np.random.choice([0, 1], size=200, p=[0.8, 0.2])
    
    # Train ensemble
    training_metrics = coordinator.train_ensemble(X_train, y_train)
    print(f"Training metrics: {training_metrics}")
    
    # Make predictions
    predictions = coordinator.predict(X_test, return_detailed=True)
    simple_predictions = coordinator.predict(X_test, return_detailed=False)
    
    print(f"Made {len(predictions)} predictions")
    print(f"Sample detailed prediction: {predictions[0]}")
    print(f"Prediction distribution: {np.bincount(simple_predictions)}")
    
    # Evaluate ensemble
    evaluation = coordinator.evaluate_ensemble(X_test, y_test)
    print(f"Evaluation metrics: {evaluation}")
    
    # Update weights
    weight_update = coordinator.update_weights(X_test[:100], y_test[:100])
    print(f"Weight update: {weight_update}")
    
    # Get status
    status = coordinator.get_ensemble_status()
    print(f"Ensemble status: {status}")