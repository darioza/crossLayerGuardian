"""
CrossLayerGuardian MLP Neural Network
Implements Multi-Layer Perceptron for deep anomaly detection
Second stage of the two-stage ensemble system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class MLPAnomalyClassifier:
    """
    Multi-Layer Perceptron for anomaly detection in cross-layer correlation events
    Second stage of the two-stage ensemble system
    Architecture: [127, 256, 128, 64, 2] with ReLU activation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # MLP architecture parameters from dissertation
        self.input_dim = 127
        self.hidden_layers = config.get('mlp_hidden_layers', [256, 128, 64])
        self.output_dim = 2  # Binary classification
        self.activation = config.get('mlp_activation', 'relu')
        self.dropout_rate = config.get('mlp_dropout_rate', 0.3)
        self.l2_regularization = config.get('mlp_l2_reg', 0.001)
        
        # Training parameters
        self.learning_rate = config.get('mlp_learning_rate', 0.001)
        self.batch_size = config.get('mlp_batch_size', 32)
        self.epochs = config.get('mlp_epochs', 100)
        self.patience = config.get('mlp_patience', 15)
        self.validation_split = config.get('mlp_validation_split', 0.2)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.training_history = None
        self.callbacks_list = []
        
        # Performance tracking
        self.last_training_time = None
        self.prediction_times = []
        self.model_version = "1.0.0"
        
        # Model persistence
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize random seeds for reproducibility
        self.random_seed = config.get('random_state', 42)
        self._set_random_seeds()
        
        logger.info(f"MLPAnomalyClassifier initialized with architecture: {[self.input_dim] + self.hidden_layers + [self.output_dim]}")
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
    
    def _build_model(self) -> keras.Model:
        """
        Build MLP model with specified architecture
        Architecture: [127, 256, 128, 64, 2]
        """
        model = keras.Sequential([
            # Input layer
            layers.Dense(
                self.hidden_layers[0], 
                input_dim=self.input_dim,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_regularization),
                name='hidden_layer_1'
            ),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Hidden layer 2
            layers.Dense(
                self.hidden_layers[1],
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_regularization),
                name='hidden_layer_2'
            ),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Hidden layer 3
            layers.Dense(
                self.hidden_layers[2],
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_regularization),
                name='hidden_layer_3'
            ),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            
            # Output layer
            layers.Dense(
                self.output_dim,
                activation='softmax',
                name='output_layer'
            )
        ])
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _setup_callbacks(self, model_path: str) -> List[callbacks.Callback]:
        """Setup training callbacks"""
        callback_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=0
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger for training history
            callbacks.CSVLogger(
                str(self.model_dir / 'training_log.csv'),
                append=True
            )
        ]
        
        return callback_list
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              save_model: bool = True,
              plot_history: bool = False) -> Dict[str, Any]:
        """
        Train MLP classifier
        
        Args:
            X: Feature matrix (n_samples, 127)
            y: Labels (0=normal, 1=anomaly)
            validation_data: Optional validation data tuple
            save_model: Whether to save trained model
            plot_history: Whether to plot training history
            
        Returns:
            Training metrics and history
        """
        start_time = time.time()
        logger.info(f"Starting MLP training with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Validate input dimensions
        if X.shape[1] != 127:
            raise ValueError(f"Expected 127 features, got {X.shape[1]}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data if validation data not provided
        if validation_data is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_encoded,
                test_size=self.validation_split,
                random_state=self.random_seed,
                stratify=y_encoded
            )
        else:
            X_train, y_train = X_scaled, y_encoded
            X_val, y_val = validation_data
            X_val = self.scaler.transform(X_val)
            y_val = self.label_encoder.transform(y_val)
        
        # Build model
        self.model = self._build_model()
        
        # Setup callbacks
        model_path = self.model_dir / f"mlp_model_checkpoint_{int(time.time())}.h5"
        self.callbacks_list = self._setup_callbacks(str(model_path))
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=self.callbacks_list,
            verbose=1,
            shuffle=True
        )
        
        # Store training history
        self.training_history = history.history
        
        # Generate predictions for validation set
        y_pred_proba = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba[:, 1]) if len(np.unique(y_val)) > 1 else 0.0,
            'training_time': time.time() - start_time,
            'epochs_trained': len(history.history['loss']),
            'best_val_loss': min(history.history['val_loss']),
            'best_val_accuracy': max(history.history['val_accuracy'])
        }
        
        # Add training history metrics
        metrics['final_train_loss'] = history.history['loss'][-1]
        metrics['final_train_accuracy'] = history.history['accuracy'][-1]
        metrics['final_val_loss'] = history.history['val_loss'][-1]
        metrics['final_val_accuracy'] = history.history['val_accuracy'][-1]
        
        self.last_training_time = time.time()
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        # Plot training history if requested
        if plot_history:
            self._plot_training_history()
        
        logger.info(f"MLP training completed in {metrics['training_time']:.2f}s")
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
        probabilities = self.model.predict(X_scaled, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
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
        
        return self.model.predict(X_scaled, verbose=0)
    
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
    
    def get_layer_activations(self, X: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Get activations from a specific layer
        
        Args:
            X: Input data
            layer_name: Name of the layer
            
        Returns:
            Layer activations
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create a model that outputs the desired layer
        layer_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        return layer_model.predict(X_scaled, verbose=0)
    
    def analyze_feature_importance(self, X: np.ndarray, method: str = 'permutation') -> Dict[int, float]:
        """
        Analyze feature importance using permutation importance
        
        Args:
            X: Feature matrix
            method: Method for importance calculation
            
        Returns:
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get baseline predictions
        baseline_proba = self.model.predict(X_scaled, verbose=0)
        baseline_loss = keras.losses.sparse_categorical_crossentropy(
            np.argmax(baseline_proba, axis=1), baseline_proba
        )
        baseline_score = np.mean(baseline_loss)
        
        importance_scores = {}
        
        if method == 'permutation':
            # Permutation importance
            for feature_idx in range(X_scaled.shape[1]):
                # Create permuted data
                X_permuted = X_scaled.copy()
                np.random.shuffle(X_permuted[:, feature_idx])
                
                # Get predictions with permuted feature
                permuted_proba = self.model.predict(X_permuted, verbose=0)
                permuted_loss = keras.losses.sparse_categorical_crossentropy(
                    np.argmax(permuted_proba, axis=1), permuted_proba
                )
                permuted_score = np.mean(permuted_loss)
                
                # Importance is the increase in loss
                importance_scores[feature_idx] = permuted_score - baseline_score
        
        return importance_scores
    
    def _plot_training_history(self):
        """Plot training history"""
        if self.training_history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.training_history['loss'], label='Training Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.training_history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in self.training_history:
            axes[1, 0].plot(self.training_history['precision'], label='Training Precision')
            axes[1, 0].plot(self.training_history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Recall
        if 'recall' in self.training_history:
            axes[1, 1].plot(self.training_history['recall'], label='Training Recall')
            axes[1, 1].plot(self.training_history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = self.model_dir / 'training_history.png'
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Training history plot saved to {plot_path}")
    
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
            filename = f"mlp_model_v{self.model_version}_{timestamp}"
        
        # Save Keras model
        model_path = self.model_dir / f"{filename}.h5"
        self.model.save(model_path)
        
        # Save additional components
        components_path = self.model_dir / f"{filename}_components.pkl"
        components_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_version': self.model_version,
            'training_history': self.training_history,
            'config': self.config
        }
        
        joblib.dump(components_data, components_path)
        
        # Save metadata
        metadata = {
            'model_path': str(model_path),
            'components_path': str(components_path),
            'model_version': self.model_version,
            'save_timestamp': time.time(),
            'architecture': [self.input_dim] + self.hidden_layers + [self.output_dim],
            'config': self.config,
            'last_training_time': self.last_training_time
        }
        
        metadata_path = self.model_dir / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from disk
        
        Args:
            model_path: Path to saved model (.h5 file)
            
        Returns:
            Success status
        """
        try:
            # Load Keras model
            self.model = keras.models.load_model(model_path)
            
            # Load additional components
            components_path = model_path.replace('.h5', '_components.pkl')
            if Path(components_path).exists():
                components_data = joblib.load(components_path)
                
                self.scaler = components_data['scaler']
                self.label_encoder = components_data['label_encoder']
                self.model_version = components_data.get('model_version', '1.0.0')
                self.training_history = components_data.get('training_history')
                self.config.update(components_data.get('config', {}))
            
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
            'architecture': [self.input_dim] + self.hidden_layers + [self.output_dim],
            'total_parameters': self.model.count_params() if self.model else 0
        }
        
        if self.training_history:
            metrics.update({
                'epochs_trained': len(self.training_history['loss']),
                'best_val_loss': min(self.training_history['val_loss']),
                'best_val_accuracy': max(self.training_history['val_accuracy']),
                'final_train_loss': self.training_history['loss'][-1],
                'final_val_loss': self.training_history['val_loss'][-1]
            })
        
        if self.prediction_times:
            metrics.update({
                'avg_prediction_time': np.mean(self.prediction_times),
                'prediction_count': len(self.prediction_times)
            })
        
        return metrics
    
    def fine_tune(self, 
                  X: np.ndarray, 
                  y: np.ndarray,
                  learning_rate: float = 0.0001,
                  epochs: int = 10) -> Dict[str, Any]:
        """
        Fine-tune existing model with new data
        
        Args:
            X: New feature data
            y: New labels
            learning_rate: Lower learning rate for fine-tuning
            epochs: Number of fine-tuning epochs
            
        Returns:
            Fine-tuning metrics
        """
        if self.model is None:
            raise ValueError("No model to fine-tune. Train model first.")
        
        logger.info(f"Fine-tuning model with {X.shape[0]} samples")
        
        # Update learning rate
        self.model.optimizer.learning_rate = learning_rate
        
        # Prepare data
        y_encoded = self.label_encoder.transform(y)
        X_scaled = self.scaler.transform(X)
        
        # Fine-tune
        history = self.model.fit(
            X_scaled, y_encoded,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        # Calculate metrics
        metrics = {
            'fine_tuning_epochs': epochs,
            'fine_tuning_lr': learning_rate,
            'final_loss': history.history['loss'][-1],
            'final_accuracy': history.history['accuracy'][-1]
        }
        
        logger.info(f"Fine-tuning completed: Loss={metrics['final_loss']:.4f}, Accuracy={metrics['final_accuracy']:.4f}")
        
        return metrics
    
    def reset_model(self):
        """Reset model to untrained state"""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.training_history = None
        self.callbacks_list = []
        self.prediction_times = []
        self.last_training_time = None
        logger.info("Model reset to untrained state")

if __name__ == "__main__":
    # Test MLP classifier
    config = {
        'mlp_hidden_layers': [256, 128, 64],
        'mlp_learning_rate': 0.001,
        'mlp_batch_size': 32,
        'mlp_epochs': 50,
        'mlp_patience': 10,
        'model_dir': 'test_models',
        'random_state': 42
    }
    
    classifier = MLPAnomalyClassifier(config)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 127)  # 127 features
    y = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])  # 80% normal, 20% anomaly
    
    # Train model
    metrics = classifier.train(X, y, plot_history=False)
    print(f"Training metrics: {metrics}")
    
    # Test predictions
    X_test = np.random.randn(100, 127)
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    confidence = classifier.get_confidence_score(X_test)
    
    print(f"Test predictions: {np.bincount(predictions)}")
    print(f"Average confidence: {np.mean(confidence):.4f}")
    
    # Performance metrics
    perf_metrics = classifier.get_performance_metrics()
    print(f"Performance metrics: {perf_metrics}")
    
    # Feature importance (sample)
    importance = classifier.analyze_feature_importance(X_test[:50])  # Small sample for demo
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top 5 important features: {top_features}")