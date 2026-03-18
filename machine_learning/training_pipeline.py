"""
CrossLayerGuardian ML Training Pipeline
Comprehensive training system for the ensemble ML components
Handles data preprocessing, model training, validation, and persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import time
import logging
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

from .ensemble_coordinator import EnsembleCoordinator
from .feature_extractor import CrossLayerFeatureExtractor, CorrelatedEventGroup

logger = logging.getLogger(__name__)

@dataclass
class TrainingDataset:
    """Container for training dataset"""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingResults:
    """Container for training results"""
    ensemble_metrics: Dict[str, Any]
    xgboost_metrics: Dict[str, Any]
    mlp_metrics: Dict[str, Any]
    cross_validation_results: Dict[str, Any]
    training_time: float
    model_paths: Dict[str, str]
    dataset_info: Dict[str, Any]
    validation_report: str
    feature_importance: Dict[str, Any]

class MLTrainingPipeline:
    """
    Comprehensive ML training pipeline for CrossLayerGuardian
    Handles end-to-end training process from raw data to deployed models
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Training parameters
        self.test_size = config.get('test_size', 0.2)
        self.val_size = config.get('val_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.cross_validation_folds = config.get('cv_folds', 5)
        self.stratify = config.get('stratify', True)
        
        # Data preprocessing
        self.handle_imbalanced_data = config.get('handle_imbalanced', True)
        self.feature_selection = config.get('feature_selection', False)
        self.feature_selection_k = config.get('feature_selection_k', 100)
        
        # Training options
        self.enable_hyperparameter_tuning = config.get('hyperparameter_tuning', False)
        self.enable_cross_validation = config.get('cross_validation', True)
        self.save_models = config.get('save_models', True)
        self.generate_reports = config.get('generate_reports', True)
        
        # Paths
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.report_dir = Path(config.get('report_dir', 'reports'))
        
        # Create directories
        for dir_path in [self.data_dir, self.model_dir, self.report_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.ensemble_coordinator = EnsembleCoordinator(config)
        self.feature_extractor = CrossLayerFeatureExtractor(config)
        
        # Training state
        self.last_training_results = None
        self.training_history = []
        
        logger.info(f"MLTrainingPipeline initialized with test_size={self.test_size}, cv_folds={self.cross_validation_folds}")
    
    def prepare_dataset(self, 
                       data_source: Union[str, np.ndarray, List[CorrelatedEventGroup]], 
                       labels: Optional[np.ndarray] = None,
                       feature_extraction: bool = True) -> TrainingDataset:
        """
        Prepare dataset for training
        
        Args:
            data_source: Path to data file, feature matrix, or raw events
            labels: Labels array (if data_source is feature matrix)
            feature_extraction: Whether to perform feature extraction
            
        Returns:
            Prepared training dataset
        """
        logger.info("Preparing dataset for training...")
        
        # Load data based on source type
        if isinstance(data_source, str):
            # Load from file
            X, y = self._load_data_from_file(data_source)
        elif isinstance(data_source, np.ndarray):
            # Feature matrix provided
            if labels is None:
                raise ValueError("Labels must be provided when data_source is a feature matrix")
            X, y = data_source, labels
        elif isinstance(data_source, list):
            # Raw events - need feature extraction
            if not feature_extraction:
                raise ValueError("Feature extraction required for raw event data")
            if labels is None:
                raise ValueError("Labels must be provided for raw event data")
            
            logger.info("Extracting features from raw events...")
            X = np.array([
                self.feature_extractor.extract_features([event_group]) 
                if isinstance(event_group, CorrelatedEventGroup) 
                else self.feature_extractor.extract_features(event_group)
                for event_group in data_source
            ])
            y = labels
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")
        
        # Validate dimensions
        if X.shape[1] != 127:
            if feature_extraction:
                logger.warning(f"Expected 127 features after extraction, got {X.shape[1]}")
            else:
                raise ValueError(f"Expected 127 features, got {X.shape[1]}")
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split dataset
        dataset = self._split_dataset(X, y)
        
        # Store metadata
        dataset.metadata = {
            'total_samples': X.shape[0],
            'n_features': X.shape[1],
            'train_samples': len(dataset.y_train),
            'val_samples': len(dataset.y_val),
            'test_samples': len(dataset.y_test),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'feature_extraction_applied': feature_extraction,
            'data_source_type': type(data_source).__name__
        }
        
        # Feature names
        if hasattr(self.feature_extractor, 'get_feature_names'):
            dataset.feature_names = self.feature_extractor.get_feature_names()
        
        # Class names
        unique_labels = sorted(np.unique(y))
        dataset.class_names = [f"Class_{label}" for label in unique_labels]
        
        logger.info(f"Dataset prepared: Train={len(dataset.y_train)}, Val={len(dataset.y_val)}, Test={len(dataset.y_test)}")
        
        return dataset
    
    def _load_data_from_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            # Assume last column is the label
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif file_path.suffix == '.npz':
            data = np.load(file_path)
            X = data['X']
            y = data['y']
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            X = data['X']
            y = data['y']
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return X, y
    
    def _split_dataset(self, X: np.ndarray, y: np.ndarray) -> TrainingDataset:
        """Split dataset into train/val/test sets"""
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp if self.stratify else None
        )
        
        return TrainingDataset(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )
    
    def train_models(self, dataset: TrainingDataset, experiment_name: Optional[str] = None) -> TrainingResults:
        """
        Train ensemble models with comprehensive evaluation
        
        Args:
            dataset: Prepared training dataset
            experiment_name: Name for this training experiment
            
        Returns:
            Training results
        """
        start_time = time.time()
        
        if experiment_name is None:
            experiment_name = f"training_{int(time.time())}"
        
        logger.info(f"Starting training experiment: {experiment_name}")
        
        # Handle class imbalance
        if self.handle_imbalanced_data:
            self._handle_class_imbalance(dataset)
        
        # Feature selection (optional)
        if self.feature_selection:
            dataset = self._apply_feature_selection(dataset)
        
        # Train ensemble
        logger.info("Training ensemble models...")
        ensemble_metrics = self.ensemble_coordinator.train_ensemble(
            dataset.X_train, dataset.y_train,
            train_xgboost=True,
            train_mlp=True,
            cross_validate=self.enable_cross_validation
        )
        
        # Hyperparameter tuning (optional)
        if self.enable_hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            self._perform_hyperparameter_tuning(dataset)
        
        # Cross-validation evaluation
        cv_results = {}
        if self.enable_cross_validation:
            logger.info("Performing cross-validation...")
            cv_results = self._perform_cross_validation(dataset)
        
        # Validation evaluation
        logger.info("Evaluating on validation set...")
        validation_metrics = self.ensemble_coordinator.evaluate_ensemble(dataset.X_val, dataset.y_val)
        
        # Test evaluation
        logger.info("Evaluating on test set...")
        test_metrics = self.ensemble_coordinator.evaluate_ensemble(dataset.X_test, dataset.y_test)
        
        # Feature importance analysis
        feature_importance = self._analyze_feature_importance(dataset)
        
        # Generate classification report
        predictions = self.ensemble_coordinator.predict(dataset.X_test)
        validation_report = classification_report(
            dataset.y_test, predictions,
            target_names=dataset.class_names,
            digits=4
        )
        
        # Save models
        model_paths = {}
        if self.save_models:
            logger.info("Saving trained models...")
            model_paths = self.ensemble_coordinator.save_ensemble(experiment_name)
        
        # Calculate total training time
        training_time = time.time() - start_time
        
        # Compile results
        results = TrainingResults(
            ensemble_metrics=test_metrics,
            xgboost_metrics=ensemble_metrics.get('xgboost', {}),
            mlp_metrics=ensemble_metrics.get('mlp', {}),
            cross_validation_results=cv_results,
            training_time=training_time,
            model_paths=model_paths,
            dataset_info=dataset.metadata,
            validation_report=validation_report,
            feature_importance=feature_importance
        )
        
        # Store results
        self.last_training_results = results
        self.training_history.append({
            'experiment_name': experiment_name,
            'timestamp': time.time(),
            'results_summary': {
                'ensemble_accuracy': test_metrics.get('ensemble_accuracy', 0),
                'ensemble_f1': test_metrics.get('ensemble_f1', 0),
                'training_time': training_time
            }
        })
        
        # Generate reports
        if self.generate_reports:
            self._generate_training_report(results, experiment_name, dataset)
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Final test accuracy: {test_metrics.get('ensemble_accuracy', 0):.4f}")
        logger.info(f"Final test F1: {test_metrics.get('ensemble_f1', 0):.4f}")
        
        return results
    
    def _handle_class_imbalance(self, dataset: TrainingDataset):
        """Handle class imbalance in the dataset"""
        class_counts = np.bincount(dataset.y_train)
        class_ratio = class_counts.min() / class_counts.max()
        
        if class_ratio < 0.5:  # Significant imbalance
            logger.info(f"Class imbalance detected (ratio: {class_ratio:.3f}), computing class weights")
            
            # Compute class weights
            classes = np.unique(dataset.y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=dataset.y_train)
            
            # Store in config for models to use
            self.config['class_weights'] = dict(zip(classes, class_weights))
            
            logger.info(f"Class weights: {self.config['class_weights']}")
    
    def _apply_feature_selection(self, dataset: TrainingDataset) -> TrainingDataset:
        """Apply feature selection to reduce dimensionality"""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        logger.info(f"Applying feature selection: selecting top {self.feature_selection_k} features")
        
        selector = SelectKBest(score_func=f_classif, k=self.feature_selection_k)
        
        # Fit on training data
        X_train_selected = selector.fit_transform(dataset.X_train, dataset.y_train)
        X_val_selected = selector.transform(dataset.X_val)
        X_test_selected = selector.transform(dataset.X_test)
        
        # Get selected feature indices
        selected_features = selector.get_support(indices=True)
        
        # Update feature names if available
        if dataset.feature_names:
            dataset.feature_names = [dataset.feature_names[i] for i in selected_features]
        
        # Update dataset
        dataset.X_train = X_train_selected
        dataset.X_val = X_val_selected
        dataset.X_test = X_test_selected
        dataset.metadata['feature_selection_applied'] = True
        dataset.metadata['selected_features'] = selected_features.tolist()
        dataset.metadata['n_features_after_selection'] = X_train_selected.shape[1]
        
        logger.info(f"Feature selection completed: {X_train_selected.shape[1]} features selected")
        
        return dataset
    
    def _perform_hyperparameter_tuning(self, dataset: TrainingDataset):
        """Perform hyperparameter tuning for both models"""
        logger.info("Tuning XGBoost hyperparameters...")
        
        # XGBoost tuning
        xgb_param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [50, 100, 150],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        xgb_tuning_results = self.ensemble_coordinator.xgboost_classifier.hyperparameter_tuning(
            dataset.X_train, dataset.y_train, param_grid=xgb_param_grid
        )
        
        logger.info(f"XGBoost tuning completed: {xgb_tuning_results}")
        
        # Note: MLP hyperparameter tuning would require additional implementation
        # For now, we use the default architecture
    
    def _perform_cross_validation(self, dataset: TrainingDataset) -> Dict[str, Any]:
        """Perform cross-validation evaluation"""
        from sklearn.base import BaseEstimator, ClassifierMixin
        from sklearn.model_selection import cross_validate
        
        # Create a wrapper for ensemble coordinator
        class EnsembleWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, coordinator):
                self.coordinator = coordinator
            
            def fit(self, X, y):
                self.coordinator.train_ensemble(X, y, cross_validate=False)
                return self
            
            def predict(self, X):
                return self.coordinator.predict(X)
        
        # Perform cross-validation
        wrapper = EnsembleWrapper(self.ensemble_coordinator)
        
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = cross_validate(
            wrapper, dataset.X_train, dataset.y_train,
            cv=self.cross_validation_folds,
            scoring=scoring,
            n_jobs=1,  # Ensemble is already parallel
            return_train_score=True
        )
        
        # Summarize results
        cv_summary = {}
        for metric in scoring:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            
            cv_summary[f'{metric}_test_mean'] = cv_results[test_key].mean()
            cv_summary[f'{metric}_test_std'] = cv_results[test_key].std()
            cv_summary[f'{metric}_train_mean'] = cv_results[train_key].mean()
            cv_summary[f'{metric}_train_std'] = cv_results[train_key].std()
        
        cv_summary['cv_folds'] = self.cross_validation_folds
        cv_summary['fit_time_mean'] = cv_results['fit_time'].mean()
        cv_summary['score_time_mean'] = cv_results['score_time'].mean()
        
        logger.info(f"Cross-validation results: Accuracy={cv_summary['accuracy_test_mean']:.4f}±{cv_summary['accuracy_test_std']:.4f}")
        
        return cv_summary
    
    def _analyze_feature_importance(self, dataset: TrainingDataset) -> Dict[str, Any]:
        """Analyze feature importance from both models"""
        importance_analysis = {}
        
        # XGBoost feature importance
        if self.ensemble_coordinator.xgboost_classifier.model is not None:
            xgb_importance = self.ensemble_coordinator.xgboost_classifier.get_feature_importance(top_k=20)
            importance_analysis['xgboost_top_features'] = xgb_importance
        
        # MLP feature importance (if available)
        if self.ensemble_coordinator.mlp_classifier.model is not None:
            try:
                # Sample-based importance for MLP
                sample_size = min(100, len(dataset.X_test))
                mlp_importance = self.ensemble_coordinator.mlp_classifier.analyze_feature_importance(
                    dataset.X_test[:sample_size]
                )
                # Get top features
                top_mlp_features = dict(sorted(mlp_importance.items(), key=lambda x: x[1], reverse=True)[:20])
                importance_analysis['mlp_top_features'] = top_mlp_features
            except Exception as e:
                logger.warning(f"MLP feature importance analysis failed: {e}")
        
        return importance_analysis
    
    def _generate_training_report(self, results: TrainingResults, experiment_name: str, dataset: TrainingDataset):
        """Generate comprehensive training report"""
        report_path = self.report_dir / f"{experiment_name}_report.html"
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CrossLayerGuardian Training Report - {experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
                .metric-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .confusion-matrix {{ max-width: 400px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CrossLayerGuardian Training Report</h1>
                <p><strong>Experiment:</strong> {experiment_name}</p>
                <p><strong>Training Time:</strong> {results.training_time:.2f} seconds</p>
                <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Information</h2>
                <div class="metrics">
                    <div class="metric-box">
                        <h3>Dataset Stats</h3>
                        <p>Total Samples: {results.dataset_info['total_samples']}</p>
                        <p>Features: {results.dataset_info['n_features']}</p>
                        <p>Train/Val/Test: {results.dataset_info['train_samples']}/{results.dataset_info['val_samples']}/{results.dataset_info['test_samples']}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Class Distribution</h3>
                        {''.join([f'<p>Class {k}: {v} samples</p>' for k, v in results.dataset_info['class_distribution'].items()])}
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Ensemble Performance</h2>
                <div class="metrics">
                    <div class="metric-box">
                        <h3>Test Set Metrics</h3>
                        <p>Accuracy: {results.ensemble_metrics.get('ensemble_accuracy', 0):.4f}</p>
                        <p>Precision: {results.ensemble_metrics.get('ensemble_precision', 0):.4f}</p>
                        <p>Recall: {results.ensemble_metrics.get('ensemble_recall', 0):.4f}</p>
                        <p>F1-Score: {results.ensemble_metrics.get('ensemble_f1', 0):.4f}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Individual Models</h3>
                        <p>XGBoost Accuracy: {results.ensemble_metrics.get('xgboost_accuracy', 0):.4f}</p>
                        <p>MLP Accuracy: {results.ensemble_metrics.get('mlp_accuracy', 0):.4f}</p>
                        <p>Agreement Rate: {results.ensemble_metrics.get('agreement_rate', 0):.4f}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Classification Report</h2>
                <pre>{results.validation_report}</pre>
            </div>
            
            <div class="section">
                <h2>Cross-Validation Results</h2>
                {self._format_cv_results_html(results.cross_validation_results)}
            </div>
            
            <div class="section">
                <h2>Feature Importance</h2>
                {self._format_feature_importance_html(results.feature_importance, dataset.feature_names)}
            </div>
            
        </body>
        </html>
        """
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Training report saved to {report_path}")
        
        # Also save confusion matrix plot
        self._save_confusion_matrix(results, experiment_name, dataset)
    
    def _format_cv_results_html(self, cv_results: Dict[str, Any]) -> str:
        """Format cross-validation results as HTML"""
        if not cv_results:
            return "<p>Cross-validation not performed</p>"
        
        html = "<table><tr><th>Metric</th><th>Mean</th><th>Std</th></tr>"
        
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        for metric in metrics:
            mean_key = f'{metric}_test_mean'
            std_key = f'{metric}_test_std'
            if mean_key in cv_results:
                html += f"<tr><td>{metric.replace('_weighted', '').title()}</td><td>{cv_results[mean_key]:.4f}</td><td>±{cv_results[std_key]:.4f}</td></tr>"
        
        html += "</table>"
        return html
    
    def _format_feature_importance_html(self, importance: Dict[str, Any], feature_names: List[str]) -> str:
        """Format feature importance as HTML"""
        html = ""
        
        if 'xgboost_top_features' in importance:
            html += "<h3>XGBoost Top Features</h3><table><tr><th>Feature</th><th>Importance</th></tr>"
            for feat_idx, imp_score in importance['xgboost_top_features'].items():
                feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
                html += f"<tr><td>{feat_name}</td><td>{imp_score:.4f}</td></tr>"
            html += "</table>"
        
        if 'mlp_top_features' in importance:
            html += "<h3>MLP Top Features</h3><table><tr><th>Feature</th><th>Importance</th></tr>"
            for feat_idx, imp_score in importance['mlp_top_features'].items():
                feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
                html += f"<tr><td>{feat_name}</td><td>{imp_score:.4f}</td></tr>"
            html += "</table>"
        
        return html if html else "<p>Feature importance analysis not available</p>"
    
    def _save_confusion_matrix(self, results: TrainingResults, experiment_name: str, dataset: TrainingDataset):
        """Save confusion matrix plot"""
        predictions = self.ensemble_coordinator.predict(dataset.X_test)
        cm = confusion_matrix(dataset.y_test, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=dataset.class_names, 
                    yticklabels=dataset.class_names)
        plt.title(f'Confusion Matrix - {experiment_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = self.report_dir / f"{experiment_name}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {cm_path}")
    
    def load_and_evaluate(self, model_path: str, test_data: Union[str, np.ndarray], test_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Load trained model and evaluate on test data
        
        Args:
            model_path: Path to saved model
            test_data: Test data (file path or feature matrix)
            test_labels: Test labels (if test_data is feature matrix)
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Loading model from {model_path}")
        
        # Load ensemble
        success = self.ensemble_coordinator.load_ensemble(model_path)
        if not success:
            raise ValueError(f"Failed to load model from {model_path}")
        
        # Prepare test data
        if isinstance(test_data, str):
            X_test, y_test = self._load_data_from_file(test_data)
        else:
            if test_labels is None:
                raise ValueError("Test labels must be provided when test_data is a feature matrix")
            X_test, y_test = test_data, test_labels
        
        # Evaluate
        evaluation_results = self.ensemble_coordinator.evaluate_ensemble(X_test, y_test)
        
        logger.info(f"Evaluation completed: Accuracy={evaluation_results['ensemble_accuracy']:.4f}")
        
        return evaluation_results
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history
    
    def export_model_for_deployment(self, model_name: str, export_format: str = 'pkl') -> str:
        """
        Export model for deployment
        
        Args:
            model_name: Base name for exported model
            export_format: Format for export ('pkl', 'onnx', etc.)
            
        Returns:
            Path to exported model
        """
        if not self.ensemble_coordinator.is_trained:
            raise ValueError("No trained model available for export")
        
        export_path = self.model_dir / f"{model_name}_deployment.{export_format}"
        
        if export_format == 'pkl':
            # Export as pickle (default)
            self.ensemble_coordinator.save_ensemble(model_name + "_deployment")
        else:
            raise ValueError(f"Export format '{export_format}' not supported yet")
        
        logger.info(f"Model exported for deployment to {export_path}")
        return str(export_path)

if __name__ == "__main__":
    # Test training pipeline
    config = {
        'test_size': 0.2,
        'val_size': 0.2,
        'cv_folds': 3,  # Reduced for testing
        'cross_validation': True,
        'hyperparameter_tuning': False,  # Disabled for quick testing
        'save_models': True,
        'generate_reports': True,
        'model_dir': 'test_models',
        'report_dir': 'test_reports',
        'random_state': 42
    }
    
    pipeline = MLTrainingPipeline(config)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 127)  # 127 features
    y = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])  # 80% normal, 20% anomaly
    
    # Prepare dataset
    dataset = pipeline.prepare_dataset(X, y, feature_extraction=False)
    print(f"Dataset prepared: {dataset.metadata}")
    
    # Train models
    results = pipeline.train_models(dataset, experiment_name="test_experiment")
    print(f"Training completed: Accuracy={results.ensemble_metrics['ensemble_accuracy']:.4f}")
    
    # Get training history
    history = pipeline.get_training_history()
    print(f"Training history: {len(history)} experiments")