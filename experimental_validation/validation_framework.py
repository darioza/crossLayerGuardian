"""
CrossLayerGuardian Experimental Validation Framework
Comprehensive testing and validation system for dissertation evaluation
Supports CICIDS2018 dataset processing and synthetic data generation
"""

import numpy as np
import pandas as pd
import time
import logging
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Generator
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import subprocess
import threading
import queue
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import CrossLayerGuardian components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from machine_learning.feature_extractor import CrossLayerFeatureExtractor, CorrelatedEventGroup
from machine_learning.ensemble_coordinator import EnsembleCoordinator
from machine_learning.training_pipeline import MLTrainingPipeline
from machine_learning.ml_integration import MLIntegrationBridge
from config_loader import get_config_loader

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for experimental validation"""
    experiment_name: str
    description: str
    dataset_path: Optional[str] = None
    synthetic_data_size: int = 10000
    test_duration_seconds: int = 300
    performance_monitoring: bool = True
    generate_reports: bool = True
    output_dir: str = "experimental_results"
    random_seed: int = 42
    
@dataclass
class ValidationResult:
    """Results from validation experiment"""
    experiment_name: str
    timestamp: datetime
    accuracy_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    timing_metrics: Dict[str, float]
    system_metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    roc_data: Dict[str, np.ndarray]
    feature_importance: Dict[str, Any]
    detailed_results: Dict[str, Any]

class CICIDS2018DatasetProcessor:
    """
    Processor for CICIDS2018 dataset
    Converts CSV format to CrossLayerGuardian event format
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.label_mapping = {
            'BENIGN': 0,
            'Bot': 1,
            'DDoS': 1,
            'DoS GoldenEye': 1,
            'DoS Hulk': 1,
            'DoS Slowhttptest': 1,
            'DoS slowloris': 1,
            'FTP-Patator': 1,
            'Heartbleed': 1,
            'Infiltration': 1,
            'PortScan': 1,
            'SSH-Patator': 1,
            'Web Attack': 1
        }
        
    def load_dataset(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load CICIDS2018 dataset"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        logger.info(f"Loading CICIDS2018 dataset from {self.dataset_path}")
        
        # Load CSV file
        df = pd.read_csv(self.dataset_path)
        
        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Extract labels
        if 'Label' in df.columns:
            labels = df['Label'].map(self.label_mapping).fillna(1).astype(int)
        else:
            # If no labels, assume all benign for feature extraction testing
            labels = np.zeros(len(df))
        
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
        return df, labels.values
    
    def convert_to_event_groups(self, df: pd.DataFrame, labels: np.ndarray) -> List[Tuple[CorrelatedEventGroup, int]]:
        """Convert CICIDS2018 data to CrossLayerGuardian event format"""
        event_groups = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            # Create synthetic network event
            network_event = {
                'timestamp': time.time() + idx * 0.001,  # Simulate timing
                'src_ip': f"192.168.1.{(idx % 254) + 1}",
                'dst_ip': f"10.0.0.{((idx + 50) % 254) + 1}",
                'src_port': int(row.get('Src Port', 0)) if pd.notna(row.get('Src Port', 0)) else 80,
                'dst_port': int(row.get('Dst Port', 0)) if pd.notna(row.get('Dst Port', 0)) else 443,
                'protocol': row.get('Protocol', 'TCP'),
                'bytes': int(row.get('Total Length of Fwd Packets', 0)) if pd.notna(row.get('Total Length of Fwd Packets', 0)) else 1500,
                'pid': 1000 + (idx % 100)
            }
            
            # Create synthetic filesystem event (for some samples)
            filesystem_event = None
            if idx % 3 == 0:  # Add filesystem events for 1/3 of samples
                filesystem_event = {
                    'timestamp': network_event['timestamp'] + 0.0005,
                    'pid': network_event['pid'],
                    'syscall': 'write' if idx % 2 == 0 else 'read',
                    'filename': f"/tmp/file_{idx % 1000}.log",
                    'bytes': int(row.get('Total Backward Packets', 0)) if pd.notna(row.get('Total Backward Packets', 0)) else 512
                }
            
            # Create event group
            events = [network_event]
            if filesystem_event:
                events.append(filesystem_event)
            
            event_group = CorrelatedEventGroup(
                events=events,
                correlation_score=0.8 if filesystem_event else 0.6,
                timestamp=network_event['timestamp'],
                duration=0.01 + (idx % 10) * 0.001,
                event_types={'network'} | ({'filesystem'} if filesystem_event else set())
            )
            
            event_groups.append((event_group, labels[idx]))
        
        logger.info(f"Converted {len(event_groups)} samples to event groups")
        return event_groups

class SyntheticDataGenerator:
    """
    Generates synthetic cross-layer events for testing
    Creates realistic attack and normal behavior patterns
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.attack_patterns = {
            'port_scan': self._generate_port_scan,
            'ddos': self._generate_ddos,
            'data_exfiltration': self._generate_data_exfiltration,
            'lateral_movement': self._generate_lateral_movement,
            'privilege_escalation': self._generate_privilege_escalation
        }
    
    def generate_dataset(self, 
                        n_samples: int, 
                        attack_ratio: float = 0.2) -> List[Tuple[CorrelatedEventGroup, int]]:
        """Generate synthetic dataset with normal and attack patterns"""
        
        n_attacks = int(n_samples * attack_ratio)
        n_normal = n_samples - n_attacks
        
        logger.info(f"Generating {n_samples} synthetic samples ({n_normal} normal, {n_attacks} attacks)")
        
        dataset = []
        
        # Generate normal traffic
        for i in range(n_normal):
            event_group = self._generate_normal_pattern(i)
            dataset.append((event_group, 0))  # 0 = normal
        
        # Generate attack patterns
        attack_types = list(self.attack_patterns.keys())
        for i in range(n_attacks):
            attack_type = np.random.choice(attack_types)
            event_group = self.attack_patterns[attack_type](i + n_normal)
            dataset.append((event_group, 1))  # 1 = attack
        
        # Shuffle dataset
        np.random.shuffle(dataset)
        
        logger.info(f"Generated synthetic dataset: {len(dataset)} samples")
        return dataset
    
    def _generate_normal_pattern(self, idx: int) -> CorrelatedEventGroup:
        """Generate normal traffic pattern"""
        base_time = time.time() + idx * 0.1
        
        # Normal web browsing pattern
        network_event = {
            'timestamp': base_time,
            'src_ip': f"192.168.1.{np.random.randint(1, 255)}",
            'dst_ip': f"203.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
            'src_port': np.random.randint(32768, 65535),
            'dst_port': np.random.choice([80, 443, 22, 53]),
            'protocol': 'TCP',
            'bytes': np.random.randint(500, 2000),
            'pid': np.random.randint(1000, 2000)
        }
        
        events = [network_event]
        
        # Sometimes add file access
        if np.random.random() < 0.3:
            file_event = {
                'timestamp': base_time + np.random.uniform(0.001, 0.01),
                'pid': network_event['pid'],
                'syscall': np.random.choice(['read', 'write', 'open']),
                'filename': f"/home/user/documents/file_{np.random.randint(1, 100)}.txt",
                'bytes': np.random.randint(100, 1000)
            }
            events.append(file_event)
        
        return CorrelatedEventGroup(
            events=events,
            correlation_score=np.random.uniform(0.3, 0.7),
            timestamp=base_time,
            duration=np.random.uniform(0.01, 0.1),
            event_types={'network'} | ({'filesystem'} if len(events) > 1 else set())
        )
    
    def _generate_port_scan(self, idx: int) -> CorrelatedEventGroup:
        """Generate port scanning attack pattern"""
        base_time = time.time() + idx * 0.01
        src_ip = f"10.0.0.{np.random.randint(1, 255)}"
        target_ip = "192.168.1.100"
        
        events = []
        # Multiple connection attempts to different ports
        for i in range(np.random.randint(5, 20)):
            event = {
                'timestamp': base_time + i * 0.001,
                'src_ip': src_ip,
                'dst_ip': target_ip,
                'src_port': np.random.randint(32768, 65535),
                'dst_port': np.random.randint(1, 1024),
                'protocol': 'TCP',
                'bytes': 60,  # Small SYN packets
                'pid': 2000 + idx
            }
            events.append(event)
        
        return CorrelatedEventGroup(
            events=events,
            correlation_score=0.9,
            timestamp=base_time,
            duration=len(events) * 0.001,
            event_types={'network'}
        )
    
    def _generate_ddos(self, idx: int) -> CorrelatedEventGroup:
        """Generate DDoS attack pattern"""
        base_time = time.time() + idx * 0.001
        target_ip = "192.168.1.10"
        
        events = []
        # High-volume traffic from multiple sources
        for i in range(np.random.randint(20, 50)):
            event = {
                'timestamp': base_time + i * 0.0001,
                'src_ip': f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                'dst_ip': target_ip,
                'src_port': np.random.randint(1024, 65535),
                'dst_port': 80,
                'protocol': 'TCP',
                'bytes': np.random.randint(1000, 5000),
                'pid': 3000 + idx
            }
            events.append(event)
        
        return CorrelatedEventGroup(
            events=events,
            correlation_score=0.95,
            timestamp=base_time,
            duration=len(events) * 0.0001,
            event_types={'network'}
        )
    
    def _generate_data_exfiltration(self, idx: int) -> CorrelatedEventGroup:
        """Generate data exfiltration pattern"""
        base_time = time.time() + idx * 0.1
        pid = 1500 + idx
        
        # File access followed by network transfer
        file_event = {
            'timestamp': base_time,
            'pid': pid,
            'syscall': 'read',
            'filename': '/etc/passwd',
            'bytes': 2048
        }
        
        network_event = {
            'timestamp': base_time + 0.05,
            'src_ip': '192.168.1.50',
            'dst_ip': f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
            'src_port': np.random.randint(32768, 65535),
            'dst_port': np.random.choice([21, 22, 443, 80]),
            'protocol': 'TCP',
            'bytes': 2048,
            'pid': pid
        }
        
        return CorrelatedEventGroup(
            events=[file_event, network_event],
            correlation_score=0.85,
            timestamp=base_time,
            duration=0.05,
            event_types={'filesystem', 'network'}
        )
    
    def _generate_lateral_movement(self, idx: int) -> CorrelatedEventGroup:
        """Generate lateral movement pattern"""
        base_time = time.time() + idx * 0.1
        
        events = []
        src_ip = "192.168.1.100"
        
        # SSH connections to multiple internal hosts
        for i in range(np.random.randint(3, 8)):
            event = {
                'timestamp': base_time + i * 0.02,
                'src_ip': src_ip,
                'dst_ip': f"192.168.1.{i + 10}",
                'src_port': np.random.randint(32768, 65535),
                'dst_port': 22,
                'protocol': 'TCP',
                'bytes': np.random.randint(500, 1500),
                'pid': 4000 + idx
            }
            events.append(event)
        
        return CorrelatedEventGroup(
            events=events,
            correlation_score=0.8,
            timestamp=base_time,
            duration=len(events) * 0.02,
            event_types={'network'}
        )
    
    def _generate_privilege_escalation(self, idx: int) -> CorrelatedEventGroup:
        """Generate privilege escalation pattern"""
        base_time = time.time() + idx * 0.1
        pid = 5000 + idx
        
        # Suspicious system file access
        events = [
            {
                'timestamp': base_time,
                'pid': pid,
                'syscall': 'open',
                'filename': '/etc/shadow',
                'bytes': 0
            },
            {
                'timestamp': base_time + 0.01,
                'pid': pid,
                'syscall': 'write',
                'filename': '/etc/crontab',
                'bytes': 128
            },
            {
                'timestamp': base_time + 0.02,
                'pid': pid,
                'syscall': 'execve',
                'filename': '/bin/bash',
                'bytes': 0
            }
        ]
        
        return CorrelatedEventGroup(
            events=events,
            correlation_score=0.9,
            timestamp=base_time,
            duration=0.02,
            event_types={'filesystem'}
        )

class ExperimentalValidator:
    """
    Main experimental validation class
    Coordinates all validation experiments and measurements
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.ml_config = get_config_loader().get_ml_config()
        self.feature_extractor = CrossLayerFeatureExtractor(self.ml_config)
        self.ensemble_coordinator = EnsembleCoordinator(self.ml_config)
        self.training_pipeline = MLTrainingPipeline(self.ml_config)
        
        # Data generators
        self.cicids_processor = None
        if config.dataset_path:
            self.cicids_processor = CICIDS2018DatasetProcessor(config.dataset_path)
        
        self.synthetic_generator = SyntheticDataGenerator(config.random_seed)
        
        # Results storage
        self.results = []
        
        logger.info(f"ExperimentalValidator initialized: {config.experiment_name}")
    
    def run_full_validation(self) -> List[ValidationResult]:
        """Run complete validation suite"""
        logger.info(f"Starting full validation: {self.config.experiment_name}")
        
        # 1. Data preparation
        dataset = self._prepare_dataset()
        
        # 2. Model training and evaluation
        training_result = self._train_and_evaluate_models(dataset)
        
        # 3. Performance benchmarking
        performance_result = self._benchmark_performance(dataset)
        
        # 4. System-level testing
        system_result = self._test_system_integration(dataset)
        
        # 5. Generate comprehensive report
        if self.config.generate_reports:
            self._generate_validation_report([training_result, performance_result, system_result])
        
        return [training_result, performance_result, system_result]
    
    def _prepare_dataset(self) -> List[Tuple[CorrelatedEventGroup, int]]:
        """Prepare dataset for validation"""
        logger.info("Preparing validation dataset...")
        
        dataset = []
        
        # Use CICIDS2018 if available
        if self.cicids_processor:
            try:
                df, labels = self.cicids_processor.load_dataset(sample_size=5000)
                cicids_data = self.cicids_processor.convert_to_event_groups(df, labels)
                dataset.extend(cicids_data)
                logger.info(f"Added {len(cicids_data)} CICIDS2018 samples")
            except Exception as e:
                logger.warning(f"Failed to load CICIDS2018 data: {e}")
        
        # Generate synthetic data
        synthetic_data = self.synthetic_generator.generate_dataset(
            self.config.synthetic_data_size,
            attack_ratio=0.3
        )
        dataset.extend(synthetic_data)
        
        logger.info(f"Total dataset size: {len(dataset)} samples")
        
        # Save dataset for reproducibility
        dataset_path = self.output_dir / "validation_dataset.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"Dataset saved to {dataset_path}")
        
        return dataset
    
    def _train_and_evaluate_models(self, dataset: List[Tuple[CorrelatedEventGroup, int]]) -> ValidationResult:
        """Train models and evaluate ML performance"""
        logger.info("Training and evaluating ML models...")
        
        start_time = time.time()
        
        # Extract features
        event_groups, labels = zip(*dataset)
        X = np.array([self.feature_extractor.extract_features([eg]) for eg in event_groups])
        y = np.array(labels)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.config.random_seed, stratify=y
        )
        
        # Train ensemble
        training_metrics = self.ensemble_coordinator.train_ensemble(X_train, y_train)
        
        # Evaluate on test set
        predictions = self.ensemble_coordinator.predict(X_test, return_detailed=True)
        y_pred = np.array([p.final_prediction for p in predictions])
        y_proba = np.array([p.confidence_score for p in predictions])
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = {}
        if hasattr(self.ensemble_coordinator.xgboost_classifier, 'get_feature_importance'):
            feature_importance = self.ensemble_coordinator.xgboost_classifier.get_feature_importance(top_k=20)
        
        training_time = time.time() - start_time
        
        result = ValidationResult(
            experiment_name=f"{self.config.experiment_name}_training",
            timestamp=datetime.now(),
            accuracy_metrics=accuracy_metrics,
            performance_metrics={'training_time': training_time, 'roc_auc': roc_auc},
            timing_metrics={'total_training_time': training_time},
            system_metrics={},
            confusion_matrix=cm,
            roc_data={'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
            feature_importance=feature_importance,
            detailed_results={
                'training_metrics': training_metrics,
                'test_predictions': predictions[:100],  # Sample predictions
                'dataset_info': {
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'feature_dims': X.shape[1],
                    'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
                }
            }
        )
        
        logger.info(f"ML evaluation completed: Accuracy={accuracy_metrics['accuracy']:.4f}, AUC={roc_auc:.4f}")
        return result
    
    def _benchmark_performance(self, dataset: List[Tuple[CorrelatedEventGroup, int]]) -> ValidationResult:
        """Benchmark system performance"""
        logger.info("Benchmarking system performance...")
        
        # Performance monitoring setup
        performance_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'feature_extraction_times': [],
            'prediction_times': [],
            'throughput': []
        }
        
        # Sample dataset for benchmarking
        sample_size = min(1000, len(dataset))
        sample_dataset = dataset[:sample_size]
        
        start_time = time.time()
        
        # Benchmark feature extraction
        extraction_times = []
        event_groups = [eg for eg, _ in sample_dataset]
        
        for event_group in event_groups[:100]:  # Sample for timing
            extract_start = time.time()
            features = self.feature_extractor.extract_features([event_group])
            extraction_time = time.time() - extract_start
            extraction_times.append(extraction_time)
        
        # Benchmark ML prediction (if trained)
        prediction_times = []
        if self.ensemble_coordinator.is_trained:
            X_sample = np.array([self.feature_extractor.extract_features([eg]) for eg, _ in sample_dataset[:100]])
            
            for i in range(len(X_sample)):
                pred_start = time.time()
                _ = self.ensemble_coordinator.predict(X_sample[i:i+1])
                prediction_time = time.time() - pred_start
                prediction_times.append(prediction_time)
        
        # System resource monitoring
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        total_time = time.time() - start_time
        throughput = sample_size / total_time if total_time > 0 else 0
        
        performance_metrics = {
            'avg_feature_extraction_time_ms': np.mean(extraction_times) * 1000 if extraction_times else 0,
            'avg_prediction_time_ms': np.mean(prediction_times) * 1000 if prediction_times else 0,
            'cpu_usage_percent': cpu_percent,
            'memory_usage_mb': memory_mb,
            'throughput_samples_per_second': throughput
        }
        
        result = ValidationResult(
            experiment_name=f"{self.config.experiment_name}_performance",
            timestamp=datetime.now(),
            accuracy_metrics={},
            performance_metrics=performance_metrics,
            timing_metrics={
                'total_benchmark_time': total_time,
                'feature_extraction_times': extraction_times,
                'prediction_times': prediction_times
            },
            system_metrics={
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'throughput': throughput
            },
            confusion_matrix=np.array([]),
            roc_data={},
            feature_importance={},
            detailed_results={
                'sample_size': sample_size,
                'benchmark_duration': total_time
            }
        )
        
        logger.info(f"Performance benchmark completed: {throughput:.1f} samples/sec, "
                   f"{np.mean(extraction_times)*1000:.2f}ms extraction")
        return result
    
    def _test_system_integration(self, dataset: List[Tuple[CorrelatedEventGroup, int]]) -> ValidationResult:
        """Test end-to-end system integration"""
        logger.info("Testing system integration...")
        
        # Initialize ML integration bridge
        ml_bridge = MLIntegrationBridge(self.ml_config)
        
        integration_metrics = {
            'total_events_processed': 0,
            'successful_classifications': 0,
            'failed_classifications': 0,
            'alerts_generated': 0,
            'average_processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Test with sample data
            sample_events = [eg for eg, _ in dataset[:100]]
            
            # Process events
            if self.ensemble_coordinator.is_trained:
                results = ml_bridge.process_correlated_events(sample_events)
                
                integration_metrics['total_events_processed'] = len(sample_events)
                integration_metrics['successful_classifications'] = len(results)
                integration_metrics['alerts_generated'] = sum(1 for r in results if r.alert_generated)
                integration_metrics['average_processing_time'] = np.mean([r.processing_time for r in results])
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            integration_metrics['failed_classifications'] = len(dataset)
        
        total_time = time.time() - start_time
        
        result = ValidationResult(
            experiment_name=f"{self.config.experiment_name}_integration",
            timestamp=datetime.now(),
            accuracy_metrics={},
            performance_metrics=integration_metrics,
            timing_metrics={'total_integration_test_time': total_time},
            system_metrics=integration_metrics,
            confusion_matrix=np.array([]),
            roc_data={},
            feature_importance={},
            detailed_results={
                'integration_test_duration': total_time,
                'test_sample_size': len(dataset)
            }
        )
        
        logger.info(f"Integration test completed: {integration_metrics['successful_classifications']} successful")
        return result
    
    def _generate_validation_report(self, results: List[ValidationResult]):
        """Generate comprehensive validation report"""
        logger.info("Generating validation report...")
        
        # Create HTML report
        report_path = self.output_dir / f"{self.config.experiment_name}_validation_report.html"
        
        html_content = self._create_html_report(results)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        # Save results as JSON
        json_path = self.output_dir / f"{self.config.experiment_name}_results.json"
        self._save_results_json(results, json_path)
        
        # Generate plots
        self._generate_validation_plots(results)
        
        logger.info(f"Validation report generated: {report_path}")
    
    def _create_html_report(self, results: List[ValidationResult]) -> str:
        """Create HTML validation report"""
        # This would be a comprehensive HTML report
        # For brevity, returning a basic template
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CrossLayerGuardian Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>CrossLayerGuardian Experimental Validation</h1>
                <p>Experiment: {self.config.experiment_name}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Validation Summary</h2>
                <p>Total experiments conducted: {len(results)}</p>
                <!-- Detailed results would be inserted here -->
            </div>
        </body>
        </html>
        """
    
    def _save_results_json(self, results: List[ValidationResult], path: Path):
        """Save results as JSON"""
        json_data = []
        for result in results:
            json_data.append({
                'experiment_name': result.experiment_name,
                'timestamp': result.timestamp.isoformat(),
                'accuracy_metrics': result.accuracy_metrics,
                'performance_metrics': result.performance_metrics,
                'timing_metrics': result.timing_metrics,
                'system_metrics': result.system_metrics
            })
        
        with open(path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _generate_validation_plots(self, results: List[ValidationResult]):
        """Generate validation plots"""
        # ROC curves, confusion matrices, performance charts
        # Implementation would create matplotlib/seaborn plots
        pass

if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig(
        experiment_name="crosslayer_validation_phase4",
        description="Comprehensive validation of CrossLayerGuardian system",
        synthetic_data_size=5000,
        test_duration_seconds=300,
        generate_reports=True
    )
    
    validator = ExperimentalValidator(config)
    results = validator.run_full_validation()
    
    print(f"Validation completed: {len(results)} experiments")
    for result in results:
        print(f"  {result.experiment_name}: {result.accuracy_metrics}")