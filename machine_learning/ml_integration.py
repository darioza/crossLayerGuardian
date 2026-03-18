"""
CrossLayerGuardian ML Integration Bridge
Connects EventCorrelator output to ML Ensemble for real-time threat classification
Manages feature extraction from correlated events and ML prediction pipeline
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import asyncio
from pathlib import Path

from .ensemble_coordinator import EnsembleCoordinator, EnsemblePrediction
from .feature_extractor import CrossLayerFeatureExtractor, CorrelatedEventGroup
from ..data_processing.event_correlator import CrossLayerCorrelator

logger = logging.getLogger(__name__)

@dataclass
class MLClassificationResult:
    """Container for ML classification results"""
    event_group: CorrelatedEventGroup
    prediction: int  # 0=normal, 1=anomaly
    confidence: float
    ensemble_details: EnsemblePrediction
    features: np.ndarray
    processing_time: float
    timestamp: float = field(default_factory=time.time)
    alert_generated: bool = False

@dataclass
class MLIntegrationMetrics:
    """Metrics for ML integration performance"""
    total_classifications: int = 0
    anomalies_detected: int = 0
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    feature_extraction_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    ml_prediction_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    queue_sizes: deque = field(default_factory=lambda: deque(maxlen=100))
    errors_encountered: int = 0
    last_update: float = field(default_factory=time.time)

class MLIntegrationBridge:
    """
    Bridge between EventCorrelator and ML Ensemble
    Handles real-time classification of correlated events
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Component initialization
        self.ensemble_coordinator = EnsembleCoordinator(config)
        self.feature_extractor = CrossLayerFeatureExtractor(config)
        
        # Processing configuration
        self.batch_size = config.get('ml_batch_size', 32)
        self.max_queue_size = config.get('ml_max_queue_size', 1000)
        self.processing_timeout = config.get('ml_processing_timeout', 5.0)
        self.enable_batching = config.get('ml_enable_batching', True)
        self.batch_timeout = config.get('ml_batch_timeout', 1.0)
        
        # Real-time processing
        self.enable_realtime = config.get('ml_enable_realtime', True)
        self.num_workers = config.get('ml_workers', 2)
        self.alert_threshold = config.get('ml_alert_threshold', 0.7)
        
        # Queues and threading
        self.event_queue = queue.Queue(maxsize=self.max_queue_size)
        self.result_queue = queue.Queue()
        self.workers_active = False
        self.worker_threads = []
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        # Callbacks and handlers
        self.alert_callbacks: List[Callable[[MLClassificationResult], None]] = []
        self.result_callbacks: List[Callable[[MLClassificationResult], None]] = []
        
        # Metrics and monitoring
        self.metrics = MLIntegrationMetrics()
        self.performance_monitor = config.get('ml_performance_monitor', True)
        
        # State management
        self.is_trained = False
        self.last_model_check = 0
        self.model_check_interval = config.get('ml_model_check_interval', 60.0)  # 60 seconds
        
        # Adaptive processing
        self.adaptive_batching = config.get('ml_adaptive_batching', True)
        self.recent_load = deque(maxlen=100)
        
        logger.info(f"MLIntegrationBridge initialized with {self.num_workers} workers, batch_size={self.batch_size}")
    
    def start_processing(self):
        """Start ML processing workers"""
        if self.workers_active:
            logger.warning("ML processing already active")
            return
        
        if not self.is_trained:
            logger.warning("ML models not trained, starting with prediction disabled")
        
        self.workers_active = True
        
        # Start worker threads
        for i in range(self.num_workers):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"MLWorker-{i}",
                daemon=True
            )
            worker_thread.start()
            self.worker_threads.append(worker_thread)
        
        # Start batch processor if enabled
        if self.enable_batching:
            batch_thread = threading.Thread(
                target=self._batch_processor_loop,
                name="MLBatchProcessor",
                daemon=True
            )
            batch_thread.start()
            self.worker_threads.append(batch_thread)
        
        # Start metrics updater
        if self.performance_monitor:
            metrics_thread = threading.Thread(
                target=self._metrics_updater_loop,
                name="MLMetricsUpdater",
                daemon=True
            )
            metrics_thread.start()
            self.worker_threads.append(metrics_thread)
        
        logger.info(f"ML processing started with {len(self.worker_threads)} threads")
    
    def stop_processing(self):
        """Stop ML processing workers"""
        if not self.workers_active:
            return
        
        logger.info("Stopping ML processing...")
        self.workers_active = False
        
        # Signal workers to stop
        for _ in range(self.num_workers):
            try:
                self.event_queue.put(None, timeout=1.0)  # Sentinel value
            except queue.Full:
                pass
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.worker_threads.clear()
        logger.info("ML processing stopped")
    
    def load_trained_models(self, model_path: str) -> bool:
        """
        Load trained ML models
        
        Args:
            model_path: Path to trained ensemble models
            
        Returns:
            Success status
        """
        try:
            success = self.ensemble_coordinator.load_ensemble(model_path)
            if success:
                self.is_trained = True
                self.last_model_check = time.time()
                logger.info(f"ML models loaded successfully from {model_path}")
            else:
                logger.error(f"Failed to load ML models from {model_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            return False
    
    def process_correlated_events(self, event_groups: List[CorrelatedEventGroup]) -> List[MLClassificationResult]:
        """
        Process correlated events through ML pipeline
        
        Args:
            event_groups: List of correlated event groups from EventCorrelator
            
        Returns:
            List of ML classification results
        """
        if not self.is_trained:
            logger.warning("ML models not trained, skipping classification")
            return []
        
        if not event_groups:
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # Extract features
            feature_start = time.time()
            features = self._extract_features_batch(event_groups)
            feature_time = time.time() - feature_start
            
            # Make predictions
            ml_start = time.time()
            predictions = self.ensemble_coordinator.predict(features, return_detailed=True)
            ml_time = time.time() - ml_start
            
            # Create results
            for i, (event_group, prediction) in enumerate(zip(event_groups, predictions)):
                result = MLClassificationResult(
                    event_group=event_group,
                    prediction=prediction.final_prediction,
                    confidence=prediction.confidence_score,
                    ensemble_details=prediction,
                    features=features[i],
                    processing_time=time.time() - start_time
                )
                
                # Check for alerts
                if (result.prediction == 1 and 
                    result.confidence >= self.alert_threshold):
                    result.alert_generated = True
                    self._trigger_alert(result)
                
                results.append(result)
                
                # Call result callbacks
                for callback in self.result_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Result callback error: {e}")
            
            # Update metrics
            total_time = time.time() - start_time
            self._update_processing_metrics(len(results), total_time, feature_time, ml_time)
            
        except Exception as e:
            logger.error(f"Error processing correlated events: {e}")
            self.metrics.errors_encountered += 1
        
        return results
    
    def queue_for_processing(self, event_groups: List[CorrelatedEventGroup]) -> bool:
        """
        Queue correlated events for asynchronous processing
        
        Args:
            event_groups: Event groups to process
            
        Returns:
            Success status
        """
        if not self.workers_active:
            logger.warning("ML processing not active, cannot queue events")
            return False
        
        try:
            # Add to queue with timeout
            self.event_queue.put(event_groups, timeout=0.1)
            return True
            
        except queue.Full:
            logger.warning("ML processing queue full, dropping event groups")
            return False
    
    def _worker_loop(self):
        """Main worker loop for processing events"""
        logger.debug(f"ML worker {threading.current_thread().name} started")
        
        while self.workers_active:
            try:
                # Get event groups from queue
                event_groups = self.event_queue.get(timeout=1.0)
                
                # Check for sentinel value
                if event_groups is None:
                    break
                
                # Process events
                results = self.process_correlated_events(event_groups)
                
                # Put results in result queue
                for result in results:
                    try:
                        self.result_queue.put(result, timeout=0.1)
                    except queue.Full:
                        logger.warning("Result queue full, dropping result")
                
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self.metrics.errors_encountered += 1
        
        logger.debug(f"ML worker {threading.current_thread().name} stopped")
    
    def _batch_processor_loop(self):
        """Batch processor for efficient ML inference"""
        logger.debug("ML batch processor started")
        
        batch = []
        last_batch_time = time.time()
        
        while self.workers_active:
            try:
                # Try to get event from queue
                try:
                    event_groups = self.event_queue.get(timeout=0.1)
                    if event_groups is None:
                        break
                    
                    batch.extend(event_groups)
                    
                except queue.Empty:
                    pass
                
                # Process batch if conditions met
                current_time = time.time()
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and current_time - last_batch_time >= self.batch_timeout)
                )
                
                if should_process and batch:
                    # Process batch
                    results = self.process_correlated_events(batch)
                    
                    # Put results in result queue
                    for result in results:
                        try:
                            self.result_queue.put(result, timeout=0.1)
                        except queue.Full:
                            pass
                    
                    # Reset batch
                    batch.clear()
                    last_batch_time = current_time
                
                # Adaptive batch sizing
                if self.adaptive_batching:
                    self._adapt_batch_size()
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                batch.clear()
        
        logger.debug("ML batch processor stopped")
    
    def _metrics_updater_loop(self):
        """Metrics updater loop"""
        while self.workers_active:
            try:
                # Update queue size metrics
                self.metrics.queue_sizes.append(self.event_queue.qsize())
                
                # Check model status periodically
                if time.time() - self.last_model_check > self.model_check_interval:
                    self._check_model_status()
                    self.last_model_check = time.time()
                
                # Update timestamp
                self.metrics.last_update = time.time()
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Metrics updater error: {e}")
    
    def _extract_features_batch(self, event_groups: List[CorrelatedEventGroup]) -> np.ndarray:
        """Extract features from batch of event groups"""
        features = []
        
        for event_group in event_groups:
            feature_vector = self.feature_extractor.extract_features([event_group])
            features.append(feature_vector)
        
        return np.array(features)
    
    def _update_processing_metrics(self, num_processed: int, total_time: float, feature_time: float, ml_time: float):
        """Update processing metrics"""
        self.metrics.total_classifications += num_processed
        self.metrics.processing_times.append(total_time)
        self.metrics.feature_extraction_times.append(feature_time)
        self.metrics.ml_prediction_times.append(ml_time)
    
    def _trigger_alert(self, result: MLClassificationResult):
        """Trigger alert for high-confidence anomaly"""
        logger.warning(f"ANOMALY DETECTED: Confidence={result.confidence:.4f}, "
                      f"Events={len(result.event_group.events)}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _adapt_batch_size(self):
        """Adapt batch size based on processing load"""
        if len(self.recent_load) < 10:
            return
        
        avg_load = np.mean(list(self.recent_load))
        
        if avg_load > 0.8:  # High load - increase batch size
            self.batch_size = min(self.batch_size + 5, 128)
        elif avg_load < 0.3:  # Low load - decrease batch size
            self.batch_size = max(self.batch_size - 5, 8)
        
        self.recent_load.append(self.event_queue.qsize() / self.max_queue_size)
    
    def _check_model_status(self):
        """Check ML model status"""
        if not self.is_trained:
            return
        
        try:
            # Verify models are still functional
            test_features = np.random.randn(1, 127)
            self.ensemble_coordinator.predict(test_features)
            
        except Exception as e:
            logger.error(f"Model status check failed: {e}")
            self.is_trained = False
    
    def register_alert_callback(self, callback: Callable[[MLClassificationResult], None]):
        """Register callback for anomaly alerts"""
        self.alert_callbacks.append(callback)
        logger.info(f"Alert callback registered: {callback.__name__}")
    
    def register_result_callback(self, callback: Callable[[MLClassificationResult], None]):
        """Register callback for all classification results"""
        self.result_callbacks.append(callback)
        logger.info(f"Result callback registered: {callback.__name__}")
    
    def get_results(self, timeout: float = 1.0) -> List[MLClassificationResult]:
        """
        Get processed results from result queue
        
        Args:
            timeout: Timeout for getting results
            
        Returns:
            List of classification results
        """
        results = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.result_queue.get(timeout=0.1)
                results.append(result)
                self.result_queue.task_done()
                
            except queue.Empty:
                break
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics"""
        metrics_dict = {
            'total_classifications': self.metrics.total_classifications,
            'anomalies_detected': self.metrics.anomalies_detected,
            'errors_encountered': self.metrics.errors_encountered,
            'is_trained': self.is_trained,
            'workers_active': self.workers_active,
            'num_workers': len(self.worker_threads),
            'queue_size': self.event_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'batch_size': self.batch_size,
            'last_update': self.metrics.last_update
        }
        
        # Processing time statistics
        if self.metrics.processing_times:
            processing_times = list(self.metrics.processing_times)
            metrics_dict.update({
                'avg_processing_time': np.mean(processing_times),
                'max_processing_time': np.max(processing_times),
                'min_processing_time': np.min(processing_times)
            })
        
        # Feature extraction time statistics
        if self.metrics.feature_extraction_times:
            feature_times = list(self.metrics.feature_extraction_times)
            metrics_dict.update({
                'avg_feature_extraction_time': np.mean(feature_times),
                'total_feature_extraction_time': np.sum(feature_times)
            })
        
        # ML prediction time statistics
        if self.metrics.ml_prediction_times:
            ml_times = list(self.metrics.ml_prediction_times)
            metrics_dict.update({
                'avg_ml_prediction_time': np.mean(ml_times),
                'total_ml_prediction_time': np.sum(ml_times)
            })
        
        # Queue size statistics
        if self.metrics.queue_sizes:
            queue_sizes = list(self.metrics.queue_sizes)
            metrics_dict.update({
                'avg_queue_size': np.mean(queue_sizes),
                'max_queue_size_observed': np.max(queue_sizes)
            })
        
        return metrics_dict
    
    def reset_metrics(self):
        """Reset integration metrics"""
        self.metrics = MLIntegrationMetrics()
        logger.info("ML integration metrics reset")
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update configuration parameters"""
        self.config.update(new_config)
        
        # Update parameters that can be changed dynamically
        self.batch_size = new_config.get('ml_batch_size', self.batch_size)
        self.alert_threshold = new_config.get('ml_alert_threshold', self.alert_threshold)
        self.adaptive_batching = new_config.get('ml_adaptive_batching', self.adaptive_batching)
        
        logger.info(f"Configuration updated: batch_size={self.batch_size}, alert_threshold={self.alert_threshold}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_processing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_processing()

# Integration helper functions
def create_alert_handler(log_file: Optional[str] = None) -> Callable[[MLClassificationResult], None]:
    """Create alert handler function"""
    def alert_handler(result: MLClassificationResult):
        alert_msg = (f"ANOMALY ALERT: {result.prediction} (confidence: {result.confidence:.4f}) "
                    f"- {len(result.event_group.events)} correlated events")
        
        logger.critical(alert_msg)
        
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {alert_msg}\n")
    
    return alert_handler

def create_metrics_logger() -> Callable[[MLClassificationResult], None]:
    """Create metrics logging function"""
    def metrics_logger(result: MLClassificationResult):
        logger.info(f"ML Classification: {result.prediction} "
                   f"(conf: {result.confidence:.3f}, "
                   f"time: {result.processing_time:.3f}s)")
    
    return metrics_logger

if __name__ == "__main__":
    # Test ML integration bridge
    config = {
        'ml_batch_size': 16,
        'ml_max_queue_size': 100,
        'ml_workers': 2,
        'ml_alert_threshold': 0.7,
        'ml_enable_batching': True,
        'ml_adaptive_batching': True,
        'model_dir': 'test_models',
        'random_state': 42
    }
    
    # Create integration bridge
    bridge = MLIntegrationBridge(config)
    
    # Register callbacks
    alert_handler = create_alert_handler("test_alerts.log")
    metrics_logger = create_metrics_logger()
    
    bridge.register_alert_callback(alert_handler)
    bridge.register_result_callback(metrics_logger)
    
    # Test with sample data (would normally come from EventCorrelator)
    sample_events = [
        CorrelatedEventGroup(
            events=[
                {'timestamp': time.time(), 'src_ip': '192.168.1.10', 'protocol': 'TCP'},
                {'timestamp': time.time() + 0.1, 'filename': '/tmp/test.log', 'syscall': 'write'}
            ],
            correlation_score=0.8,
            timestamp=time.time(),
            duration=2.5,
            event_types={'network', 'filesystem'}
        )
    ]
    
    print(f"Testing ML integration bridge...")
    
    # Test synchronous processing
    results = bridge.process_correlated_events(sample_events)
    print(f"Processed {len(results)} event groups synchronously")
    
    # Get metrics
    metrics = bridge.get_metrics()
    print(f"Integration metrics: {metrics}")
    
    print("ML integration bridge test completed")