"""
CrossLayerGuardian ML Feature Extractor
Implements 127-dimensional feature extraction from correlated cross-layer events
Based on dissertation specifications for multi-dimensional feature space
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import math
import logging
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class EventFeatures:
    """Container for extracted features from a single event"""
    temporal_features: np.ndarray = field(default_factory=lambda: np.zeros(42))
    spatial_features: np.ndarray = field(default_factory=lambda: np.zeros(35))
    behavioral_features: np.ndarray = field(default_factory=lambda: np.zeros(50))
    
    def to_vector(self) -> np.ndarray:
        """Convert to 127-dimensional feature vector"""
        return np.concatenate([
            self.temporal_features,
            self.spatial_features,
            self.behavioral_features
        ])

@dataclass
class CorrelatedEventGroup:
    """Group of correlated events for feature extraction"""
    events: List[Dict[str, Any]] = field(default_factory=list)
    correlation_score: float = 0.0
    timestamp: float = 0.0
    duration: float = 0.0
    event_types: set = field(default_factory=set)

class CrossLayerFeatureExtractor:
    """
    Advanced feature extraction for cross-layer correlation events
    Generates 127-dimensional feature vectors for ML classification
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_cache = {}
        self.event_history = deque(maxlen=10000)
        self.port_stats = defaultdict(lambda: {'count': 0, 'bytes': 0})
        self.process_stats = defaultdict(lambda: {'syscalls': 0, 'net_events': 0})
        self.flow_patterns = defaultdict(list)
        
        # Feature extraction parameters
        self.temporal_window = config.get('temporal_window', 60.0)  # 60 seconds
        self.spatial_window = config.get('spatial_window', 1000)   # 1000 events
        self.min_correlation_score = config.get('min_correlation_score', 0.3)
        
        logger.info("CrossLayerFeatureExtractor initialized with 127-dimensional feature space")
    
    def extract_features(self, correlated_events: List[CorrelatedEventGroup]) -> np.ndarray:
        """
        Extract comprehensive feature set from correlated event groups
        Returns: 127-dimensional feature vector
        """
        if not correlated_events:
            return np.zeros(127)
        
        features = EventFeatures()
        
        # Extract temporal features (42 dimensions)
        features.temporal_features = self._extract_temporal_features(correlated_events)
        
        # Extract spatial features (35 dimensions)
        features.spatial_features = self._extract_spatial_features(correlated_events)
        
        # Extract behavioral features (50 dimensions)
        features.behavioral_features = self._extract_behavioral_features(correlated_events)
        
        # Update event history for future extractions
        self._update_event_history(correlated_events)
        
        return features.to_vector()
    
    def _extract_temporal_features(self, event_groups: List[CorrelatedEventGroup]) -> np.ndarray:
        """
        Extract temporal features (42 dimensions):
        - Inter-arrival times (10)
        - Event duration statistics (8)
        - Temporal clustering (8)
        - Frequency analysis (8)
        - Periodicity features (8)
        """
        features = np.zeros(42)
        
        if not event_groups:
            return features
        
        # Collect all timestamps
        timestamps = []
        durations = []
        for group in event_groups:
            timestamps.append(group.timestamp)
            durations.append(group.duration)
            for event in group.events:
                if 'timestamp' in event:
                    timestamps.append(event['timestamp'])
        
        timestamps.sort()
        
        # Inter-arrival times (10 features)
        if len(timestamps) > 1:
            inter_arrivals = np.diff(timestamps)
            features[0] = np.mean(inter_arrivals)  # Mean inter-arrival
            features[1] = np.std(inter_arrivals)   # Std inter-arrival
            features[2] = np.min(inter_arrivals)   # Min inter-arrival
            features[3] = np.max(inter_arrivals)   # Max inter-arrival
            features[4] = np.median(inter_arrivals)  # Median inter-arrival
            features[5] = stats.skew(inter_arrivals) if len(inter_arrivals) > 2 else 0
            features[6] = stats.kurtosis(inter_arrivals) if len(inter_arrivals) > 3 else 0
            features[7] = np.percentile(inter_arrivals, 25)  # Q1
            features[8] = np.percentile(inter_arrivals, 75)  # Q3
            features[9] = len(inter_arrivals) / (timestamps[-1] - timestamps[0]) if timestamps[-1] != timestamps[0] else 0
        
        # Duration statistics (8 features)
        if durations:
            durations = np.array(durations)
            features[10] = np.mean(durations)
            features[11] = np.std(durations)
            features[12] = np.min(durations)
            features[13] = np.max(durations)
            features[14] = np.median(durations)
            features[15] = stats.skew(durations) if len(durations) > 2 else 0
            features[16] = stats.kurtosis(durations) if len(durations) > 3 else 0
            features[17] = np.sum(durations)
        
        # Temporal clustering (8 features)
        if len(timestamps) > 2:
            # Time gaps analysis
            gaps = np.diff(timestamps)
            gap_threshold = np.mean(gaps) + 2 * np.std(gaps)
            large_gaps = gaps[gaps > gap_threshold]
            
            features[18] = len(large_gaps)  # Number of temporal clusters
            features[19] = np.mean(large_gaps) if len(large_gaps) > 0 else 0
            features[20] = timestamps[-1] - timestamps[0]  # Total time span
            features[21] = len(timestamps) / (timestamps[-1] - timestamps[0]) if timestamps[-1] != timestamps[0] else 0  # Event density
            
            # Burst detection
            burst_threshold = np.mean(gaps) / 3
            burst_events = np.sum(gaps < burst_threshold)
            features[22] = burst_events / len(gaps) if len(gaps) > 0 else 0
            features[23] = np.max(np.diff(np.where(gaps < burst_threshold)[0])) if len(np.where(gaps < burst_threshold)[0]) > 1 else 0
            features[24] = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0  # Coefficient of variation
            features[25] = len(np.where(gaps > 2 * np.mean(gaps))[0])  # Outlier gaps
        
        # Frequency analysis (8 features)
        if len(timestamps) > 4:
            # FFT-based frequency analysis
            time_series = np.histogram(timestamps, bins=min(50, len(timestamps)//2))[0]
            if len(time_series) > 1:
                fft = np.fft.fft(time_series)
                power_spectrum = np.abs(fft[:len(fft)//2])
                
                features[26] = np.argmax(power_spectrum)  # Dominant frequency index
                features[27] = np.max(power_spectrum)     # Peak power
                features[28] = np.mean(power_spectrum)    # Average power
                features[29] = np.std(power_spectrum)     # Power variability
                features[30] = np.sum(power_spectrum[:5]) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0  # Low freq ratio
                features[31] = np.sum(power_spectrum[-5:]) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0  # High freq ratio
                features[32] = stats.entropy(power_spectrum) if np.sum(power_spectrum) > 0 else 0  # Spectral entropy
                features[33] = np.sum(power_spectrum > np.mean(power_spectrum))  # Significant frequencies
        
        # Periodicity features (8 features)
        if len(timestamps) > 10:
            # Autocorrelation analysis
            time_diffs = np.diff(timestamps)
            if len(time_diffs) > 5:
                autocorr = np.correlate(time_diffs, time_diffs, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
                
                features[34] = np.max(autocorr[1:]) if len(autocorr) > 1 else 0  # Max autocorrelation
                features[35] = np.argmax(autocorr[1:]) if len(autocorr) > 1 else 0  # Period estimate
                features[36] = np.mean(autocorr[1:6]) if len(autocorr) > 5 else 0  # Short-term periodicity
                features[37] = np.mean(autocorr[-5:]) if len(autocorr) > 5 else 0  # Long-term periodicity
                
                # Regularity measures
                features[38] = np.std(time_diffs) / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
                features[39] = len(np.where(np.abs(time_diffs - np.mean(time_diffs)) < np.std(time_diffs))[0]) / len(time_diffs)
                features[40] = np.sum(np.abs(np.diff(time_diffs))) / len(time_diffs) if len(time_diffs) > 1 else 0
                features[41] = len(set(np.round(time_diffs, 3))) / len(time_diffs)  # Uniqueness ratio
        
        return features
    
    def _extract_spatial_features(self, event_groups: List[CorrelatedEventGroup]) -> np.ndarray:
        """
        Extract spatial features (35 dimensions):
        - Network topology (12)
        - Process hierarchy (8)
        - Resource distribution (8)
        - Cross-layer connectivity (7)
        """
        features = np.zeros(35)
        
        # Collect spatial information
        ip_addresses = set()
        ports = set()
        processes = set()
        syscalls = set()
        protocols = set()
        
        for group in event_groups:
            for event in group.events:
                if 'src_ip' in event:
                    ip_addresses.add(event['src_ip'])
                if 'dst_ip' in event:
                    ip_addresses.add(event['dst_ip'])
                if 'src_port' in event:
                    ports.add(event['src_port'])
                if 'dst_port' in event:
                    ports.add(event['dst_port'])
                if 'pid' in event:
                    processes.add(event['pid'])
                if 'syscall' in event:
                    syscalls.add(event['syscall'])
                if 'protocol' in event:
                    protocols.add(event['protocol'])
        
        # Network topology features (12)
        features[0] = len(ip_addresses)  # Unique IP count
        features[1] = len(ports)         # Unique port count
        features[2] = len(protocols)     # Protocol diversity
        
        # Port analysis
        if ports:
            port_list = list(ports)
            features[3] = np.mean(port_list)
            features[4] = np.std(port_list)
            features[5] = len([p for p in port_list if p < 1024])  # Well-known ports
            features[6] = len([p for p in port_list if p >= 1024 and p < 49152])  # Registered ports
            features[7] = len([p for p in port_list if p >= 49152])  # Dynamic ports
        
        # IP address analysis
        if ip_addresses:
            # Private IP ratio
            private_ips = 0
            for ip in ip_addresses:
                if self._is_private_ip(ip):
                    private_ips += 1
            features[8] = private_ips / len(ip_addresses)
            features[9] = len(ip_addresses) - private_ips  # Public IP count
        
        # Network activity concentration
        features[10] = len(event_groups) / max(1, len(ip_addresses))  # Events per IP
        features[11] = len(event_groups) / max(1, len(ports))        # Events per port
        
        # Process hierarchy features (8)
        features[12] = len(processes)    # Process diversity
        features[13] = len(syscalls)     # Syscall diversity
        
        if processes:
            # Process activity distribution
            process_events = defaultdict(int)
            for group in event_groups:
                for event in group.events:
                    if 'pid' in event:
                        process_events[event['pid']] += 1
            
            event_counts = list(process_events.values())
            features[14] = np.mean(event_counts)
            features[15] = np.std(event_counts)
            features[16] = np.max(event_counts)
            features[17] = np.min(event_counts)
            features[18] = len([c for c in event_counts if c > np.mean(event_counts)])  # Active processes
            features[19] = stats.entropy(event_counts) if len(event_counts) > 1 else 0  # Process entropy
        
        # Resource distribution features (8)
        layer_distribution = defaultdict(int)
        for group in event_groups:
            for event in group.events:
                layer = self._identify_layer(event)
                layer_distribution[layer] += 1
        
        if layer_distribution:
            dist_values = list(layer_distribution.values())
            features[20] = len(layer_distribution)  # Layer diversity
            features[21] = np.std(dist_values) / np.mean(dist_values) if np.mean(dist_values) > 0 else 0  # Distribution evenness
            features[22] = layer_distribution.get('network', 0) / sum(dist_values)
            features[23] = layer_distribution.get('filesystem', 0) / sum(dist_values)
            features[24] = layer_distribution.get('process', 0) / sum(dist_values)
            features[25] = layer_distribution.get('system', 0) / sum(dist_values)
            features[26] = max(dist_values) / sum(dist_values)  # Dominant layer ratio
            features[27] = stats.entropy(dist_values)  # Layer entropy
        
        # Cross-layer connectivity features (7)
        net_to_fs_events = 0
        fs_to_net_events = 0
        process_correlations = 0
        
        for group in event_groups:
            layers = set()
            for event in group.events:
                layers.add(self._identify_layer(event))
            
            if 'network' in layers and 'filesystem' in layers:
                net_to_fs_events += 1
            if len(layers) > 1:
                process_correlations += 1
        
        total_groups = len(event_groups)
        features[28] = net_to_fs_events / max(1, total_groups)      # Network-FS correlation ratio
        features[29] = process_correlations / max(1, total_groups)  # Cross-layer correlation ratio
        features[30] = len(set(ip_addresses) & set(map(str, processes)))  # IP-Process overlap
        features[31] = np.mean([group.correlation_score for group in event_groups])  # Average correlation
        features[32] = np.std([group.correlation_score for group in event_groups])   # Correlation variability
        features[33] = len([g for g in event_groups if g.correlation_score > 0.7])   # High correlation events
        features[34] = len([g for g in event_groups if len(g.event_types) > 1])      # Multi-type correlations
        
        return features
    
    def _extract_behavioral_features(self, event_groups: List[CorrelatedEventGroup]) -> np.ndarray:
        """
        Extract behavioral features (50 dimensions):
        - Access patterns (15)
        - Anomaly indicators (15)
        - Communication patterns (10)
        - Resource usage patterns (10)
        """
        features = np.zeros(50)
        
        # Collect behavioral data
        file_accesses = defaultdict(int)
        network_flows = defaultdict(int)
        syscall_patterns = defaultdict(int)
        resource_usage = defaultdict(list)
        
        for group in event_groups:
            for event in group.events:
                if 'filename' in event:
                    file_accesses[event['filename']] += 1
                if 'src_ip' in event and 'dst_ip' in event:
                    flow = f"{event['src_ip']}:{event.get('src_port', 0)}->{event['dst_ip']}:{event.get('dst_port', 0)}"
                    network_flows[flow] += 1
                if 'syscall' in event:
                    syscall_patterns[event['syscall']] += 1
                if 'bytes' in event:
                    resource_usage['bytes'].append(event['bytes'])
                if 'cpu_time' in event:
                    resource_usage['cpu'].append(event['cpu_time'])
        
        # Access patterns (15 features)
        if file_accesses:
            access_counts = list(file_accesses.values())
            features[0] = len(file_accesses)  # File diversity
            features[1] = np.mean(access_counts)
            features[2] = np.std(access_counts)
            features[3] = np.max(access_counts)
            features[4] = len([c for c in access_counts if c > 1])  # Repeated accesses
            features[5] = stats.entropy(access_counts)  # Access entropy
            
            # File type analysis
            file_types = defaultdict(int)
            for filename in file_accesses.keys():
                ext = filename.split('.')[-1] if '.' in filename else 'no_ext'
                file_types[ext] += 1
            features[6] = len(file_types)  # File type diversity
            features[7] = max(file_types.values()) / sum(file_types.values()) if file_types else 0
        
        if network_flows:
            flow_counts = list(network_flows.values())
            features[8] = len(network_flows)  # Flow diversity
            features[9] = np.mean(flow_counts)
            features[10] = np.std(flow_counts)
            features[11] = len([c for c in flow_counts if c > 5])  # High-volume flows
            features[12] = stats.entropy(flow_counts)  # Flow entropy
        
        if syscall_patterns:
            syscall_counts = list(syscall_patterns.values())
            features[13] = len(syscall_patterns)  # Syscall diversity
            features[14] = stats.entropy(syscall_counts)  # Syscall entropy
        
        # Anomaly indicators (15 features)
        # Statistical anomalies
        all_values = []
        for group in event_groups:
            all_values.extend([group.correlation_score, group.duration])
        
        if all_values:
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            features[15] = len([v for v in all_values if abs(v - mean_val) > 2 * std_val])  # Statistical outliers
            features[16] = np.max(all_values) / mean_val if mean_val > 0 else 0  # Max deviation
            features[17] = stats.skew(all_values) if len(all_values) > 2 else 0
            features[18] = stats.kurtosis(all_values) if len(all_values) > 3 else 0
        
        # Behavioral anomalies
        features[19] = len([g for g in event_groups if g.duration > 10.0])  # Long-duration events
        features[20] = len([g for g in event_groups if len(g.events) > 100])  # High-cardinality groups
        features[21] = len([g for g in event_groups if g.correlation_score < 0.1])  # Low-correlation groups
        
        # Rare event detection
        event_types = defaultdict(int)
        for group in event_groups:
            for event_type in group.event_types:
                event_types[event_type] += 1
        
        rare_threshold = np.mean(list(event_types.values())) * 0.1
        features[22] = len([t for t, c in event_types.items() if c < rare_threshold])  # Rare event types
        features[23] = np.std(list(event_types.values())) / np.mean(list(event_types.values())) if np.mean(list(event_types.values())) > 0 else 0
        
        # Temporal anomalies
        timestamps = [group.timestamp for group in event_groups]
        if len(timestamps) > 1:
            time_gaps = np.diff(sorted(timestamps))
            gap_threshold = np.mean(time_gaps) + 2 * np.std(time_gaps)
            features[24] = len([g for g in time_gaps if g > gap_threshold])  # Unusual time gaps
            features[25] = np.max(time_gaps) / np.mean(time_gaps) if np.mean(time_gaps) > 0 else 0
        
        # Sequence anomalies
        features[26] = len([g for g in event_groups if len(g.events) != len(set(e.get('type', '') for e in g.events))])  # Repeated patterns
        features[27] = np.mean([len(g.event_types) for g in event_groups])  # Average type diversity
        features[28] = len(set().union(*[g.event_types for g in event_groups]))  # Total type diversity
        features[29] = len([g for g in event_groups if 'error' in str(g.events).lower()])  # Error events
        
        # Communication patterns (10 features)
        if network_flows:
            # Direction analysis
            inbound = len([f for f in network_flows.keys() if self._is_inbound_flow(f)])
            outbound = len(network_flows) - inbound
            features[30] = inbound / len(network_flows)
            features[31] = outbound / len(network_flows)
            
            # Port usage patterns
            src_ports = set()
            dst_ports = set()
            for flow in network_flows.keys():
                parts = flow.split('->')
                if len(parts) == 2:
                    src_port = parts[0].split(':')[-1]
                    dst_port = parts[1].split(':')[-1]
                    src_ports.add(src_port)
                    dst_ports.add(dst_port)
            
            features[32] = len(src_ports)  # Source port diversity
            features[33] = len(dst_ports)  # Destination port diversity
            features[34] = len(src_ports & dst_ports)  # Bidirectional ports
        
        # Protocol distribution
        protocols = defaultdict(int)
        for group in event_groups:
            for event in group.events:
                if 'protocol' in event:
                    protocols[event['protocol']] += 1
        
        if protocols:
            features[35] = len(protocols)  # Protocol diversity
            features[36] = max(protocols.values()) / sum(protocols.values())  # Dominant protocol ratio
            features[37] = protocols.get('TCP', 0) / sum(protocols.values())
            features[38] = protocols.get('UDP', 0) / sum(protocols.values())
            features[39] = sum(v for k, v in protocols.items() if k not in ['TCP', 'UDP']) / sum(protocols.values())  # Other protocols
        
        # Resource usage patterns (10 features)
        if 'bytes' in resource_usage and resource_usage['bytes']:
            bytes_data = resource_usage['bytes']
            features[40] = np.mean(bytes_data)
            features[41] = np.std(bytes_data)
            features[42] = np.sum(bytes_data)
            features[43] = len([b for b in bytes_data if b > np.mean(bytes_data) + 2 * np.std(bytes_data)])  # Large transfers
            features[44] = len(set(bytes_data)) / len(bytes_data)  # Size diversity
        
        if 'cpu' in resource_usage and resource_usage['cpu']:
            cpu_data = resource_usage['cpu']
            features[45] = np.mean(cpu_data)
            features[46] = np.std(cpu_data)
            features[47] = np.max(cpu_data)
            features[48] = len([c for c in cpu_data if c > 0.8])  # High CPU usage events
        
        # Overall resource intensity
        total_events = sum(len(g.events) for g in event_groups)
        # Count unique processes from current event groups
        unique_pids = set()
        for group in event_groups:
            for event in group.events:
                if 'pid' in event:
                    unique_pids.add(event['pid'])
        features[49] = total_events / max(1, len(unique_pids))  # Events per process
        
        return features
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is in private range"""
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            
            first = int(parts[0])
            second = int(parts[1])
            
            # 10.0.0.0/8
            if first == 10:
                return True
            # 172.16.0.0/12
            if first == 172 and 16 <= second <= 31:
                return True
            # 192.168.0.0/16
            if first == 192 and second == 168:
                return True
            
            return False
        except:
            return False
    
    def _identify_layer(self, event: Dict[str, Any]) -> str:
        """Identify the system layer of an event"""
        if 'src_ip' in event or 'dst_ip' in event or 'protocol' in event:
            return 'network'
        elif 'filename' in event or 'syscall' in event:
            if event.get('syscall', '').startswith(('read', 'write', 'open', 'close')):
                return 'filesystem'
            else:
                return 'system'
        elif 'pid' in event:
            return 'process'
        else:
            return 'unknown'
    
    def _is_inbound_flow(self, flow: str) -> bool:
        """Determine if network flow is inbound"""
        try:
            parts = flow.split('->')
            if len(parts) == 2:
                src_ip = parts[0].split(':')[0]
                return not self._is_private_ip(src_ip)
            return False
        except:
            return False
    
    def _update_event_history(self, event_groups: List[CorrelatedEventGroup]):
        """Update event history for temporal analysis"""
        for group in event_groups:
            self.event_history.append({
                'timestamp': group.timestamp,
                'correlation_score': group.correlation_score,
                'event_count': len(group.events),
                'duration': group.duration,
                'event_types': group.event_types
            })
    
    def get_feature_names(self) -> List[str]:
        """Return names of all 127 features for interpretability"""
        temporal_names = [
            'mean_inter_arrival', 'std_inter_arrival', 'min_inter_arrival', 'max_inter_arrival',
            'median_inter_arrival', 'skew_inter_arrival', 'kurtosis_inter_arrival', 'q1_inter_arrival',
            'q3_inter_arrival', 'event_rate',
            'mean_duration', 'std_duration', 'min_duration', 'max_duration', 'median_duration',
            'skew_duration', 'kurtosis_duration', 'total_duration',
            'temporal_clusters', 'mean_cluster_gap', 'time_span', 'event_density', 'burst_ratio',
            'max_burst_length', 'gap_coefficient_variation', 'outlier_gaps',
            'dominant_freq_idx', 'peak_power', 'avg_power', 'power_variability', 'low_freq_ratio',
            'high_freq_ratio', 'spectral_entropy', 'significant_frequencies',
            'max_autocorr', 'period_estimate', 'short_term_periodicity', 'long_term_periodicity',
            'regularity_coefficient', 'regular_ratio', 'variation_intensity', 'uniqueness_ratio'
        ]
        
        spatial_names = [
            'unique_ips', 'unique_ports', 'protocol_diversity', 'mean_port', 'std_port',
            'wellknown_ports', 'registered_ports', 'dynamic_ports', 'private_ip_ratio',
            'public_ip_count', 'events_per_ip', 'events_per_port',
            'process_diversity', 'syscall_diversity', 'mean_process_events', 'std_process_events',
            'max_process_events', 'min_process_events', 'active_processes', 'process_entropy',
            'layer_diversity', 'distribution_evenness', 'network_layer_ratio', 'filesystem_layer_ratio',
            'process_layer_ratio', 'system_layer_ratio', 'dominant_layer_ratio', 'layer_entropy',
            'net_fs_correlation_ratio', 'cross_layer_correlation_ratio', 'ip_process_overlap',
            'avg_correlation_score', 'correlation_variability', 'high_correlation_events', 'multi_type_correlations'
        ]
        
        behavioral_names = [
            'file_diversity', 'mean_file_accesses', 'std_file_accesses', 'max_file_accesses',
            'repeated_file_accesses', 'file_access_entropy', 'file_type_diversity', 'dominant_file_type_ratio',
            'flow_diversity', 'mean_flow_count', 'std_flow_count', 'high_volume_flows', 'flow_entropy',
            'syscall_diversity', 'syscall_entropy',
            'statistical_outliers', 'max_deviation_ratio', 'value_skewness', 'value_kurtosis',
            'long_duration_events', 'high_cardinality_groups', 'low_correlation_groups', 'rare_event_types',
            'event_type_variation', 'unusual_time_gaps', 'max_gap_ratio', 'repeated_patterns',
            'avg_type_diversity', 'total_type_diversity', 'error_events',
            'inbound_flow_ratio', 'outbound_flow_ratio', 'src_port_diversity', 'dst_port_diversity',
            'bidirectional_ports', 'protocol_diversity_comm', 'dominant_protocol_ratio', 'tcp_ratio',
            'udp_ratio', 'other_protocols_ratio',
            'mean_bytes', 'std_bytes', 'total_bytes', 'large_transfers', 'size_diversity',
            'mean_cpu', 'std_cpu', 'max_cpu', 'high_cpu_events', 'events_per_process'
        ]
        
        return temporal_names + spatial_names + behavioral_names
    
    def reset_cache(self):
        """Reset feature extraction cache"""
        self.feature_cache.clear()
        self.event_history.clear()
        self.port_stats.clear()
        self.process_stats.clear()
        self.flow_patterns.clear()
        logger.info("Feature extraction cache reset")

if __name__ == "__main__":
    # Test feature extractor
    config = {
        'temporal_window': 60.0,
        'spatial_window': 1000,
        'min_correlation_score': 0.3
    }
    
    extractor = CrossLayerFeatureExtractor(config)
    
    # Create sample correlated events
    sample_events = [
        CorrelatedEventGroup(
            events=[
                {'timestamp': time.time(), 'src_ip': '192.168.1.10', 'dst_ip': '10.0.0.1', 'protocol': 'TCP', 'pid': 1234},
                {'timestamp': time.time() + 0.1, 'filename': '/tmp/test.log', 'syscall': 'write', 'pid': 1234}
            ],
            correlation_score=0.8,
            timestamp=time.time(),
            duration=2.5,
            event_types={'network', 'filesystem'}
        )
    ]
    
    features = extractor.extract_features(sample_events)
    feature_names = extractor.get_feature_names()
    
    print(f"Extracted {len(features)} features:")
    for i, (name, value) in enumerate(zip(feature_names[:10], features[:10])):
        print(f"  {i+1:2d}. {name:25s}: {value:.4f}")
    print(f"  ... and {len(features)-10} more features")