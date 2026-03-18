"""
CrossLayerGuardian Configuration Loader
Loads and parses configuration from config.ini file
Provides type-safe configuration access for all system components
"""

import configparser
import os
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigurationLoader:
    """
    Centralized configuration loader for CrossLayerGuardian
    Handles parsing and type conversion from config.ini
    """
    
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser()
        self.loaded_config = {}
        
        if not self.config_file.exists():
            logger.warning(f"Configuration file {config_file} not found, using defaults")
            self._create_default_config()
        else:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            self.config.read(self.config_file)
            self._parse_all_sections()
            logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration if file doesn't exist"""
        logger.info("Using default configuration")
        self.loaded_config = self._get_default_config()
    
    def _parse_all_sections(self):
        """Parse all configuration sections"""
        self.loaded_config = {}
        
        # Parse each section
        for section_name in self.config.sections():
            section_dict = {}
            for key, value in self.config.items(section_name):
                section_dict[key] = self._parse_value(value)
            self.loaded_config[section_name.lower()] = section_dict
        
        # Flatten ML configuration for convenience
        self._create_ml_config()
    
    def _parse_value(self, value: str) -> Union[str, int, float, bool, List]:
        """Parse configuration value with appropriate type conversion"""
        if not value:
            return ""
        
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # List values (comma-separated)
        if ',' in value:
            items = [item.strip() for item in value.split(',')]
            # Try to convert to numbers if possible
            try:
                return [int(item) for item in items]
            except ValueError:
                try:
                    return [float(item) for item in items]
                except ValueError:
                    return items
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String values
        return value
    
    def _create_ml_config(self):
        """Create unified ML configuration"""
        ml_config = {}
        
        # Merge all ML-related sections
        ml_sections = ['ml_ensemble', 'ml_xgboost', 'ml_mlp', 'ml_feature_extraction',
                      'ml_integration', 'ml_training', 'ml_paths', 'ml_performance']
        
        for section in ml_sections:
            if section in self.loaded_config:
                for key, value in self.loaded_config[section].items():
                    # Add section prefix to avoid key conflicts
                    prefixed_key = f"{section.split('_', 1)[1]}_{key}" if '_' in section else key
                    ml_config[prefixed_key] = value

        # Aliases para compatibilidade com xgboost_classifier.py (usa prefixo 'xgb_')
        xgb_aliases = {
            'xgboost_max_depth':       'xgb_max_depth',
            'xgboost_learning_rate':   'xgb_learning_rate',
            'xgboost_n_estimators':    'xgb_n_estimators',
            'xgboost_subsample':       'xgb_subsample',
            'xgboost_colsample_bytree':'xgb_colsample_bytree',
            'xgboost_reg_alpha':       'xgb_reg_alpha',
            'xgboost_reg_lambda':      'xgb_reg_lambda',
        }
        for src, dst in xgb_aliases.items():
            if src in ml_config:
                ml_config[dst] = ml_config[src]

        # Add common parameters
        ml_config.update({
            'random_state': 42,
            'n_jobs': -1,
            'verbose': False
        })
        
        self.loaded_config['ml_unified'] = ml_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'system': {
                'log_level': 'INFO',
                'max_events_per_second': 50000,
                'enable_tsc_sync': True,
                'tsc_sync_precision_us': 2.3
            },
            'ebpf': {
                'ring_buffer_size': 8388608,
                'max_entries': 100000
            },
            'correlation': {
                'min_correlation_score': 0.3,
                'temporal_window_ms': 100,
                'adaptive_windows_enabled': True
            },
            'ml_unified': {
                # Ensemble parameters
                'ensemble_alpha': 0.3,
                'confidence_threshold': 0.7,
                'decision_strategy': 'confidence_weighted',
                
                # XGBoost parameters
                'xgboost_max_depth': 6,
                'xgboost_learning_rate': 0.1,
                'xgboost_n_estimators': 50,
                'xgb_n_estimators': 50,
                'xgboost_subsample': 0.8,

                # MLP parameters
                'mlp_hidden_layers': [256, 128, 64],
                'mlp_learning_rate': 0.001,
                'mlp_batch_size': 32,
                'mlp_epochs': 100,
                
                # Integration parameters
                'integration_batch_size': 32,
                'integration_max_queue_size': 1000,
                'integration_workers': 2,
                'integration_alert_threshold': 0.7,
                
                # Paths
                'paths_model_dir': 'models',
                'paths_data_dir': 'data',
                'paths_report_dir': 'reports',
                
                'random_state': 42
            }
        }
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.loaded_config.get(section_name.lower(), {})
    
    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get specific configuration value"""
        section_config = self.get_section(section)
        return section_config.get(key.lower(), default)
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get unified ML configuration"""
        return self.loaded_config.get('ml_unified', {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.get_section('system')
    
    def get_ebpf_config(self) -> Dict[str, Any]:
        """Get eBPF configuration"""
        return self.get_section('ebpf')
    
    def get_correlation_config(self) -> Dict[str, Any]:
        """Get correlation configuration"""
        return self.get_section('correlation')
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get_section('monitoring')
    
    def get_alerts_config(self) -> Dict[str, Any]:
        """Get alerts configuration"""
        return self.get_section('alerts')
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """Update configuration values"""
        if section.lower() not in self.loaded_config:
            self.loaded_config[section.lower()] = {}
        
        self.loaded_config[section.lower()].update(updates)
        
        # Re-create ML config if ML section updated
        if section.lower().startswith('ml_'):
            self._create_ml_config()
        
        logger.info(f"Configuration updated for section: {section}")
    
    def save_config(self, output_file: Optional[str] = None):
        """Save current configuration to file"""
        if output_file is None:
            output_file = self.config_file
        
        # Recreate config parser from loaded config
        new_config = configparser.ConfigParser()
        
        for section_name, section_data in self.loaded_config.items():
            if section_name == 'ml_unified':  # Skip unified section
                continue
                
            new_config.add_section(section_name.upper())
            for key, value in section_data.items():
                if isinstance(value, list):
                    value_str = ','.join(map(str, value))
                else:
                    value_str = str(value)
                new_config.set(section_name.upper(), key, value_str)
        
        with open(output_file, 'w') as f:
            new_config.write(f)
        
        logger.info(f"Configuration saved to {output_file}")
    
    def validate_ml_config(self) -> List[str]:
        """Validate ML configuration and return any issues"""
        issues = []
        ml_config = self.get_ml_config()
        
        # Check required ML parameters
        required_params = [
            'ensemble_alpha', 'confidence_threshold', 'decision_strategy',
            'xgboost_max_depth', 'xgboost_learning_rate', 'xgboost_n_estimators',
            'mlp_hidden_layers', 'mlp_learning_rate', 'mlp_batch_size'
        ]
        
        for param in required_params:
            if param not in ml_config:
                issues.append(f"Missing required ML parameter: {param}")
        
        # Validate parameter ranges
        if ml_config.get('ensemble_alpha', 0) <= 0 or ml_config.get('ensemble_alpha', 0) >= 1:
            issues.append("ensemble_alpha must be between 0 and 1")
        
        if ml_config.get('confidence_threshold', 0) <= 0 or ml_config.get('confidence_threshold', 0) > 1:
            issues.append("confidence_threshold must be between 0 and 1")
        
        if ml_config.get('xgboost_max_depth', 0) <= 0:
            issues.append("xgboost_max_depth must be positive")
        
        if ml_config.get('mlp_learning_rate', 0) <= 0:
            issues.append("mlp_learning_rate must be positive")
        
        # Validate paths
        paths_to_check = ['paths_model_dir', 'paths_data_dir', 'paths_report_dir']
        for path_key in paths_to_check:
            if path_key in ml_config:
                path_value = ml_config[path_key]
                if not isinstance(path_value, str) or not path_value:
                    issues.append(f"Invalid path configuration: {path_key}")
        
        return issues
    
    def get_performance_targets(self) -> Dict[str, Any]:
        """Get performance targets from dissertation"""
        perf_config = self.get_section('ml_performance')
        return {
            'target_throughput_mbps': perf_config.get('target_throughput_mbps', 850),
            'max_cpu_overhead_percent': perf_config.get('max_cpu_overhead_percent', 8),
            'max_memory_overhead_mb': perf_config.get('max_memory_overhead_mb', 100),
            'target_latency_us': perf_config.get('target_latency_us', 10),
            'correlation_precision_us': perf_config.get('correlation_precision_us', 2.3),
            'max_prediction_time_ms': perf_config.get('max_prediction_time_ms', 50)
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        sections = list(self.loaded_config.keys())
        return f"ConfigurationLoader(sections={sections}, config_file={self.config_file})"

# Global configuration instance
_config_loader = None

def get_config_loader(config_file: str = "config.ini") -> ConfigurationLoader:
    """Get global configuration loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigurationLoader(config_file)
    return _config_loader

def get_ml_config() -> Dict[str, Any]:
    """Get ML configuration (convenience function)"""
    return get_config_loader().get_ml_config()

def get_system_config() -> Dict[str, Any]:
    """Get system configuration (convenience function)"""
    return get_config_loader().get_system_config()

if __name__ == "__main__":
    # Test configuration loader
    loader = ConfigurationLoader()
    
    print("=== Configuration Loader Test ===")
    print(f"Loader: {loader}")
    
    # Test ML config
    ml_config = loader.get_ml_config()
    print(f"\nML Config keys: {list(ml_config.keys())}")
    print(f"Ensemble alpha: {ml_config.get('ensemble_alpha')}")
    print(f"XGBoost max_depth: {ml_config.get('xgboost_max_depth')}")
    print(f"MLP hidden layers: {ml_config.get('mlp_hidden_layers')}")
    
    # Validate ML config
    issues = loader.validate_ml_config()
    if issues:
        print(f"\nConfiguration issues: {issues}")
    else:
        print("\n✅ ML configuration is valid")
    
    # Performance targets
    perf_targets = loader.get_performance_targets()
    print(f"\nPerformance targets: {perf_targets}")
    
    print("\nConfiguration loader test completed")