"""
CrossLayerGuardian - Advanced Monitoring System
Sistema de monitoramento avançado com métricas detalhadas e dashboard em tempo real.
Integra com EventCorrelator para correlação cross-layer completa.
"""

import time
import threading
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

@dataclass
class SystemMetrics:
    timestamp: float
    correlations_per_second: float
    events_per_second: float
    cpu_usage_percent: float
    memory_usage_mb: float
    network_events_count: int
    syscall_events_count: int
    correlation_window_us: int
    avg_correlation_time_us: float
    precision_score: float
    recall_score: float
    f1_score: float

@dataclass
class CorrelationMetrics:
    total_correlations: int
    pid_correlations: int
    flow_correlations: int
    temporal_correlations: int
    resource_correlations: int
    avg_score: float
    distribution_by_type: Dict[str, int]
    distribution_by_score: Dict[str, int]

class AdvancedMonitor:
    """
    Sistema de monitoramento avançado para CrossLayerGuardian.
    Coleta métricas detalhadas e fornece insights para otimização.
    """
    
    def __init__(self, correlator=None, ring_buffer_manager=None):
        self.correlator = correlator
        self.ring_buffer_manager = ring_buffer_manager
        
        # Histórico de métricas
        self.metrics_history: deque = deque(maxlen=10000)  # 10k samples
        self.correlation_history: deque = deque(maxlen=5000)
        
        # Alertas e thresholds
        self.alert_thresholds = {
            'high_cpu': 85.0,
            'high_memory': 8000.0,  # MB
            'low_correlation_rate': 10.0,  # correlations/sec
            'low_precision': 0.7,
            'high_latency': 100.0  # µs
        }
        
        # Estado de alertas
        self.active_alerts = set()
        self.alert_history = deque(maxlen=1000)
        
        # Threads
        self.running = False
        self.monitor_thread = None
        self.analysis_thread = None
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Inicia monitoramento avançado"""
        self.running = True
        
        # Thread de coleta de métricas
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, name="Monitor")
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Thread de análise e alertas
        self.analysis_thread = threading.Thread(target=self._analysis_loop, name="Analysis")
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        self.logger.info("AdvancedMonitor iniciado")
    
    def stop(self):
        """Para monitoramento"""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2)
        
        self.logger.info("AdvancedMonitor parado")
    
    def _monitoring_loop(self):
        """Loop principal de coleta de métricas"""
        while self.running:
            try:
                # Coletar métricas do sistema
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Coletar métricas de correlação
                if self.correlator:
                    corr_metrics = self._collect_correlation_metrics()
                    self.correlation_history.append(corr_metrics)
                
                time.sleep(1)  # Coleta a cada segundo
                
            except Exception as e:
                if self.running:
                    self.logger.error(f"Erro na coleta de métricas: {e}")
                time.sleep(5)
    
    def _analysis_loop(self):
        """Loop de análise e detecção de anomalias"""
        while self.running:
            try:
                # Análise a cada 10 segundos
                time.sleep(10)
                
                if not self.running:
                    break
                
                # Detectar anomalias
                self._detect_anomalies()
                
                # Gerar alertas
                self._process_alerts()
                
                # Análise de tendências
                self._analyze_trends()
                
            except Exception as e:
                if self.running:
                    self.logger.error(f"Erro na análise: {e}")
                time.sleep(10)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Coleta métricas detalhadas do sistema"""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
        except ImportError:
            cpu_usage = 0.0
            memory_usage_mb = 0.0
        
        # Métricas do correlator
        correlations_per_second = 0.0
        correlation_window_us = 50000
        avg_correlation_time_us = 0.0
        precision_score = 0.0
        recall_score = 0.0
        f1_score = 0.0
        
        if self.correlator:
            stats = self.correlator.get_statistics()
            correlation_window_us = self.correlator.correlation_window_us
            avg_correlation_time_us = stats.get('avg_correlation_time_us', 0.0)
            precision_score = stats.get('precision_score', 0.0)
            recall_score = stats.get('recall_score', 0.0)
            f1_score = stats.get('f1_score', 0.0)
            
            # Calcular correlações por segundo
            if hasattr(self, '_last_correlation_count'):
                time_delta = time.time() - getattr(self, '_last_metrics_time', time.time())
                if time_delta > 0:
                    corr_delta = stats['total_correlations'] - self._last_correlation_count
                    correlations_per_second = corr_delta / time_delta
            
            self._last_correlation_count = stats['total_correlations']
            self._last_metrics_time = time.time()
        
        # Métricas do ring buffer manager
        events_per_second = 0.0
        network_events_count = 0
        syscall_events_count = 0
        
        if self.ring_buffer_manager:
            rb_stats = self.ring_buffer_manager.get_statistics()
            events_per_second = rb_stats.get('events_per_second', 0.0)
            network_events_count = rb_stats.get('network_events_processed', 0)
            syscall_events_count = rb_stats.get('syscall_events_processed', 0)
        
        return SystemMetrics(
            timestamp=time.time(),
            correlations_per_second=correlations_per_second,
            events_per_second=events_per_second,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            network_events_count=network_events_count,
            syscall_events_count=syscall_events_count,
            correlation_window_us=correlation_window_us,
            avg_correlation_time_us=avg_correlation_time_us,
            precision_score=precision_score,
            recall_score=recall_score,
            f1_score=f1_score
        )
    
    def _collect_correlation_metrics(self) -> CorrelationMetrics:
        """Coleta métricas detalhadas de correlação"""
        if not self.correlator:
            return None
        
        stats = self.correlator.get_statistics()
        
        # Distribuição por tipo
        distribution_by_type = {
            'PID_MATCH': stats.get('pid_correlations', 0),
            'FLOW_MATCH': stats.get('flow_correlations', 0),
            'TEMPORAL': stats.get('temporal_correlations', 0),
            'RESOURCE': stats.get('resource_correlations', 0)
        }
        
        # Distribuição por score (seria coletada de correlações recentes)
        distribution_by_score = {
            '0.9-1.0': 0,  # Alta confiança
            '0.7-0.9': 0,  # Média confiança
            '0.5-0.7': 0,  # Baixa confiança
            '0.0-0.5': 0   # Muito baixa
        }
        
        # Calcular score médio
        total = stats.get('total_correlations', 0)
        if total > 0:
            avg_score = (
                (distribution_by_type['PID_MATCH'] * 0.95) +
                (distribution_by_type['FLOW_MATCH'] * 0.85) +
                (distribution_by_type['TEMPORAL'] * 0.65) +
                (distribution_by_type['RESOURCE'] * 0.55)
            ) / total
        else:
            avg_score = 0.0
        
        return CorrelationMetrics(
            total_correlations=total,
            pid_correlations=distribution_by_type['PID_MATCH'],
            flow_correlations=distribution_by_type['FLOW_MATCH'],
            temporal_correlations=distribution_by_type['TEMPORAL'],
            resource_correlations=distribution_by_type['RESOURCE'],
            avg_score=avg_score,
            distribution_by_type=distribution_by_type,
            distribution_by_score=distribution_by_score
        )
    
    def _detect_anomalies(self):
        """Detecta anomalias baseado em métricas históricas"""
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = list(self.metrics_history)[-10:]  # Últimos 10 samples
        
        # Detectar anomalias de CPU
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        if avg_cpu > self.alert_thresholds['high_cpu']:
            self._trigger_alert('high_cpu', f"CPU usage: {avg_cpu:.1f}%")
        else:
            self._clear_alert('high_cpu')
        
        # Detectar anomalias de memória
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        if avg_memory > self.alert_thresholds['high_memory']:
            self._trigger_alert('high_memory', f"Memory usage: {avg_memory:.0f}MB")
        else:
            self._clear_alert('high_memory')
        
        # Detectar baixa taxa de correlação
        avg_corr_rate = sum(m.correlations_per_second for m in recent_metrics) / len(recent_metrics)
        if avg_corr_rate < self.alert_thresholds['low_correlation_rate']:
            self._trigger_alert('low_correlation_rate', f"Correlation rate: {avg_corr_rate:.1f}/s")
        else:
            self._clear_alert('low_correlation_rate')
        
        # Detectar baixa precisão
        recent_precision = [m.precision_score for m in recent_metrics if m.precision_score > 0]
        if recent_precision:
            avg_precision = sum(recent_precision) / len(recent_precision)
            if avg_precision < self.alert_thresholds['low_precision']:
                self._trigger_alert('low_precision', f"Precision: {avg_precision:.3f}")
            else:
                self._clear_alert('low_precision')
        
        # Detectar alta latência de correlação
        recent_latency = [m.avg_correlation_time_us for m in recent_metrics if m.avg_correlation_time_us > 0]
        if recent_latency:
            avg_latency = sum(recent_latency) / len(recent_latency)
            if avg_latency > self.alert_thresholds['high_latency']:
                self._trigger_alert('high_latency', f"Avg latency: {avg_latency:.1f}µs")
            else:
                self._clear_alert('high_latency')
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Dispara um alerta"""
        if alert_type not in self.active_alerts:
            self.active_alerts.add(alert_type)
            
            alert = {
                'timestamp': time.time(),
                'type': alert_type,
                'message': message,
                'status': 'triggered'
            }
            
            self.alert_history.append(alert)
            self.logger.warning(f"ALERT: {alert_type} - {message}")
    
    def _clear_alert(self, alert_type: str):
        """Limpa um alerta"""
        if alert_type in self.active_alerts:
            self.active_alerts.remove(alert_type)
            
            alert = {
                'timestamp': time.time(),
                'type': alert_type,
                'message': 'Alert cleared',
                'status': 'cleared'
            }
            
            self.alert_history.append(alert)
            self.logger.info(f"Alert cleared: {alert_type}")
    
    def _process_alerts(self):
        """Processa alertas ativos e toma ações"""
        for alert_type in list(self.active_alerts):
            if alert_type == 'high_cpu' and self.correlator:
                # Reduzir janela de correlação para diminuir CPU
                current_window = self.correlator.correlation_window_us
                new_window = max(5000, current_window * 0.9)
                if new_window != current_window:
                    self.correlator.correlation_window_us = new_window
                    self.correlator.correlation_window_tsc = new_window * 3000
                    self.logger.info(f"Auto-ajuste: janela reduzida para {new_window}µs devido alto CPU")
            
            elif alert_type == 'low_correlation_rate' and self.correlator:
                # Aumentar janela para capturar mais correlações
                current_window = self.correlator.correlation_window_us
                new_window = min(200000, current_window * 1.1)
                if new_window != current_window:
                    self.correlator.correlation_window_us = new_window
                    self.correlator.correlation_window_tsc = new_window * 3000
                    self.logger.info(f"Auto-ajuste: janela aumentada para {new_window}µs devido baixa taxa")
    
    def _analyze_trends(self):
        """Analisa tendências de longo prazo"""
        if len(self.metrics_history) < 60:  # Precisa de pelo menos 1 minuto de dados
            return
        
        recent = list(self.metrics_history)[-60:]  # Último minuto
        older = list(self.metrics_history)[-120:-60] if len(self.metrics_history) >= 120 else []
        
        if not older:
            return
        
        # Tendência de CPU
        recent_cpu = sum(m.cpu_usage_percent for m in recent) / len(recent)
        older_cpu = sum(m.cpu_usage_percent for m in older) / len(older)
        cpu_trend = recent_cpu - older_cpu
        
        # Tendência de correlações
        recent_corr = sum(m.correlations_per_second for m in recent) / len(recent)
        older_corr = sum(m.correlations_per_second for m in older) / len(older)
        corr_trend = recent_corr - older_corr
        
        # Log tendências significativas
        if abs(cpu_trend) > 10:
            direction = "aumentando" if cpu_trend > 0 else "diminuindo"
            self.logger.info(f"Tendência CPU: {direction} {abs(cpu_trend):.1f}%")
        
        if abs(corr_trend) > 5:
            direction = "aumentando" if corr_trend > 0 else "diminuindo"
            self.logger.info(f"Tendência correlações: {direction} {abs(corr_trend):.1f}/s")
    
    def get_current_status(self) -> dict:
        """Retorna status atual detalhado"""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        latest_metrics = self.metrics_history[-1]
        latest_correlation = self.correlation_history[-1] if self.correlation_history else None
        
        return {
            'timestamp': latest_metrics.timestamp,
            'system_metrics': asdict(latest_metrics),
            'correlation_metrics': asdict(latest_correlation) if latest_correlation else None,
            'active_alerts': list(self.active_alerts),
            'alert_count': len(self.active_alerts),
            'status': 'healthy' if not self.active_alerts else 'alerts_active'
        }
    
    def get_performance_report(self) -> dict:
        """Gera relatório de performance detalhado"""
        if len(self.metrics_history) < 10:
            return {'error': 'insufficient_data'}
        
        recent_metrics = list(self.metrics_history)[-60:]  # Último minuto
        
        # Estatísticas agregadas
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_corr_rate = sum(m.correlations_per_second for m in recent_metrics) / len(recent_metrics)
        avg_event_rate = sum(m.events_per_second for m in recent_metrics) / len(recent_metrics)
        
        # Performance scores (0-100)
        cpu_score = max(0, 100 - avg_cpu)  # Menos CPU = melhor
        memory_score = max(0, 100 - (avg_memory / 100))  # Menos memória = melhor
        correlation_score = min(100, avg_corr_rate * 2)  # Mais correlações = melhor
        
        overall_score = (cpu_score + memory_score + correlation_score) / 3
        
        return {
            'timestamp': time.time(),
            'period_minutes': len(recent_metrics) / 60,
            'performance_scores': {
                'overall': overall_score,
                'cpu': cpu_score,
                'memory': memory_score,
                'correlation': correlation_score
            },
            'averages': {
                'cpu_usage_percent': avg_cpu,
                'memory_usage_mb': avg_memory,
                'correlations_per_second': avg_corr_rate,
                'events_per_second': avg_event_rate
            },
            'alert_summary': {
                'active_count': len(self.active_alerts),
                'total_triggered': len([a for a in self.alert_history if a['status'] == 'triggered']),
                'most_recent': list(self.alert_history)[-5:] if self.alert_history else []
            }
        }