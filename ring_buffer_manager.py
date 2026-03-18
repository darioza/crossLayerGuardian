"""
CrossLayerGuardian - Ring Buffer Communication
Interface Python para comunicação com ring buffers eBPF lock-free.
Implementa leitura eficiente de eventos de NetMonitor e FileMonitor.
"""

import struct
import time
import threading
from typing import Callable, Optional
import logging
from ctypes import Structure, c_uint64, c_uint32, c_uint16, c_uint8, c_char
from bcc import BPF
from event_correlator import NetworkEvent, SyscallEvent, get_correlator

class FlowKey(Structure):
    """Estrutura equivalente à flow_key do eBPF"""
    _fields_ = [
        ("src_ip", c_uint32),
        ("dst_ip", c_uint32), 
        ("src_port", c_uint16),
        ("dst_port", c_uint16),
        ("protocol", c_uint8)
    ]

class NetworkEventC(Structure):
    """Estrutura equivalente à network_event do eBPF"""
    _fields_ = [
        ("timestamp", c_uint64),
        ("flow_id", c_uint32),
        ("pid", c_uint32),
        ("event_type", c_uint16),
        ("severity", c_uint16),
        ("key", FlowKey)
    ]

class SyscallEventC(Structure):
    """Estrutura equivalente à syscall_event do eBPF"""
    _fields_ = [
        ("timestamp", c_uint64),
        ("pid", c_uint32),
        ("flow_id", c_uint32),
        ("syscall_id", c_uint16),
        ("severity", c_uint16),
        ("target_path", c_char * 64)
    ]

class RingBufferManager:
    """
    Gerenciador de ring buffers para comunicação kernel-userspace eficiente.
    Implementa leitura lock-free de eventos com preservação de ordem temporal.
    """
    
    def __init__(self, net_program_path: str, file_program_path: str):
        self.logger = logging.getLogger(__name__)
        self.correlator = get_correlator()
        
        # Carregar programas eBPF
        try:
            self.net_bpf = BPF(src_file=net_program_path)
            self.file_bpf = BPF(src_file=file_program_path)
        except Exception as e:
            self.logger.error(f"Erro ao carregar programas eBPF: {e}")
            raise
        
        # Ring buffers
        self.network_rb = None
        self.syscall_rb = None
        
        # Threads de processamento
        self.running = False
        self.net_thread = None
        self.file_thread = None
        
        # Estatísticas
        self.stats = {
            'network_events_processed': 0,
            'syscall_events_processed': 0,
            'events_per_second': 0,
            'last_stats_time': time.time()
        }
    
    def start(self):
        """Inicia processamento dos ring buffers"""
        try:
            # Anexar programas XDP e kprobes
            self._attach_programs()
            
            # Configurar ring buffers
            self._setup_ring_buffers()
            
            # Iniciar correlador
            self.correlator.start()
            
            # Iniciar threads de processamento
            self.running = True
            self.net_thread = threading.Thread(target=self._process_network_events)
            self.file_thread = threading.Thread(target=self._process_syscall_events)
            
            self.net_thread.daemon = True
            self.file_thread.daemon = True
            
            self.net_thread.start()
            self.file_thread.start()
            
            self.logger.info("RingBufferManager iniciado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar RingBufferManager: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Para processamento dos ring buffers"""
        self.running = False
        
        # Parar threads
        if self.net_thread:
            self.net_thread.join(timeout=2)
        if self.file_thread:
            self.file_thread.join(timeout=2)
        
        # Parar correlador
        self.correlator.stop()
        
        # Limpar programas eBPF
        if hasattr(self, 'net_bpf'):
            self.net_bpf.cleanup()
        if hasattr(self, 'file_bpf'):
            self.file_bpf.cleanup()
        
        self.logger.info("RingBufferManager parado")
    
    def _attach_programs(self):
        """Anexa programas eBPF aos hooks apropriados"""
        try:
            # Anexar XDP program para network monitoring
            interface = "lo"  # Pode ser configurável
            self.net_bpf.attach_xdp(interface, self.net_bpf.load_func("xdp_correlator", BPF.XDP))
            self.logger.info(f"Programa XDP anexado à interface {interface}")
            
            # Anexar kprobes para file monitoring  
            self.file_bpf.attach_kprobe(event="vfs_read", fn_name="handle_vfs_read")
            self.file_bpf.attach_kprobe(event="vfs_write", fn_name="handle_vfs_write")
            
            # Anexar tracepoints
            self.file_bpf.attach_tracepoint(tp="sched:sched_process_fork", fn_name="handle_process_fork")
            self.file_bpf.attach_tracepoint(tp="sched:sched_process_exit", fn_name="handle_process_exit")
            
            self.logger.info("Programas eBPF anexados com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao anexar programas eBPF: {e}")
            raise
    
    def _setup_ring_buffers(self):
        """Configura ring buffers para leitura de eventos"""
        try:
            # Ring buffer para eventos de rede
            self.network_rb = self.net_bpf.get_table("network_events")
            self.network_rb.open_ring_buffer(self._handle_network_event)
            
            # Ring buffer para eventos de syscall
            self.syscall_rb = self.file_bpf.get_table("syscall_events") 
            self.syscall_rb.open_ring_buffer(self._handle_syscall_event)
            
            self.logger.info("Ring buffers configurados")
            
        except Exception as e:
            self.logger.error(f"Erro ao configurar ring buffers: {e}")
            raise
    
    def _handle_network_event(self, ctx, data, size):
        """Callback para processar eventos de rede do ring buffer"""
        try:
            # Deserializar evento
            event_c = NetworkEventC.from_buffer_copy(data)
            
            # Converter para formato Python
            event = NetworkEvent(
                timestamp=event_c.timestamp,
                flow_id=event_c.flow_id,
                pid=event_c.pid,
                event_type=event_c.event_type,
                severity=event_c.severity,
                src_ip=self._ip_to_string(event_c.key.src_ip),
                dst_ip=self._ip_to_string(event_c.key.dst_ip),
                src_port=event_c.key.src_port,
                dst_port=event_c.key.dst_port,
                protocol=event_c.key.protocol
            )
            
            # Enviar para correlador
            self.correlator.add_network_event(event)
            
            # Atualizar estatísticas
            self.stats['network_events_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"Erro ao processar evento de rede: {e}")
    
    def _handle_syscall_event(self, ctx, data, size):
        """Callback para processar eventos de syscall do ring buffer"""
        try:
            # Deserializar evento
            event_c = SyscallEventC.from_buffer_copy(data)
            
            # Converter para formato Python
            event = SyscallEvent(
                timestamp=event_c.timestamp,
                pid=event_c.pid,
                flow_id=event_c.flow_id,
                syscall_id=event_c.syscall_id,
                severity=event_c.severity,
                target_path=event_c.target_path.decode('utf-8', errors='ignore')
            )
            
            # Enviar para correlador
            self.correlator.add_syscall_event(event)
            
            # Atualizar estatísticas
            self.stats['syscall_events_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"Erro ao processar evento de syscall: {e}")
    
    def _process_network_events(self):
        """Thread para processar eventos de rede continuamente"""
        while self.running:
            try:
                self.network_rb.ring_buffer_poll(timeout=100)  # 100ms timeout
            except Exception as e:
                if self.running:  # Só log se ainda estiver rodando
                    self.logger.error(f"Erro no processamento de eventos de rede: {e}")
                time.sleep(0.1)
    
    def _process_syscall_events(self):
        """Thread para processar eventos de syscall continuamente"""
        while self.running:
            try:
                self.syscall_rb.ring_buffer_poll(timeout=100)  # 100ms timeout
            except Exception as e:
                if self.running:  # Só log se ainda estiver rodando
                    self.logger.error(f"Erro no processamento de eventos de syscall: {e}")
                time.sleep(0.1)
    
    def _ip_to_string(self, ip_int: int) -> str:
        """Converte IP integer para string"""
        return f"{ip_int & 0xFF}.{(ip_int >> 8) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 24) & 0xFF}"
    
    def get_statistics(self) -> dict:
        """Retorna estatísticas de processamento"""
        current_time = time.time()
        time_delta = current_time - self.stats['last_stats_time']
        
        if time_delta > 0:
            total_events = (self.stats['network_events_processed'] + 
                           self.stats['syscall_events_processed'])
            self.stats['events_per_second'] = total_events / time_delta
        
        # Adicionar estatísticas do correlador
        correlation_stats = self.correlator.get_statistics()
        
        return {
            **self.stats,
            'correlation_stats': correlation_stats,
            'last_update': current_time
        }
    
    def reset_statistics(self):
        """Reseta estatísticas de processamento"""
        self.stats = {
            'network_events_processed': 0,
            'syscall_events_processed': 0,
            'events_per_second': 0,
            'last_stats_time': time.time()
        }


def create_ring_buffer_manager(net_prog_path: str = None, file_prog_path: str = None) -> RingBufferManager:
    """Factory function para criar RingBufferManager com paths padrão"""
    import os
    
    base_path = os.path.dirname(__file__)
    
    if net_prog_path is None:
        net_prog_path = os.path.join(base_path, "data_collection/ebpf_programs/net_monitor.bpf.c")
    
    if file_prog_path is None:
        file_prog_path = os.path.join(base_path, "data_collection/ebpf_programs/file_monitor.bpf.c")
    
    return RingBufferManager(net_prog_path, file_prog_path)