# CrossLayerGuardian

## Arquitetura eBPF para Correlação Cross-layer em Sistemas de Detecção de Intrusão

O CrossLayerGuardian implementa a primeira arquitetura eBPF com correlação PID-flow sub-10µs, resolvendo limitações fundamentais de sincronização entre domínios NIDS/HIDS através de correlação cross-layer eficiente.

### Características Principais

- **Correlação Cross-layer Avançada**: Sincronização TSC <2,3µs com 3 fatores (temporal, causal, resource)
- **Performance Otimizada**: Throughput 850 Mbps com overhead <8% CPU
- **Socket Tracking Inteligente**: Mapeamento processo-conexão O(1) para 100K conexões
- **Ring Buffers Lock-free**: Comunicação kernel-userspace preservando ordem temporal
- **Janelas Adaptativas**: Auto-ajuste baseado em feedback e características do ambiente
- **Monitoramento Avançado**: Sistema de alertas inteligentes e métricas detalhadas
- **Ensemble ML Adaptativo**: XGBoost + MLP com weight update α=0.3

### Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    UserSpace                                │
├─────────────────┬─────────────────┬─────────────────────────┤
│  EventCorrelator│   ML Ensemble   │     Web Interface       │
│  - PID-flow     │  - XGBoost      │    - Flask App          │
│  - TSC sync     │  - MLP          │    - Statistics         │
│  - Correlação   │  - Weight α=0.3 │    - Dashboard          │
├─────────────────┴─────────────────┴─────────────────────────┤
│                Ring Buffers (Lock-free)                     │
├─────────────────────────────────────────────────────────────┤
│                    Kernel Space                             │
├─────────────────┬───────────────────────────────────────────┤
│   NetMonitor    │              FileMonitor                  │
│   - XDP 24M pps │          - VFS kprobes                    │
│   - TCP kprobes │          - Process lifecycle              │
│   - Flow track  │          - 47 syscalls críticas           │
└─────────────────┴───────────────────────────────────────────┘
```

### Instalação

#### Dependências do Sistema

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    clang \
    llvm \
    linux-headers-$(uname -r) \
    python3-dev \
    python3-pip

# Instalar BCC (Berkeley Packet Filter Compiler Collection)
sudo apt-get install -y bpfcc-tools linux-headers-$(uname -r)
```

#### Dependências Python

```bash
pip3 install -r requirements.txt
```

#### Compilar Programas eBPF

```bash
./build_ebpf.sh
```

### Execução

```bash
# Executar sistema completo (requer root)
sudo python3 main.py

# Executar apenas interface web (desenvolvimento)
python3 app.py
```

### Monitoramento

O sistema coleta automaticamente:

- **Eventos de Rede**: XDP capture com correlação PID-flow
- **Eventos de Sistema**: VFS operations, process lifecycle
- **Correlações**: Temporal, causal e resource-based
- **Estatísticas**: Performance, throughput, overhead

### Configuração

#### Janela de Correlação

A janela padrão é 50ms, mas se adapta automaticamente baseado no feedback:

```python
# Reduz janela se alta precisão (>90%)
correlation_window = max(5ms, current_window * 0.9)

# Aumenta janela se baixa precisão (<70%)
correlation_window = min(200ms, current_window * 1.1)
```

#### Ensemble ML

Pesos adaptativos baseados em F1-score windowed:

```python
P_final = w1 * P_XGBoost + w2 * P_MLP
# onde w1 + w2 = 1, α = 0.3 (learning rate)
```

### Performance

Baseado na validação experimental da dissertação:

| Métrica | Valor | Baseline (Suricata+OSSEC) | Melhoria |
|---------|-------|--------------------------|----------|
| Throughput | 850 Mbps | 520 Mbps | +63% |
| CPU Overhead | 7,8% ±0,4% | 41% | -81% |
| F1-Score | 0,947 | 0,593 | +60% |
| Detecção Multi-vetor | 95% TPR | 52% (sistemas isolados) | +83% |
| Falsos Positivos | 1,2% | 2,8% | -57% |
| Correlação | <10µs | 50-200ms (fragmentados) | -99% |
| Precisão TSC | 2,3µs ±0,8µs | 10-100µs (timestamps app) | -90% |
| Adaptação Janela | 5-200ms | Fixo 50ms | Dinâmico |

### Validação

O sistema foi validado com:

- **CICIDS2018**: Dataset padrão para IDS
- **Cenários APT**: APT28 e Carbanak
- **Ambientes Controlados**: Testbed dedicado
- **Produção**: Tráfego real 24h contínuas

### Debugging

#### Logs

```bash
# Logs do sistema
tail -f /var/log/crosslayer.log

# Debug correlação
export CROSSLAYER_DEBUG=1
sudo python3 main.py
```

#### Estatísticas em Tempo Real

```python
from ring_buffer_manager import create_ring_buffer_manager

manager = create_ring_buffer_manager()
stats = manager.get_statistics()
print(f"Events/sec: {stats['events_per_second']}")
print(f"Correlations: {stats['correlation_stats']['total_correlations']}")
```

### Componentes Principais

#### 1. Sincronização TSC (`crosslayer_common.h`)
- `normalize_tsc_timestamp()`: Normalização TSC multi-CPU
- Compensação drift <0.1ppm
- Precisão temporal <2,3µs entre domínios

#### 2. NetMonitor (`net_monitor.bpf.c`)
- Programa XDP para filtragem 24M pacotes/s
- Kprobes TCP para socket tracking avançado
- Hash tables otimizadas para PID-flow mapping

#### 3. FileMonitor (`file_monitor.bpf.c`)
- Kprobes VFS para 47 syscalls críticas
- Process lifecycle tracking completo
- Correlação temporal com eventos de rede

#### 4. EventCorrelator (`event_correlator.py`)
- Algoritmos de correlação multi-dimensionais:
  - Temporal (w1=0.4): TSC drift compensation
  - Causal (w2=0.4): PID + socket tracking + flow states
  - Resource (w3=0.2): Semantic analysis + IP/port correlation
- Janelas adaptativas: 5ms-200ms baseado em feedback
- Múltiplas estratégias de busca: PID, Flow, Temporal, Semântica
- Performance otimizada: O(log n) lookup, deduplicação

#### 5. RingBufferManager (`ring_buffer_manager.py`)
- Interface Python para ring buffers eBPF
- Comunicação lock-free preservando ordem
- Event deserialization eficiente

#### 6. AdvancedMonitor (`advanced_monitor.py`)
- Coleta de métricas em tempo real
- Detecção automática de anomalias
- Alertas adaptativos com auto-resolução
- Auto-tuning baseado em performance

### Referência

Este código implementa a arquitetura descrita na dissertação de mestrado:

> Arioza, Daniel. *CrossLayerGuardian: Arquitetura eBPF para Correlação Cross-layer em Sistemas de Detecção de Intrusão*. Dissertação de Mestrado — PPGC/UFRGS, Porto Alegre, 2026.

Contribuições técnicas reutilizáveis:
1. TSC normalization algorithm para sistemas distribuídos
2. Ring buffer lock-free design para observability
3. Ensemble adaptativo para ML temporal
4. Socket tracking O(1) para network forensics

### Licença

MIT License. Consulte o arquivo [LICENSE](LICENSE) para detalhes.

### Links

- Dissertação completa: [Link para dissertação]
- Dataset CICIDS2018: [https://www.unb.ca/cic/datasets/ids-2018.html](https://www.unb.ca/cic/datasets/ids-2018.html)
