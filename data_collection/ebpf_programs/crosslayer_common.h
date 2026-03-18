/*
 * CrossLayerGuardian - Common Headers for Cross-layer Correlation
 * 
 * Estruturas e funcoes baseadas na dissertacao para correlacao cross-layer
 * com sincronizacao TSC precisa <2.3µs e mapeamento PID-flow eficiente
 */

#ifndef __CROSSLAYER_COMMON_H__
#define __CROSSLAYER_COMMON_H__

#include <linux/types.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#define MAX_PATH_LEN 256
#define MAX_PID 4194304  // 2^22 PIDs maximum
#define MAX_CONNECTIONS 100000  // 100K connections support

// Estruturas de dados para correlacao cross-layer (conforme dissertacao)

struct flow_key {
    __be32 src_ip;
    __be32 dst_ip;
    __be16 src_port;
    __be16 dst_port;
    __u8 protocol;
} __attribute__((packed));

struct network_event {
    __u64 timestamp;        // TSC normalizado
    __u32 flow_id;         // Identificador para correlação
    __u32 pid;             // Processo associado
    __u16 event_type;
    __u16 severity;
    struct flow_key key;
} __attribute__((packed));

struct syscall_event {
    __u64 timestamp;        // TSC normalizado  
    __u32 pid;             // Chave de correlação
    __u32 flow_id;         // Conexão associada (se aplicável)
    __u16 syscall_id;
    __u16 severity;
    char target_path[64];   // Recurso acessado
} __attribute__((packed));

// Estrutura para sincronização TSC multi-CPU
struct cpu_sync_info {
    __u64 base_tsc;
    __s64 offset_ns;        // Compensação drift <0.1ppm
    __u32 freq_khz;
    __u8 calibrated;
} __attribute__((aligned(64)));

// Estrutura para correlação PID-Flow
struct pid_flow_map_entry {
    __u32 pid;
    __u32 flow_id;
    __u64 last_seen;
    __u32 fd;               // File descriptor
    __u16 state;            // Connection state
} __attribute__((packed));

// Ring buffer para correlação com preservação de ordem temporal
struct correlation_ringbuf {
    __u32 producer_pos;
    __u32 consumer_pos;
    __u64 base_timestamp;     // Referência temporal global
    __u32 overflow_count;     // Estatísticas de perda
} __attribute__((aligned(64)));

// Mapas eBPF para correlação cross-layer
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_CONNECTIONS);
    __type(key, struct flow_key);
    __type(value, struct pid_flow_map_entry);
} pid_flow_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_PID);
    __type(key, __u32);  // PID
    __type(value, __u32);  // Flow ID
} socket_pid_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 16 * 1024 * 1024);  // 16MB ring buffer
} network_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 8 * 1024 * 1024);   // 8MB ring buffer  
} syscall_events SEC(".maps");

// Funcoes para sincronizacao TSC (baseadas na dissertacao)

static __always_inline __u64 rdtsc_ordered(void) {
    __u32 low, high;
    asm volatile("lfence; rdtsc" : "=a" (low), "=d" (high) :: "memory");
    return ((__u64)high << 32) | low;
}

static __always_inline __u64 normalize_tsc_timestamp(void) {
    // Implementacao simplificada - sera expandida com per-CPU sync
    __u64 current_tsc = rdtsc_ordered();
    
    // Por agora, retornamos TSC raw
    // TODO: Implementar compensacao drift multi-CPU conforme dissertacao
    return current_tsc;
}

// Hash function para flow keys (jhash implementation simplificada)
static __always_inline __u32 jhash_flow_key(struct flow_key *key) {
    __u32 hash = 0;
    
    hash ^= key->src_ip;
    hash ^= key->dst_ip;
    hash ^= ((__u32)key->src_port << 16) | key->dst_port;
    hash ^= key->protocol;
    
    // Simple hash mixing
    hash ^= hash >> 16;
    hash ^= hash >> 8;
    
    return hash;
}

// Socket PID lookup para correlacao cross-layer
static __always_inline __u32 socket_pid_lookup(struct flow_key *flow) {
    struct pid_flow_map_entry *entry;
    
    entry = bpf_map_lookup_elem(&pid_flow_map, flow);
    if (!entry) {
        return 0;  // PID not found
    }
    
    // Update last_seen timestamp
    entry->last_seen = normalize_tsc_timestamp();
    
    return entry->pid;
}

// Funcao para estabelecer mapeamento PID-Flow
static __always_inline int establish_pid_flow_mapping(__u32 pid, struct flow_key *flow) {
    struct pid_flow_map_entry entry = {
        .pid = pid,
        .flow_id = jhash_flow_key(flow),
        .last_seen = normalize_tsc_timestamp(),
        .fd = 0,  // Will be updated by syscall monitoring
        .state = 1  // Connected state
    };
    
    // Store in both directions for efficient lookup
    bpf_map_update_elem(&pid_flow_map, flow, &entry, BPF_ANY);
    bpf_map_update_elem(&socket_pid_map, &pid, &entry.flow_id, BPF_ANY);
    
    return 0;
}

// Protocol classifier para eventos de rede
static __always_inline __u16 proto_classifier(struct iphdr *ip) {
    switch (ip->protocol) {
        case IPPROTO_TCP:
            return 1;
        case IPPROTO_UDP:
            return 2;
        case IPPROTO_ICMP:
            return 3;
        default:
            return 0;
    }
}

#endif /* __CROSSLAYER_COMMON_H__ */