#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>

// Funções auxiliares
static inline const char *get_protocol_name(int protocol) {
    switch (protocol) {
        case IPPROTO_TCP: return "TCP";
        case IPPROTO_UDP: return "UDP";
        default: return "UNKNOWN";
    }
}

static inline const char *get_service_name(int protocol, u16 port) {
    if (protocol == IPPROTO_TCP && port == 80) return "HTTP";
    if (protocol == IPPROTO_UDP && port == 53) return "DNS";
    return "UNKNOWN";
}

struct data_t {
    u32 pid;
    char comm[TASK_COMM_LEN];
    u8 family;
    u8 type;
    u32 laddr;
    u16 lport;
    char protocol[8];
    char service[16];
};

BPF_PERF_OUTPUT(events);

int kprobe____sock_create(struct pt_regs *ctx, int family, int type, int protocol, struct socket **res, int kern) {
    struct data_t data = {};
    
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    data.family = family;
    data.type = type;

    const char *proto_name = get_protocol_name(protocol);
    bpf_probe_read_str(data.protocol, sizeof(data.protocol), proto_name);

    // Simplificamos esta parte para evitar acessos de memória potencialmente problemáticos
    const char *service_name = get_service_name(protocol, 0);  // Usamos 0 como porta padrão
    bpf_probe_read_str(data.service, sizeof(data.service), service_name);

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}