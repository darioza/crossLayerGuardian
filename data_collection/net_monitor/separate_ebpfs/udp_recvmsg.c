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
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    char comm[TASK_COMM_LEN];
    char protocol[8];
    char service[16];
};

BPF_PERF_OUTPUT(events);

int kprobe__udp_recvmsg(struct pt_regs *ctx, struct sock *sk) {
    struct data_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    bpf_probe_read(&data.saddr, sizeof(data.saddr), &sk->__sk_common.skc_rcv_saddr);
    bpf_probe_read(&data.daddr, sizeof(data.daddr), &sk->__sk_common.skc_daddr);
    bpf_probe_read(&data.sport, sizeof(data.sport), &sk->__sk_common.skc_num);
    u16 dport;
    bpf_probe_read(&dport, sizeof(dport), &sk->__sk_common.skc_dport);
    data.dport = ntohs(dport);

    const char *proto_name = get_protocol_name(IPPROTO_UDP);
    bpf_probe_read_str(data.protocol, sizeof(data.protocol), proto_name);

    const char *service_name = get_service_name(IPPROTO_UDP, data.dport);
    bpf_probe_read_str(data.service, sizeof(data.service), service_name);

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}