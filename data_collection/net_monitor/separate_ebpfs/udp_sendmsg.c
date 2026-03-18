#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>

struct data_t {
    u32 pid;
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    char comm[TASK_COMM_LEN];
    u8 protocol;
    char service[16];
};

BPF_PERF_OUTPUT(events);

int kprobe__udp_sendmsg(struct pt_regs *ctx, struct sock *sk) {
    struct data_t data = {};
    events.perf_submit(ctx, &data, sizeof(data));
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    // Access the socket fields
    u32 saddr = sk->__sk_common.skc_rcv_saddr;
    u32 daddr = sk->__sk_common.skc_daddr;
    u16 sport = sk->__sk_common.skc_num;
    u16 dport = sk->__sk_common.skc_dport;

    data.saddr = saddr;
    data.daddr = daddr;
    data.sport = ntohs(sport);
    data.dport = ntohs(dport);

    // Obter o tipo de protocolo e serviço
    data.protocol = IPPROTO_UDP;
    bpf_probe_read_str(&data.service, sizeof(data.service), sk->sk_prot->name);

    return 0;
}