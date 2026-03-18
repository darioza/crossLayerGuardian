#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>

struct data_t {
    u32 pid;
    char comm[TASK_COMM_LEN];
    u16 oldstate;
    u16 newstate;
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    u8 protocol;
    char service[16];
};

BPF_PERF_OUTPUT(events);

int kprobe__tcp_set_state(struct pt_regs *ctx, struct sock *sk, int state) {
    struct data_t data = {};
    events.perf_submit(ctx, &data, sizeof(data));
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    data.oldstate = sk->sk_state;
    data.newstate = state;
    data.saddr = sk->__sk_common.skc_rcv_saddr;
    data.daddr = sk->__sk_common.skc_daddr;
    data.sport = sk->__sk_common.skc_num;
    data.dport = sk->__sk_common.skc_dport;

    // Obter o tipo de protocolo e serviço
    data.protocol = IPPROTO_TCP;
    bpf_probe_read_str(&data.service, sizeof(data.service), sk->sk_prot->name);

    return 0;
}