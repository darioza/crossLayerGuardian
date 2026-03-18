#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/socket.h>

struct data_t {
    u32 pid;
    char comm[TASK_COMM_LEN];
    u8 family;
    u8 type;
    u32 laddr;
    u16 lport;
    char protocol[8];
};

BPF_PERF_OUTPUT(events);

int kprobe__inet_create(struct pt_regs *ctx, struct socket *sock, int protocol, int kern) {
    if (!sock || !sock->ops)
        return 0;

    struct data_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    struct sock *sk = sock->sk;
    if (sk) {
        data.family = BPF_CORE_READ(sk, sk_family);
        data.type = BPF_CORE_READ(sock, type);
        data.laddr = BPF_CORE_READ(sk, __sk_common.skc_rcv_saddr);
        data.lport = BPF_CORE_READ(sk, __sk_common.skc_num);
        data.protocol = protocol;
        bpf_probe_read_str(&data.protocol, sizeof(data.protocol), get_protocol_name(protocol));
    }

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}