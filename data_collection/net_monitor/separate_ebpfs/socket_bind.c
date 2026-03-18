#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>

#define TASK_COMM_LEN 16
#define AF_INET 2

struct data_t {
    u32 pid;
    char comm[TASK_COMM_LEN];
    u8 family;
    u16 port;
    u32 addr;
    char protocol[8];
    char service[16];
};

BPF_PERF_OUTPUT(events);

int kprobe__inet_bind(struct pt_regs *ctx, struct socket *sock)
{
    struct data_t data = {};
    u16 family = 0;
    u16 type = 0;
    struct sock *sk;

    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    bpf_probe_read_kernel(&sk, sizeof(sk), &sock->sk);
    if (!sk)
        return 0;

    bpf_probe_read_kernel(&family, sizeof(family), &sk->__sk_common.skc_family);
    data.family = family;
    bpf_probe_read_kernel(&data.port, sizeof(data.port), &sk->__sk_common.skc_num);
    bpf_probe_read_kernel(&data.addr, sizeof(data.addr), &sk->__sk_common.skc_rcv_saddr);
    
    bpf_probe_read_kernel(&type, sizeof(type), &sock->type);

    if (type == SOCK_STREAM)
        bpf_probe_read_kernel_str(data.protocol, sizeof(data.protocol), "TCP");
    else if (type == SOCK_DGRAM)
        bpf_probe_read_kernel_str(data.protocol, sizeof(data.protocol), "UDP");
    else
        bpf_probe_read_kernel_str(data.protocol, sizeof(data.protocol), "UNKNOWN");

    if (type == SOCK_STREAM && data.port == 80)
        bpf_probe_read_kernel_str(data.service, sizeof(data.service), "HTTP");
    else if (type == SOCK_DGRAM && data.port == 53)
        bpf_probe_read_kernel_str(data.service, sizeof(data.service), "DNS");
    else
        bpf_probe_read_kernel_str(data.service, sizeof(data.service), "UNKNOWN");

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}