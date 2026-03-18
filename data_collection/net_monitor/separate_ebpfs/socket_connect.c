#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>

#define TASK_COMM_LEN 16
#define AF_INET 2

struct data_t {
    u32 pid;
    char comm[TASK_COMM_LEN];
    u8 family;
    u16 dport;
    u32 daddr;
    u32 saddr;
    u16 sport;
    char service[16];
    u8 tcp_flags;
};

BPF_PERF_OUTPUT(events);

int kprobe__inet_stream_connect(struct pt_regs *ctx, struct socket *sock)
{
    struct data_t data = {};
    u16 family = 0;
    u16 type = 0;

    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    // Use bpf_probe_read_kernel para ler dados do kernel
    bpf_probe_read_kernel(&family, sizeof(family), &sock->sk->__sk_common.skc_family);
    data.family = family;
    bpf_probe_read_kernel(&data.dport, sizeof(data.dport), &sock->sk->__sk_common.skc_dport);
    bpf_probe_read_kernel(&data.daddr, sizeof(data.daddr), &sock->sk->__sk_common.skc_daddr);
    bpf_probe_read_kernel(&data.saddr, sizeof(data.saddr), &sock->sk->__sk_common.skc_rcv_saddr);
    bpf_probe_read_kernel(&data.sport, sizeof(data.sport), &sock->sk->__sk_common.skc_num);
    
    // Leia o tipo do socket
    bpf_probe_read_kernel(&type, sizeof(type), &sock->type);

    // Leia o nome do serviço
    char *service_name;
    bpf_probe_read_kernel(&service_name, sizeof(service_name), &sock->sk->sk_prot->name);
    bpf_probe_read_kernel_str(data.service, sizeof(data.service), service_name);

    if (family == AF_INET && type == SOCK_STREAM) {
        u8 state = 0;
        bpf_probe_read_kernel(&state, sizeof(state), &sock->sk->__sk_common.skc_state);
        data.tcp_flags = state;
    }

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}