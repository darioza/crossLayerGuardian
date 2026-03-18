#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/tcp.h>
#include <linux/ip.h>

struct data_t {
    u32 pid;
    char comm[TASK_COMM_LEN];
    u16 sport;
    u16 dport;
    u32 saddr;
    u32 daddr;
    u8 state;
    u8 protocol;
    char service[16];
};

BPF_PERF_OUTPUT(events);

int kprobe__tcp_close(struct pt_regs *ctx, struct sock *sk) {
    if (!sk)
        return 0;

    struct data_t data = {};
    events.perf_submit(ctx, &data, sizeof(data));
  
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    // Obter informações do endereço IP e porta de origem
    struct inet_sock *inet = (struct inet_sock *)sk;
    bpf_probe_read(&data.saddr, sizeof(data.saddr), &inet->inet_saddr);
    bpf_probe_read(&data.sport, sizeof(data.sport), &inet->inet_sport);

    // Obter informações do endereço IP e porta de destino
    bpf_probe_read(&data.daddr, sizeof(data.daddr), &sk->__sk_common.skc_daddr);
    bpf_probe_read(&data.dport, sizeof(data.dport), &sk->__sk_common.skc_dport);

    // Obter o estado da conexão TCP
    u8 tcp_state;
    bpf_probe_read(&tcp_state, sizeof(tcp_state), (const void *)&sk->__sk_common.skc_state);
    data.state = tcp_state;

    // Obter o tipo de protocolo e serviço
    data.protocol = IPPROTO_TCP;
    bpf_probe_read_str(&data.service, sizeof(data.service), sk->sk_prot->name);

    return 0;
}