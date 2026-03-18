#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/sched.h>
#include <linux/skbuff.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>

struct data_t {
    u32 pid;
    char comm[TASK_COMM_LEN];
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    u16 pkt_len;
    u8 protocol;
    char service[16];
    u8 tcp_flags;
};

BPF_PERF_OUTPUT(events);

int kprobe__ip_send_skb(struct pt_regs *ctx, struct sk_buff *skb) {
    if (!skb)
        return 0;

    struct data_t data = {};
    bpf_probe_read(&data.saddr, sizeof(data.saddr), &skb->network_header + offsetof(struct iphdr, saddr));
    bpf_probe_read(&data.daddr, sizeof(data.daddr), &skb->network_header + offsetof(struct iphdr, daddr));

    // Verifica o protocolo IP
    u8 protocol;
    bpf_probe_read(&protocol, sizeof(protocol), &skb->network_header + offsetof(struct iphdr, protocol));
    if (protocol == IPPROTO_TCP) {
        bpf_probe_read(&data.sport, sizeof(data.sport), &skb->transport_header + offsetof(struct tcphdr, source));
        bpf_probe_read(&data.dport, sizeof(data.dport), &skb->transport_header + offsetof(struct tcphdr, dest));
        data.protocol = IPPROTO_TCP;
        bpf_probe_read_str(&data.service, sizeof(data.service), skb->sk->sk_prot->name);
        data.tcp_flags = skb->sk->sk_state;
    } else if (protocol == IPPROTO_UDP) {
        bpf_probe_read(&data.sport, sizeof(data.sport), &skb->transport_header + offsetof(struct udphdr, source));
        bpf_probe_read(&data.dport, sizeof(data.dport), &skb->transport_header + offsetof(struct udphdr, dest));
        data.protocol = IPPROTO_UDP;
        bpf_probe_read_str(&data.service, sizeof(data.service), skb->sk->sk_prot->name);
    }

    data.pkt_len = skb->len;
    events.perf_submit(ctx, &data, sizeof(data));
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    return 0;
}