#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <linux/netdevice.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <bcc/proto.h>

struct data_t {
    u32 pid;
    char comm[TASK_COMM_LEN];
    char ifname[IFNAMSIZ];
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
    u8 protocol;
};

BPF_PERF_OUTPUT(events);

TRACEPOINT_PROBE(net, netif_receive_skb) {
    struct data_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    struct sk_buff *skb = (struct sk_buff *)args->skbaddr;
    struct net_device *dev = skb->dev;
    if (dev) {
        bpf_probe_read_kernel(&data.ifname, sizeof(data.ifname), dev->name);
    }

    // Read IP header
    struct iphdr *iph = (struct iphdr *)(skb->head + skb->network_header);
    data.saddr = iph->saddr;
    data.daddr = iph->daddr;
    data.protocol = iph->protocol;

    // Read transport header (TCP or UDP)
    if (iph->protocol == IPPROTO_TCP) {
        struct tcphdr *tcph = (struct tcphdr *)(skb->head + skb->transport_header);
        data.sport = tcph->source;
        data.dport = tcph->dest;
    } else if (iph->protocol == IPPROTO_UDP) {
        struct udphdr *udph = (struct udphdr *)(skb->head + skb->transport_header);
        data.sport = udph->source;
        data.dport = udph->dest;
    }

    events.perf_submit(args, &data, sizeof(data));

    return 0;
}