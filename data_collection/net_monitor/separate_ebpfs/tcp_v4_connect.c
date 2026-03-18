#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>

BPF_HASH(currsock, u32, struct sock *);
BPF_PERF_OUTPUT(events);

struct event_data_t {
    u32 pid;
    u32 saddr;
    u32 daddr;
    u16 dport;
    char comm[TASK_COMM_LEN];
    u8 protocol;
    char service[16];
    u8 tcp_flags;
};

int kprobe__tcp_v4_connect(struct pt_regs *ctx, struct sock *sk) {
    u32 pid = bpf_get_current_pid_tgid();
    // stash the sock ptr for lookup on return
    currsock.update(&pid, &sk);
    return 0;
};

int kretprobe__tcp_v4_connect(struct pt_regs *ctx) {
    int ret = PT_REGS_RC(ctx);
    u32 pid = bpf_get_current_pid_tgid();
    struct sock **skpp;
    skpp = currsock.lookup(&pid);
    if (skpp == 0) {
        return 0; // missed entry
    }
    if (ret != 0) {
        // failed to send SYNC packet, may not have populated
        // socket __sk_common.{skc_rcv_saddr, ...}
        currsock.delete(&pid);
        return 0;
    }
    // pull in details
    struct sock *skp = *skpp;
    u32 saddr = skp->__sk_common.skc_rcv_saddr;
    u32 daddr = skp->__sk_common.skc_daddr;
    u16 dport = skp->__sk_common.skc_dport;
    
    // output
    struct event_data_t event = {};
    events.perf_submit(ctx, &event, sizeof(event));
    event.pid = pid;
    event.saddr = saddr;
    event.daddr = daddr;
    event.dport = ntohs(dport);
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    event.protocol = IPPROTO_TCP;
    bpf_probe_read_str(&event.service, sizeof(event.service), skp->sk_prot->name);
    event.tcp_flags = skp->sk_state;
    currsock.delete(&pid);
    return 0;
}