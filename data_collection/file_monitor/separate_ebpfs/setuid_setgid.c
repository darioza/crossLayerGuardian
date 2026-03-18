// Verificações para garantir que as macros não sejam redefinidas
#ifndef __HAVE_BUILTIN_BSWAP32__
#define __HAVE_BUILTIN_BSWAP32__ 1
#endif

#ifndef __HAVE_BUILTIN_BSWAP64__
#define __HAVE_BUILTIN_BSWAP64__ 1
#endif

#ifndef __HAVE_BUILTIN_BSWAP16__
#define __HAVE_BUILTIN_BSWAP16__ 1
#endif

#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct data_t {
    u32 pid;
    u32 uid;
    int fd;
    u64 ts;
    char comm[16];
    char fname[200];  // Nome do processo, pode não ser necessário
    char old_fname[128]; // Usado para armazenar o UID/GID antigo
    u32 size;        // Usado para armazenar o novo UID/GID
    u64 offset;
    int success;
    char op_type[10];
    char syscall[16]; // Novo campo para armazenar o nome da chamada de sistema
};

BPF_PERF_OUTPUT(events);

// Monitoramento setuid
int kprobe__sys_setuid(struct pt_regs *ctx, uid_t uid) {
    struct data_t data = {};
    strcpy(data.syscall, "setuid"); // ou "setgid", dependendo da chamada de sistema específica
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    snprintf(data.op_type, sizeof(data.op_type), "setuid");
    data.size = uid;
    data.success = 1;  // Assume success if this function is reached

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

// Monitoramento setgid
int kprobe__sys_setgid(struct pt_regs *ctx, gid_t gid) {
    struct data_t data = {};
    strcpy(data.syscall, "setgid");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    snprintf(data.op_type, sizeof(data.op_type), "setgid");
    data.size = gid;
    data.success = 1;  // Assume success if this function is reached

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
