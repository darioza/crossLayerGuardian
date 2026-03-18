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
#include <linux/capability.h>

struct data_t {
    u32 pid;
    u32 uid;
    int fd;
    u64 ts;
    char comm[16];
    char fname[200]; // Nome do processo
    char old_fname[128]; // Não utilizado
    u32 size; // Não utilizado
    u64 offset; // Não utilizado
    int success;
    char op_type[10];
    char syscall[16]; // Novo campo para armazenar o nome da chamada de sistema

};

BPF_PERF_OUTPUT(events);

// Monitoramento capset
int kprobe____x64_sys_capset(struct pt_regs *ctx, const struct cred *new, const struct kernel_cap *effective, const struct kernel_cap *inheritable, const struct kernel_cap *permitted) {
    struct data_t data = {};
    strcpy(data.syscall, "capset");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    snprintf(data.op_type, sizeof(data.op_type), "capset");
    data.success = 1;  // Assume success if this function is reached

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
