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
    char fname[200];  // Path of the executable
    char old_fname[128];
    u32 size;
    u64 offset;
    int success;
    char op_type[10];
    char syscall[16]; // Novo campo para armazenar o nome da chamada de sistema

};

BPF_PERF_OUTPUT(events);

int kprobe____x64_sys_execve(struct pt_regs *ctx, const char __user *filename, const char __user *const __user *__argv) {
    struct data_t data = {};
    strcpy(data.syscall, "execve");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(data.fname, sizeof(data.fname), filename);
    strcpy(data.op_type, "execve");
    data.success = 1;  // Assume success if this function is reached

    // Setting unused fields to default values
    data.fd = -1;
    data.old_fname[0] = '\0';
    data.size = 0;
    data.offset = 0;

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
