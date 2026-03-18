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
#include <linux/fs.h>

struct data_t {
    u32 pid;
    u32 uid;
    u64 ts;
    char comm[16];
    char fname[200];
    char old_fname[128];
    u32 size;
    u64 offset;
    int success;
    char op_type[10]; // Tipo de operação (rename)
    char syscall[16]; // Novo campo para armazenar o nome da chamada de sistema
};

BPF_PERF_OUTPUT(events);

int kprobe__vfs_rename(struct pt_regs *ctx, const struct path *old_path, const struct path *new_path) {
    struct data_t data = {};
    strcpy(data.syscall, "rename");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_probe_read_kernel(data.fname, sizeof(data.fname), new_path->dentry->d_name.name);
    bpf_probe_read_kernel(data.old_fname, sizeof(data.old_fname), old_path->dentry->d_name.name);
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    strcpy(data.op_type, "rename");

    data.size = 0; // No size involved in rename
    data.offset = 0; // No offset involved in rename
    data.success = 1; // Assuming rename success if this function is called

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
