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
    int fd;
    u64 ts;
    char comm[16];
    char fname[200];
    char old_fname[128];
    u32 size;
    u64 offset;
    int success;
    char op_type[10]; // Tipo de operação (read, write, open, etc.)
    char syscall[16]; // Novo campo para armazenar o nome da chamada de sistema
};

BPF_PERF_OUTPUT(events);

int kprobe__vfs_unlink(struct pt_regs *ctx, struct inode *dir, struct dentry *dentry) {
    struct data_t data = {};
    u32 uid = bpf_get_current_uid_gid();
    strcpy(data.syscall, "unlink");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = uid;
    data.fd = dentry->d_inode->i_ino; // Using inode number as an identifier
    data.ts = bpf_ktime_get_ns();
    data.size = 0; // No size involved in unlink
    data.offset = 0; // No offset involved in unlink
    data.success = 1; // Assuming success if this function is called
    strcpy(data.op_type, "unlink");

    bpf_probe_read_kernel(&data.fname, sizeof(data.fname), dentry->d_name.name);
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
