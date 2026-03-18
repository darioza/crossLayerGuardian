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
#include <linux/security.h>

struct data_t {
    u32 pid;
    u32 uid;
    u64 ts;
    char comm[16];
    char fname[200];  // Nome do arquivo ou dispositivo de montagem
    int success;
    char op_type[10];
    char syscall[16]; // Novo campo para armazenar o nome da chamada de sistema
};

BPF_PERF_OUTPUT(events);

// Monitoramento de criação de inodes
int kprobe__security_inode_create(struct pt_regs *ctx, struct inode *inode) {
    struct data_t data = {};
    strcpy(data.syscall, "create");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    strcpy(data.op_type, "inode_create");
    data.success = 1;

    // Obter o nome do arquivo a partir do inode
    struct dentry *dentry = d_find_alias(inode);
    if (dentry) {
        bpf_probe_read_kernel(&data.fname, sizeof(data.fname), dentry->d_name.name);
    }

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

// Monitoramento de unlink de inodes
int kprobe__security_inode_unlink(struct pt_regs *ctx, struct inode *inode) {
    struct data_t data = {};
    strcpy(data.syscall, "unlink");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    strcpy(data.op_type, "inode_unlink");
    data.success = 1;

    // Obter o nome do arquivo a partir do inode
    struct dentry *dentry = d_find_alias(inode);
    if (dentry) {
        bpf_probe_read_kernel(&data.fname, sizeof(data.fname), dentry->d_name.name);
    }

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

// Monitoramento de montagem de sistemas de arquivos
int kprobe__security_sb_mount(struct pt_regs *ctx, const char *dev_name, struct path *path, const char *type, unsigned long flags, void *data) {
    struct data_t data = {};
    strcpy(data.syscall, "mount"); // Adicionar o nome da chamada de sistema
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(data.fname, sizeof(data.fname), dev_name);
    strcpy(data.op_type, "sb_mount");
    data.success = 1;

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
