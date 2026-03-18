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
    char fname[200];  // Nome do diretório a ser criado
    u32 size;         // Pode ser usado para armazenar o modo de criação do diretório
    u64 offset;       // Não usado em mkdir, mas mantido por consistência
    int success;      // Sucesso da operação, pode ser definido como 1 assumindo que mkdir foi chamado com sucesso
    char op_type[10]; // Tipo de operação, aqui será "mkdir"
    char syscall[16]; // Novo campo para armazenar o nome da chamada de sistema
};

BPF_PERF_OUTPUT(events);

int kprobe__vfs_mkdir(struct pt_regs *ctx, const struct path *dir, umode_t mode) {
    struct data_t data = {};
    strcpy(data.syscall, "mkdir");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_probe_read_kernel(data.fname, sizeof(data.fname), dir->dentry->d_name.name);
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    strcpy(data.op_type, "mkdir");

    data.size = (u32) mode;  // Armazenando o modo de criação do diretório
    data.offset = 0;         // Não relevante para mkdir
    data.success = 1;        // Assumindo sucesso se a função foi chamada

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
