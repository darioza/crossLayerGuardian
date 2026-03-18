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
#include <linux/module.h>

struct data_t {
    u32 pid;
    u32 uid;
    u64 ts;
    char comm[16];
    char fname[200];  // Nome do módulo
    int success;
    char op_type[10];
    char syscall[16]; // Novo campo para armazenar o nome da chamada de sistema
};

BPF_PERF_OUTPUT(events);

// Monitoramento de carregamento de módulos
int kprobe__init_module(struct pt_regs *ctx, void *module_image, unsigned long len, const char *param_values) {
    struct data_t data = {};
    strcpy(data.syscall, "init_module");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(data.fname, sizeof(data.fname), param_values);  // Assume que param_values inclui o nome
    strcpy(data.op_type, "load_mod");
    data.success = 1;

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

// Monitoramento de remoção de módulos
int kprobe__delete_module(struct pt_regs *ctx, const char *name, unsigned int flags) {
    struct data_t data = {};
    strcpy(data.syscall, "delete_module");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    bpf_probe_read_user_str(data.fname, sizeof(data.fname), name);
    strcpy(data.op_type, "remove_mod");
    data.success = 1;

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
