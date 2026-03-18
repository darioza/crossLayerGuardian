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
#include <linux/sched.h>

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

// Declaração da saída para enviar dados para o espaço do usuário
BPF_PERF_OUTPUT(events);

int kprobe__sys_mmap(struct pt_regs *ctx) {
    struct data_t data = {};
    strcpy(data.syscall, "mmap");
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    // Obter o nome do arquivo ou dispositivo de montagem a partir do dentry
    struct vm_area_struct *vma = (struct vm_area_struct *)(PT_REGS_PARM1(ctx));
    if (vma && vma->vm_file) {
        bpf_probe_read_kernel(&data.fname, sizeof(data.fname), vma->vm_file->f_path.dentry->d_name.name);
    }

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
