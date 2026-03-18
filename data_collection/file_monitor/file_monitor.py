#file_monitor.py
import os
import json
import threading
from bcc import BPF
from datetime import datetime
from collections import deque
from queue import Queue
import time
import sys
import tempfile
from contextlib import contextmanager
import multiprocessing

# Criar um lock para sincronizar a escrita nos arquivos JSON
json_write_lock = threading.Lock()

# Número máximo de eventos permitidos nos arquivos JSON
MAX_EVENTS = 10000

# Período de retenção de dados (em minutos)
DATA_RETENTION_PERIOD = 1

# Caminho para o arquivo de log
LOG_FILE = "/home/darioza/eguardian/data_collection/data/drive_control.log"


# Função para suprimir stderr
@contextmanager
def suppress_stderr():
    with tempfile.TemporaryFile() as tempf:
        old_stderr = sys.stderr
        sys.stderr = tempf
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Diretório base e mapeamento dos scripts eBPF
output_directory = "/home/darioza/eguardian/data_collection/data"
os.makedirs(output_directory, exist_ok=True)
base_dir = "/home/darioza/eguardian/data_collection/file_monitor/separate_ebpfs"
bpf_files = {
    "vfs_read.c": None,
    "vfs_write.c": None,
    "vfs_open.c": None,
    "vfs_unlink.c": None,
    "vfs_rename.c": None,
    "vfs_rmdir.c": None,
    "vfs_mkdir.c": None,
    "do_execve.c": None,
    "sys_exit.c": None,
    "setuid_setgid.c": None,
    "capset.c": None,
    "security_file_mmap.c": None
}

def setup_bpf_program(bpf_file):
    bpf_path = os.path.join(base_dir, bpf_file)
    bpf_obj = BPF(src_file=bpf_path, debug=0)
    output_path = os.path.join(output_directory, "file_collected_data.json")

    def print_event(cpu, data, size):
        event = bpf_obj["events"].event(data)
        event_data = format_event_data(event, bpf_file)

        # Verificar se event_data não está vazio
        if event_data:
            # Adquirir o lock antes de escrever no arquivo JSON
            with json_write_lock:
                with open(output_path, "a") as f:
                    json.dump(event_data, f)
                    f.write("\n")

    with suppress_stderr():
        bpf_obj["events"].open_perf_buffer(print_event, page_cnt=128)
        print(f"Programa eBPF '{bpf_file}' iniciado com sucesso")

    return bpf_obj

def format_event_data(event, bpf_file):
    event_data = {
        "event": os.path.splitext(bpf_file)[0],
        "pid": event.pid,
        "comm": event.comm.decode(errors='ignore'),
        "fname": event.fname.decode(errors='ignore'),
        "timestamp": datetime.now().isoformat()
    }

    for field_name, field_type in event._fields_:
        if field_name not in ["event", "pid", "comm", "fname", "timestamp"]:
            field_value = getattr(event, field_name)
            if isinstance(field_value, bytes):
                event_data[field_name] = field_value.decode(errors='ignore')
            else:
                event_data[field_name] = field_value

    # Filtro de eventos relacionados ao inotify e eGuardian
    if "inotify" in event_data["comm"] or "inotify" in event_data["fname"]:
        return {}

    hids_files = [
        "collected_data.json",
        "net_collected_data.json",
        "file_collected_data.json"
    ]
    hids_dirs = [
        "/home/darioza/eguardian/",
        "/home/darioza/eguardian/data_collection/",
        "/home/darioza/eguardian/machine_learning/",
        "/home/darioza/eguardian/user_interface/"
    ]
    if any(hids_file in event_data["fname"] for hids_file in hids_files) or any(event_data["fname"].startswith(hids_dir) for hids_dir in hids_dirs):
        return {}

    return event_data

def poll_bpf(bpf_obj):
    while True:
        try:
            with suppress_stderr():
                bpf_obj.perf_buffer_poll()
        except KeyboardInterrupt:
            break

def preprocess_data(data):
    # Implementar a lógica para calcular a sequência e frequência de syscalls
    # e retornar os dados pré-processados no formato esperado pelo modelo
    syscall_sequence = []
    syscall_counts = {}

    for event in data:
        syscall = event['syscall']
        syscall_sequence.append(syscall)
        syscall_counts[syscall] = syscall_counts.get(syscall, 0) + 1

    total_syscalls = sum(syscall_counts.values())
    syscall_frequency = {syscall: count / total_syscalls for syscall, count in syscall_counts.items()}

    preprocessed_data = {
        'syscall_sequence': syscall_sequence,
        'syscall_frequency': syscall_frequency
    }

    return preprocessed_data

threads = []
for bpf_file in bpf_files.keys():
    try:
        with suppress_stderr():
            bpf_obj = setup_bpf_program(bpf_file)
            thread = threading.Thread(target=poll_bpf, args=(bpf_obj,))
            threads.append(thread)
            thread.start()
            print(f"Programa eBPF '{bpf_file}' carregado e iniciado com sucesso.")
    except Exception as e:
        print(f"Failed to load BPF program {bpf_file}: {str(e)}")

# Aguardar todas as threads terminarem
for thread in threads:
    with suppress_stderr():
        thread.join()


