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

# Mapeamento estático de números de protocolo para nomes
PROTOCOL_NAMES = {
    0: "HOPOPT",
    1: "ICMP",
    2: "IGMP",
    3: "GGP",
    4: "IP-IN-IP",
    5: "ST",
    6: "TCP",
    7: "CBT",
    8: "EGP",
    9: "IGP",
    10: "BBN-RCC-MON",
    11: "NVP-II",
    12: "PUP",
    13: "ARGUS",
    14: "EMCON",
    15: "XNET",
    16: "CHAOS",
    17: "UDP",
    18: "MUX",
    19: "DCN-MEAS",
    20: "HMP",
    21: "PRM",
    22: "XNS-IDP",
    23: "TRUNK-1",
    24: "TRUNK-2",
    25: "LEAF-1",
    26: "LEAF-2",
    27: "RDP",
    28: "IRTP",
    29: "ISO-TP4",
    30: "NETBLT",
    31: "MFE-NSP",
    32: "MERIT-INP",
    33: "DCCP",
    34: "3PC",
    35: "IDPR",
    36: "XTP",
    37: "DDP",
    38: "IDPR-CMTP",
    39: "TP++",
    40: "IL",
    41: "IPv6",
    42: "SDRP",
    43: "IPv6-Route",
    44: "IPv6-Frag",
    45: "IDRP",
    46: "RSVP",
    47: "GRE",
    48: "DSR",
    49: "BNA",
    50: "ESP",
    51: "AH",
    52: "I-NLSP",
    53: "SWIPE",
    54: "NARP",
    55: "MOBILE",
    56: "TLSP",
    57: "SKIP",
    58: "IPv6-ICMP",
    59: "IPv6-NoNxt",
    60: "IPv6-Opts",
    61: "HOSTPAD",
    62: "CFTP",
    63: "SAT-EXPAK",
    64: "KRYPTOLAN",
    65: "RVD",
    66: "IPPC",
    67: "DFSSRV",
    68: "SAT-MON",
    69: "VISA",
    70: "IPCV",
    71: "CPNX",
    72: "CPHB",
    73: "WSN",
    74: "PV",
    75: "BR-SAT-MO",
    76: "SUN-ND",
    77: "WB-MON",
    78: "WB-EXPAK",
    79: "ISO-IP",
    80: "VMTP",
    81: "SECURE-VM",
    82: "VINES",
    83: "TTP",
    84: "IPTM",
    85: "NSFNET-IG",
    86: "DGP",
    87: "TCF",
    88: "EIGRP",
    89: "OSPFIGR",
    90: "Sprite-RP",
    91: "LARP",
    92: "MTP",
    93: "AX.25",
    94: "IPIP",
    95: "MICP",
    96: "SCC-SP",
    97: "ETIQBE",
    98: "ENCAP",
    99: "GMTP",
    100: "IFMP",
    101: "PNNI",
    102: "PIM",
    103: "ARIS",
    104: "SCPS",
    105: "QNX",
    106: "A/N",
    107: "IPComp",
    108: "SNP",
    109: "Compaq-Pe",
    110: "IPX-in-IP",
    111: "VRRP",
    112: "PGM",
    113: "0-HOP",
    114: "L2TP",
    115: "DDX",
    116: "IATP",
    117: "STP",
    118: "SRP",
    119: "UTI",
    120: "SMP",
    121: "SM",
    122: "PTP",
    123: "IS-IS",
    124: "FIRE",
    125: "CRTP",
    126: "CRUDP",
    127: "SSCOPMCE",
    128: "IPLT",
    129: "SPS",
    130: "PIPE",
    131: "SCTP",
    132: "FC",
    133: "RSVP-E2E-I",
    134: "MOBILESP",
    135: "UDP-LITE",
    136: "MPLS-in-I",
    137: "MANET",
    138: "HIP",
    139: "SHIM6",
    140: "WESP",
    141: "ROHC",
    255: "RESERVED",
}


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
base_dir = "/home/darioza/eguardian/data_collection/net_monitor/separate_ebpfs"
bpf_files = {
    "tcp_v4_connect.c": None,
    "udp_sendmsg.c": None,
    "udp_recvmsg.c": None,
    "ip_receive.c": None,
    "ip_send_skb.c": None,
    "tcp_set_state.c": None,
    "socket_bind.c": None,
    "socket_connect.c": None,
    "tcp_close.c": None,
    "socket_create.c": None
}

def inet_ntoa(addr):
    return '.'.join(map(str, [(addr >> i) & 0xff for i in (0, 8, 16, 24)]))

def setup_bpf_program(bpf_file):
    bpf_path = os.path.join(base_dir, bpf_file)
    bpf_obj = BPF(src_file=bpf_path, debug=0)
    output_path = os.path.join(output_directory, "net_collected_data.json")

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
                    #print(f"Dados gravados no arquivo JSON: {event_data}")

    with suppress_stderr():
        bpf_obj["events"].open_perf_buffer(print_event, page_cnt=128)
        print(f"Programa eBPF '{bpf_file}' iniciado com sucesso")

    return bpf_obj

def format_event_data(event, bpf_file):
    event_data = {"event": os.path.splitext(bpf_file)[0], "pid": event.pid, "timestamp": datetime.now().isoformat()}

    # Adicionar os demais campos
    for field_name, field_type in event._fields_:
        if field_name not in ["event", "pid", "timestamp"]:
            field_value = getattr(event, field_name)
            if field_name in ["saddr", "daddr", "laddr"]:
                event_data[field_name] = inet_ntoa(field_value)
            elif field_name == "protocol":
                protocol_name = PROTOCOL_NAMES.get(field_value, str(field_value))
                event_data[field_name] = protocol_name
            elif isinstance(field_value, bytes):
                event_data[field_name] = field_value.decode(errors='ignore')
            else:
                event_data[field_name] = field_value

    return event_data

def poll_bpf(bpf_obj):
    while True:
        try:
            with suppress_stderr():
                bpf_obj.perf_buffer_poll()
        except KeyboardInterrupt:
            break

# Fila compartilhada para receber dados do net_monitor
def preprocess_data(data):
    # Implementar a lógica para extrair recursos relevantes para o modelo NSL-KDD
    # e retornar os dados pré-processados no formato esperado pelo modelo
    preprocessed_data = {}

    for event in data:
        protocol_type = event.get('protocol', '')
        if protocol_type.isdigit():
            protocol_type = PROTOCOL_NAMES.get(int(protocol_type), str(protocol_type))
        service = event.get('service', '')
        flag = event.get('flag', '')
        src_bytes = event.get('src_bytes', 0)
        dst_bytes = event.get('dst_bytes', 0)
        land = event.get('land', 0)
        wrong_fragment = event.get('wrong_fragment', 0)
        urgent = event.get('urgent', 0)
        hot = event.get('hot', 0)
        num_failed_logins = event.get('num_failed_logins', 0)
        logged_in = event.get('logged_in', 0)
        num_compromised = event.get('num_compromised', 0)
        root_shell = event.get('root_shell', 0)
        su_attempted = event.get('su_attempted', 0)
        num_root = event.get('num_root', 0)
        num_file_creations = event.get('num_file_creations', 0)
        num_shells = event.get('num_shells', 0)
        num_access_files = event.get('num_access_files', 0)
        num_outbound_cmds = event.get('num_outbound_cmds', 0)
        is_host_login = event.get('is_host_login', 0)
        is_guest_login = event.get('is_guest_login', 0)
        count = event.get('count', 0)
        srv_count = event.get('srv_count', 0)
        serror_rate = event.get('serror_rate', 0)
        srv_serror_rate = event.get('srv_serror_rate', 0)
        rerror_rate = event.get('rerror_rate', 0)
        srv_rerror_rate = event.get('srv_rerror_rate', 0)
        same_srv_rate = event.get('same_srv_rate', 0)
        diff_srv_rate = event.get('diff_srv_rate', 0)
        srv_diff_host_rate = event.get('srv_diff_host_rate', 0)
        dst_host_count = event.get('dst_host_count', 0)
        dst_host_srv_count = event.get('dst_host_srv_count', 0)
        dst_host_same_srv_rate = event.get('dst_host_same_srv_rate', 0)
        dst_host_diff_srv_rate = event.get('dst_host_diff_srv_rate', 0)
        dst_host_same_src_port_rate = event.get('dst_host_same_src_port_rate', 0)
        dst_host_srv_diff_host_rate = event.get('dst_host_srv_diff_host_rate', 0)
        dst_host_serror_rate = event.get('dst_host_serror_rate', 0)
        dst_host_srv_serror_rate = event.get('dst_host_srv_serror_rate', 0)
        dst_host_rerror_rate = event.get('dst_host_rerror_rate', 0)
        dst_host_srv_rerror_rate = event.get('dst_host_srv_rerror_rate', 0)

        preprocessed_data = {
            'protocol_type': protocol_type,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'land': land,
            'wrong_fragment': wrong_fragment,
            'urgent': urgent,
            'hot': hot,
            'num_failed_logins': num_failed_logins,
            'logged_in': logged_in,
            'num_compromised': num_compromised,
            'root_shell': root_shell,
            'su_attempted': su_attempted,
            'num_root': num_root,
            'num_file_creations': num_file_creations,
            'num_shells': num_shells,
            'num_access_files': num_access_files,
            'num_outbound_cmds': num_outbound_cmds,
            'is_host_login': is_host_login,
            'is_guest_login': is_guest_login,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror_rate,
            'srv_serror_rate': srv_serror_rate,
            'rerror_rate': rerror_rate,
            'srv_rerror_rate': srv_rerror_rate,
            'same_srv_rate': same_srv_rate,
            'diff_srv_rate': diff_srv_rate,
            'srv_diff_host_rate': srv_diff_host_rate,
            'dst_host_count': dst_host_count,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': dst_host_same_srv_rate,
            'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
            'dst_host_same_src_port_rate': dst_host_same_src_port_rate,
            'dst_host_srv_diff_host_rate': dst_host_srv_diff_host_rate,
            'dst_host_serror_rate': dst_host_serror_rate,
            'dst_host_srv_serror_rate': dst_host_srv_serror_rate,
            'dst_host_rerror_rate': dst_host_rerror_rate,
            'dst_host_srv_rerror_rate': dst_host_srv_rerror_rate
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

# Executar o monitor isoladamente
if __name__ == "__main__":
    print("Executando net_monitor.py isoladamente")