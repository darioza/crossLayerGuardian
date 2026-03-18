import os
import json
from collections import deque
from multiprocessing import Queue
from bcc import BPF
import os
import time

class DataCollector:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)  # Criar o diretório data, se não existir
        self.collected_data = []  # Inicializa o atributo collected_data como uma lista vazia
        self.max_data_size = 1000  # Definir o tamanho máximo da lista

        self.event_queue = Queue()

    def collect_event(self, event):
        self.event_queue.put(event)

    def is_hids_event(self, process_name, file_path):
        hids_files = [
            "net_collected_data.json",
            "file_collected_data.json"
        ]
        hids_dirs = [
            "/home/darioza/eguardian/",
            "/home/darioza/eguardian/data_collection/",
            "/home/darioza/eguardian/machine_learning/",
            "/home/darioza/eguardian/user_interface/"
        ]
        if any(hids_file in file_path for hids_file in hids_files):
            return True
        elif any(file_path.startswith(hids_dir) for hids_dir in hids_dirs):
            return True
        else:
            return False

    def collect_data(self, file_name):
        print(f"Coletando dados do arquivo {file_name}")
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    process_name = data.get("process_name", "")
                    file_path = data.get("file_path", "")
                    if not self.is_hids_event(process_name, file_path):
                        self.collected_data.append(data)
                    #   print(f"Dados adicionados à collected_data: {data}")  # Adicionar esse log
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")

    def limit_data_size(self):
        while len(self.collected_data) > self.max_data_size:
            self.collected_data.pop(0)  # Remover o elemento mais antigo
    '''        
    def preprocess_data(self, event):
        # Lógica de pré-processamento dos dados
        preprocessed_data = event
        # Adicionar qualquer outra lógica de pré-processamento necessária aqui

        return preprocessed_data
    '''
    def map_event_fields(self, event):
        # Mapear os campos do evento para os campos esperados pelas funções de extração
        mapped_event = {}

        if event['event'] in ['file_read', 'file_write', 'file_open', 'file_create', 'file_delete']:
            # Mapear campos para eventos de arquivo
            mapped_event['syscall'] = event.get('syscall', '')
            mapped_event['process_name'] = event.get('process_name', '')
            mapped_event['pid'] = event.get('pid', 0)
            mapped_event['file_path'] = event.get('file_path', '')
            mapped_event['file_name'] = event.get('file_name', '')
            mapped_event['file_size'] = event.get('file_size', 0)
            mapped_event['file_type'] = event.get('file_type', '')
            mapped_event['file_permissions'] = event.get('file_permissions', '')
            mapped_event['file_owner'] = event.get('file_owner', '')
            mapped_event['file_group'] = event.get('file_group', '')
            # Mapear outros campos relevantes para eventos de arquivo
        elif event['event'] in ['network_connection', 'network_packet']:
            # Mapear campos para eventos de rede
            mapped_event['protocol_type'] = event.get('protocol_type', '')
            mapped_event['service'] = event.get('service', '')
            mapped_event['flag'] = event.get('flag', '')
            mapped_event['src_bytes'] = event.get('src_bytes', 0)
            mapped_event['dst_bytes'] = event.get('dst_bytes', 0)
            mapped_event['land'] = event.get('land', 0)
            mapped_event['wrong_fragment'] = event.get('wrong_fragment', 0)
            mapped_event['urgent'] = event.get('urgent', 0)
            mapped_event['hot'] = event.get('hot', 0)
            mapped_event['num_failed_logins'] = event.get('num_failed_logins', 0)
            mapped_event['logged_in'] = event.get('logged_in', 0)
            mapped_event['num_compromised'] = event.get('num_compromised', 0)
            mapped_event['root_shell'] = event.get('root_shell', 0)
            mapped_event['su_attempted'] = event.get('su_attempted', 0)
            mapped_event['num_root'] = event.get('num_root', 0)
            mapped_event['num_file_creations'] = event.get('num_file_creations', 0)
            mapped_event['num_shells'] = event.get('num_shells', 0)
            mapped_event['num_access_files'] = event.get('num_access_files', 0)
            mapped_event['num_outbound_cmds'] = event.get('num_outbound_cmds', 0)
            mapped_event['is_host_login'] = event.get('is_host_login', 0)
            mapped_event['is_guest_login'] = event.get('is_guest_login', 0)
            mapped_event['count'] = event.get('count', 0)
            mapped_event['srv_count'] = event.get('srv_count', 0)
            mapped_event['serror_rate'] = event.get('serror_rate', 0)
            mapped_event['srv_serror_rate'] = event.get('srv_serror_rate', 0)
            mapped_event['rerror_rate'] = event.get('rerror_rate', 0)
            mapped_event['srv_rerror_rate'] = event.get('srv_rerror_rate', 0)
            mapped_event['same_srv_rate'] = event.get('same_srv_rate', 0)
            mapped_event['diff_srv_rate'] = event.get('diff_srv_rate', 0)
            mapped_event['srv_diff_host_rate'] = event.get('srv_diff_host_rate', 0)
            mapped_event['dst_host_count'] = event.get('dst_host_count', 0)
            mapped_event['dst_host_srv_count'] = event.get('dst_host_srv_count', 0)
            mapped_event['dst_host_same_srv_rate'] = event.get('dst_host_same_srv_rate', 0)
            mapped_event['dst_host_diff_srv_rate'] = event.get('dst_host_diff_srv_rate', 0)
            mapped_event['dst_host_same_src_port_rate'] = event.get('dst_host_same_src_port_rate', 0)
            mapped_event['dst_host_srv_diff_host_rate'] = event.get('dst_host_srv_diff_host_rate', 0)
            mapped_event['dst_host_serror_rate'] = event.get('dst_host_serror_rate', 0)
            mapped_event['dst_host_srv_serror_rate'] = event.get('dst_host_srv_serror_rate', 0)
            mapped_event['dst_host_rerror_rate'] = event.get('dst_host_rerror_rate', 0)
            mapped_event['dst_host_srv_rerror_rate'] = event.get('dst_host_srv_rerror_rate', 0)
            # Mapear outros campos relevantes para eventos de rede

        return mapped_event

    def run(self, interval=5):
        while True:
            self.collect_data('file_collected_data.json')
            self.collect_data('net_collected_data.json')
            time.sleep(interval)