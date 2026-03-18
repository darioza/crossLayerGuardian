import os
import json
from datetime import datetime, timedelta

# Número máximo de eventos permitidos nos arquivos JSON
MAX_EVENTS = 10000

# Período de retenção de dados (em minutos)
DATA_RETENTION_PERIOD = 1

# Caminho para o arquivo de log
LOG_FILE = "/home/darioza/eguardian/data_collection/data/drive_control.log"

def trim_old_events(file_path):
    events = []
    current_time = datetime.now()
    retention_period = timedelta(minutes=DATA_RETENTION_PERIOD)

    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line)
                    event_timestamp = event.get('timestamp')
                    if event_timestamp:
                        event_time = datetime.fromisoformat(event_timestamp)
                        if current_time - event_time <= retention_period:
                            events.append(line)
                except (ValueError, json.JSONDecodeError):
                    log_message(f'Error decoding JSON: {line}')

        if len(events) > MAX_EVENTS:
            events = events[-MAX_EVENTS:]

        with open(file_path, 'w') as f:
            f.writelines(events)

        log_message(f'Arquivo {file_path} truncado para manter apenas os {MAX_EVENTS} eventos mais recentes.')

    except PermissionError:
        log_message(f'Erro de permissão ao acessar o arquivo {file_path}')
    except Exception as e:
        log_message(f'Erro inesperado: {str(e)}')

def apply_sliding_window(output_directory):
    file_paths = [
        os.path.join(output_directory, "file_collected_data.json"),
        os.path.join(output_directory, "net_collected_data.json")
    ]

    for file_path in file_paths:
        if os.path.exists(file_path):
            trim_old_events(file_path)

def log_message(message):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.now()} - {message}\n")