from flask import Flask, render_template
import os
import json
from collections import deque
from datetime import datetime
import json
from flask import jsonify
import sys

# Obter o diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adicionar o diretório 'user_interface' ao sys.path
user_interface_dir = os.path.join(current_dir, 'user_interface')
sys.path.append(user_interface_dir)

from accumulated_stats import AccumulatedStats

app = Flask(__name__)

data_dir = "/home/darioza/eguardian/data_collection/data"
attack_packets = deque(maxlen=10)
accumulated_stats = AccumulatedStats()

def read_json_events(file_path):
    events = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):  # Verificar se o evento é um dicionário
                        events.append(event)
                    else:
                        print(f"Evento mal-formatado: {line}")
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Erro decodificando JSON: {line}")
                    print(f"Erro: {e}")
                continue
    return events

def format_event_str(event, file_name):
    event_str = ""
    css_class = "file-event" if file_name == "file_collected_data.json" else "net-event"
    if 'process_name' in event and 'pid' in event:
        event_str = f"{event['event']} - {event['process_name']} ({event['pid']})"
        file_path = event.get('file_path', '')
        if file_path:
            event_str += f" - {file_path}"
    elif 'event' in event:
        event_str = event.get('event', '')
        for key, value in event.items():
            if key not in ['event', 'saddr', 'daddr', 'laddr']:
                event_str += f" {key}={value}"
        if 'saddr' in event:
            event_str += f" saddr={event['saddr']}"
        if 'daddr' in event:
            event_str += f" daddr={event['daddr']}"
        if 'laddr' in event:
            event_str += f" laddr={event['laddr']}"
        if 'protocol_type' in event:
            event_str += f" protocol_type={event['protocol_type']}"
        if 'service' in event:
            event_str += f" service={event['service']}"

    return f"{event_str}|{file_name}"

@app.route('/')
def index():
    json_files = ['file_collected_data.json', 'net_collected_data.json']
    events = []
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        events.extend([(event, json_file) for event in read_json_events(file_path)])

    events.sort(key=lambda x: x[0].get('timestamp', 0))

    for event, file_name in events:
        accumulated_stats.update_stats(event)

    return render_template('index.html', events=[(format_event_str(event, file_name), file_name) for event, file_name in events[-10:]], accumulated_stats=accumulated_stats, attack_packets=attack_packets)

@app.route('/data', methods=['GET'])
def get_data():
    json_files = ['file_collected_data.json', 'net_collected_data.json']
    events = []
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        events.extend([(event, json_file) for event in read_json_events(file_path)])

    events.sort(key=lambda x: x[0].get('timestamp', 0))

    for event, file_name in events:
        accumulated_stats.update_stats(event)

    data = {
        'events': [format_event_str(event, file_name) for event, file_name in events[-10:]],
        'accumulated_stats': {
            'cumulative_counts': accumulated_stats.cumulative_counts,
            'process_counts': accumulated_stats.process_counts,
            'file_op_counts': accumulated_stats.file_op_counts,
            'user_counts': accumulated_stats.user_counts,
            'src_ip_counts': accumulated_stats.src_ip_counts,
            'src_network_counts': accumulated_stats.src_network_counts,
            'dst_network_counts': accumulated_stats.dst_network_counts,
            'top_files': accumulated_stats.top_files,
            'port_counts': accumulated_stats.port_counts,
            'pid_counts': accumulated_stats.pid_counts,
        },
        'attack_packets': list(attack_packets)
    }

    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)