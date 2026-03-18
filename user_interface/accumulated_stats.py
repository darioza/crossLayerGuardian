# accumulated_stats.py
from collections import defaultdict

class AccumulatedStats:
    def __init__(self):
        self.cumulative_counts = defaultdict(lambda: 0)
        self.process_counts = defaultdict(lambda: 0)
        self.file_op_counts = defaultdict(lambda: 0)
        self.user_counts = defaultdict(lambda: 0)
        self.src_ip_counts = defaultdict(lambda: 0)
        self.src_network_counts = defaultdict(lambda: 0)
        self.dst_network_counts = defaultdict(lambda: 0)
        self.top_files = defaultdict(lambda: 0)
        self.port_counts = defaultdict(lambda: 0)
        self.pid_counts = defaultdict(lambda: 0)

    def update_stats(self, event):
        event_type = self.get_event_type(event)
        self.cumulative_counts[event_type] += 1
        self.protocol_type_counts = defaultdict(int)
        
        protocol_type = event.get('protocol_type', '')
        if protocol_type:
            self.protocol_type_counts[protocol_type] += 1
            
        service = event.get('service', '')
        if service:
            self.service_counts[service] += 1

        process_name = event.get('comm', '')
        if process_name:
            self.process_counts[process_name] += 1

        pid = event.get('pid', '')
        if pid:
            self.pid_counts[pid] += 1

        file_op = event.get('event', '').split('_')[0]
        if file_op:
            self.file_op_counts[file_op] += 1

        user_id = event.get('uid', '')
        self.user_counts[user_id] += 1

        src_ip = event.get('saddr', '')
        if src_ip:
            self.src_ip_counts[src_ip] += 1
            src_network = '.'.join(src_ip.split('.')[:2]) + '.'
            self.src_network_counts[src_network] += 1

        dst_ip = event.get('daddr', '')
        if dst_ip:
            dst_network = '.'.join(dst_ip.split('.')[:2]) + '.'
            self.dst_network_counts[dst_network] += 1

        file_path = event.get('fname', '')
        self.top_files[file_path] += 1

        src_port = event.get('sport', '')
        if src_port:
            self.port_counts[src_port] += 1

        dst_port = event.get('dport', '')
        if dst_port:
            self.port_counts[dst_port] += 1

    def get_event_type(self, event):
        if 'type' in event:
            return event['type']
        elif 'event' in event:
            return event['event']
        else:
            return ' - '