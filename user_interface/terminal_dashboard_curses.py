import curses
import curses.ascii
import time
from collections import deque
import os
import sys
from datetime import datetime
import json
sys.path.append('/home/darioza/eguardian/user_interface/')

from accumulated_stats import AccumulatedStats

MIN_TERMINAL_HEIGHT = 40

def check_terminal_capabilities():
    try:
        curses.setupterm()
        if curses.tigetstr("colors") is None:
            return False
        if curses.tigetstr("cup") is None:
            return False
        height, width = curses.initscr().getmaxyx()
        curses.endwin()
        if height < MIN_TERMINAL_HEIGHT:
            return False
        return True
    except curses.error:
        return False

class TerminalDashboard:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.accumulated_stats = AccumulatedStats()
        self.attack_packets = deque(maxlen=10)

    def run(self, screen):
        print("Iniciando o Terminal Dashboard")
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()

        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

        while True:
            screen.clear()
            self.render_dashboard(screen)
            screen.refresh()
            time.sleep(0.1)

    def read_json_events(self, file_path):
        events = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except (json.JSONDecodeError, ValueError):
                        print(f"Erro ao decodificar JSON: {line}")
                    continue
        return events

    def render_dashboard(self, screen):
        try:
            height, width = screen.getmaxyx()
            half_width = width // 2

            # Render header
            header = "eGuardian HIDS"

            # Render eGuardian logo
            logo_lines = [
                "        _____                     _ _             ",
                "       / ____|                   | (_)            ",
                "   ___| |  __ _   _  __ _ _ __ __| |_  __ _ _ __  ",
                "  / _ \ | |_ | | | |/ _` | '__/ _` | |/ _` | '_ \ ",
                " |  __/ |__| | |_| | (_| | | | (_| | | (_| | | | |",
                "  \___|\_____|\__,_|\__,_|_|  \__,_|_|\__,_|_| |_|",
            ]
            logo_row = 0
            for line in logo_lines:
                screen.addstr(logo_row,
                              20,
                              line,
                              curses.color_pair(1)
                              )
                logo_row += 1

            # Render events
            event_row = logo_row + 2
            screen.addstr(event_row, 0, "Eventos:", curses.color_pair(2))
            event_row += 1

            # Ler eventos dos arquivos JSON
            json_files = ['file_collected_data.json', 'net_collected_data.json']
            events = []
            for json_file in json_files:
                file_path = os.path.join(self.data_dir, json_file)
                events.extend(self.read_json_events(file_path))

            # Ordenar eventos por timestamp
            events.sort(key=lambda x: x.get('timestamp', 0))

            # Renderizar eventos
            for event in events[-10:]:  # Mostrar apenas os 10 eventos mais recentes
                event_str = self.format_event_str(event)
                screen.addstr(event_row, 2, event_str, curses.color_pair(3))
                event_row += 1

            # Atualizar estatísticas acumuladas
            for event in events:
                self.accumulated_stats.update_stats(event)

            # Renderizar estatísticas acumuladas
            count_row = event_row + 1
            screen.addstr(count_row, 0, "Contagens de eventos:", curses.color_pair(2))
            count_row += 1
            for event_type, count in sorted(self.accumulated_stats.cumulative_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
                screen.addstr(count_row, 2, f"{event_type}: {count}", curses.color_pair(3))
                count_row += 1

            # Render process counts
            process_row = count_row + 1
            screen.addstr(process_row, 0, "Top processos:", curses.color_pair(2))
            process_row += 1
            for process, count in sorted(self.accumulated_stats.process_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                screen.addstr(process_row, 2, f"{process}: {count}", curses.color_pair(4))
                process_row += 1

            # Render file operation counts
            file_op_row = process_row + 1
            screen.addstr(file_op_row, 0, "Operações de arquivo:", curses.color_pair(2))
            file_op_row += 1
            for file_op, count in sorted(self.accumulated_stats.file_op_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                screen.addstr(file_op_row, 2, f"{file_op}: {count}", curses.color_pair(3))
                file_op_row += 1

            # Render user counts
            user_row = file_op_row + 1
            screen.addstr(user_row, 0, "Contagens por usuário:", curses.color_pair(2))
            user_row += 1
            for user_id, count in sorted(self.accumulated_stats.user_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                screen.addstr(user_row, 2, f"UID {user_id}: {count}", curses.color_pair(3))
                user_row += 1

            # Render tops (metade direita da tela)
            top_row = event_row + 1
            screen.addstr(top_row, half_width + 2, "Tops:", curses.color_pair(2))
            top_row += 1

            # Render source IP counts
            screen.addstr(top_row, half_width + 2, "Endereços IP de origem:", curses.color_pair(3))
            top_row += 1
            for src_ip, count in sorted(self.accumulated_stats.src_ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                screen.addstr(top_row, half_width + 4, f"{src_ip}: {count}", curses.color_pair(4))
                top_row += 1

            # Render source network counts
            top_row += 1
            screen.addstr(top_row, half_width + 2, "Redes de origem:", curses.color_pair(3))
            top_row += 1
            for src_net, count in sorted(self.accumulated_stats.src_network_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                if src_net:  # Verificar se a rede de origem não é vazia
                    screen.addstr(top_row, half_width + 4, f"{src_net}: {count}", curses.color_pair(4))
                    top_row += 1

            # Render destination network counts
            top_row += 1
            screen.addstr(top_row, half_width + 2, "Redes de destino:", curses.color_pair(3))
            top_row += 1
            for dst_net, count in sorted(self.accumulated_stats.dst_network_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                if dst_net:  # Verificar se a rede de destino não é vazia
                    screen.addstr(top_row, half_width + 4, f"{dst_net}: {count}", curses.color_pair(4))
                    top_row += 1

            # Render top files
            top_row += 1
            screen.addstr(top_row, half_width + 2, "Arquivos:", curses.color_pair(3))
            top_row += 1
            for file_path, count in sorted(self.accumulated_stats.top_files.items(), key=lambda x: x[1], reverse=True)[:5]:
                screen.addstr(top_row, half_width + 4, f"{file_path}: {count}", curses.color_pair(4))
                top_row += 1

            # Render port counts
            port_row = user_row + 2
            screen.addstr(port_row, 0, "Top portas:", curses.color_pair(2))
            port_row += 1
            for port, count in sorted(self.accumulated_stats.port_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                screen.addstr(port_row, 2, f"Porta {port}: {count}", curses.color_pair(3))
                port_row += 1

            # Render PID counts
            pid_row = port_row + 1
            screen.addstr(pid_row, 0, "Top PIDs:", curses.color_pair(2))
            pid_row += 1
            for pid, count in sorted(self.accumulated_stats.pid_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                screen.addstr(pid_row, 2, f"PID {pid}: {count}", curses.color_pair(4))
                pid_row += 1

            # Renderizar pacotes de ataque
            attack_row = pid_row + 1
            screen.addstr(attack_row, 0, "Pacotes de Ataque:", curses.color_pair(2))
            attack_row += 1

            for packet in self.attack_packets:
                screen.addstr(attack_row, 2, str(packet), curses.color_pair(3))
                attack_row += 1

        except curses.error as e:
            print(f"Erro curses: {e}")

    def add_attack_packet(self, packet):
        self.attack_packets.append(packet)

    def get_event_type(self, event):
        if 'type' in event:
            return event['type']
        elif 'event' in event:
            return event['event']
        else:
            return ' - '

    def format_event_str(self, event):
        event_str = ""
        if 'process_name' in event and 'pid' in event:
            event_str = f"{event['event']} - {event['process_name']} ({event['pid']})"
            file_path = event.get('file_path', '')
            if file_path:
                event_str += f" - {file_path}"
        elif 'event' in event:
            event_str = f"{event['event']}"
            for key, value in event.items():
                if key not in ['event', 'saddr', 'daddr', 'laddr']:
                    event_str += f" {key}={value}"
            if 'saddr' in event:
                event_str += f" saddr={event['saddr']}"
            if 'daddr' in event:
                event_str += f" daddr={event['daddr']}"
            if 'laddr' in event:
                event_str += f" laddr={event['laddr']}"

        return event_str

def start_terminal_dashboard(data_dir):
    dashboard = TerminalDashboard(data_dir)
    curses.wrapper(dashboard.run)

if __name__ == "__main__":
    if not check_terminal_capabilities():
        print("O terminal atual não suporta as capacidades necessárias para o eGuardian.")
        sys.exit(1)

    data_dir = "/home/darioza/eguardian/data_collection/data"
    start_terminal_dashboard(data_dir)