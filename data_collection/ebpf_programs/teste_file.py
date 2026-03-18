from bcc import BPF
import time

def load_bpf_program(file_path):
    with open(file_path, 'r') as f:
        bpf_program = f.read()
    bpf = BPF(text=bpf_program)
    return bpf

def main():
    print("Carregando o programa eBPF para monitoramento de arquivos...")
    # Ajuste o caminho para o arquivo diretamente, se o script está no mesmo diretório do código eBPF
    file_monitor = load_bpf_program('file_monitor.bpf.c')
    print("Carregando o programa eBPF para monitoramento de rede...")
    # Supondo que o arquivo de monitoramento de rede esteja no mesmo diretório
    net_monitor = load_bpf_program('net_monitor.bpf.c')
    
    print("Monitoramento iniciado. Coletando dados por 30 segundos...")
    try:
        time.sleep(30)  # Coleta dados por 30 segundos
    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")

    print("Parando o monitoramento e limpando recursos...")
    file_monitor.cleanup()
    net_monitor.cleanup()
    print("Monitoramento encerrado.")

if __name__ == '__main__':
    main()

