from bcc import BPF

# Carrega o código BPF do arquivo
bpf = BPF(src_file="net_monitor.bpf.c")

# Anexa a função eBPF ao tracepoint de rede adequado
bpf.attach_tracepoint(tp="net:net_dev_start_xmit", fn_name="trace_net_packet")

# Imprime o output dos eventos capturados
print("Tracing network packets... Press Ctrl+C to end.")
try:
    bpf.trace_print()
except KeyboardInterrupt:
    print("Exiting...")
