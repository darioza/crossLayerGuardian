#!/bin/bash

# CrossLayerGuardian - Build Script para Programas eBPF
# Compila programas eBPF com dependências necessárias

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EBPF_DIR="$SCRIPT_DIR/data_collection/ebpf_programs"

echo "=== CrossLayerGuardian eBPF Build Script ==="
echo "Compilando programas eBPF para correlação cross-layer..."

# Verificar dependências
check_dependencies() {
    echo "Verificando dependências..."
    
    if ! command -v clang &> /dev/null; then
        echo "ERRO: clang não encontrado. Instale com:"
        echo "  sudo apt-get install clang"
        exit 1
    fi
    
    if ! command -v llc &> /dev/null; then
        echo "ERRO: llc não encontrado. Instale com:"
        echo "  sudo apt-get install llvm"
        exit 1
    fi
    
    if [ ! -d "/usr/include/linux" ]; then
        echo "ERRO: Headers do kernel não encontrados. Instale com:"
        echo "  sudo apt-get install linux-headers-$(uname -r)"
        exit 1
    fi
    
    echo "✓ Dependências verificadas"
}

# Compilar programa de rede
compile_net_monitor() {
    echo "Compilando net_monitor.bpf.c..."
    
    cd "$EBPF_DIR"
    
    clang -O2 -target bpf -c net_monitor.bpf.c -o net_monitor.bpf.o \
        -I/usr/include/x86_64-linux-gnu \
        -I/usr/include \
        -I. \
        -D__KERNEL__ \
        -D__BPF_TRACING__ \
        -Wno-unused-value \
        -Wno-pointer-sign \
        -Wno-compare-distinct-pointer-types
    
    if [ $? -eq 0 ]; then
        echo "✓ net_monitor.bpf.o compilado com sucesso"
    else
        echo "✗ Falha na compilação do net_monitor.bpf.c"
        return 1
    fi
}

# Compilar programa de arquivos
compile_file_monitor() {
    echo "Compilando file_monitor.bpf.c..."
    
    cd "$EBPF_DIR"
    
    clang -O2 -target bpf -c file_monitor.bpf.c -o file_monitor.bpf.o \
        -I/usr/include/x86_64-linux-gnu \
        -I/usr/include \
        -I. \
        -D__KERNEL__ \
        -D__BPF_TRACING__ \
        -Wno-unused-value \
        -Wno-pointer-sign \
        -Wno-compare-distinct-pointer-types
    
    if [ $? -eq 0 ]; then
        echo "✓ file_monitor.bpf.o compilado com sucesso"
    else
        echo "✗ Falha na compilação do file_monitor.bpf.c"
        return 1
    fi
}

# Verificar se está rodando como root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo "AVISO: Para executar o CrossLayerGuardian será necessário privilégios de root"
        echo "Execute com: sudo python3 main.py"
    fi
}

# Função principal
main() {
    check_dependencies
    compile_net_monitor
    compile_file_monitor
    check_root
    
    echo ""
    echo "=== Build Completado ==="
    echo "Programas eBPF compilados com sucesso!"
    echo "Arquivos gerados:"
    echo "  - $EBPF_DIR/net_monitor.bpf.o"
    echo "  - $EBPF_DIR/file_monitor.bpf.o"
    echo ""
    echo "Para executar o CrossLayerGuardian:"
    echo "  sudo python3 main.py"
    echo ""
}

# Executar se chamado diretamente
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi