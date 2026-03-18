import os
import shutil
import time
import subprocess

def perform_file_operations():
    # Diretório para testes
    test_dir = "/tmp/test_monitoring"

    # Criar um diretório de teste
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    print("# Criar um diretório de teste")

    # Caminho do arquivo para testes
    test_file = os.path.join(test_dir, "test_file.txt")
    test_file_renamed = os.path.join(test_dir, "test_file_renamed.txt")
    print("# Caminho do arquivo para teste")

    # Criando e escrevendo em um arquivo
    with open(test_file, 'w') as f:
        f.write("Este é um teste de escrita.\n")

    # Lendo o arquivo
    with open(test_file, 'r') as f:
        print(f.read())
    print("# Lendo o arquivo")

    # Renomeando o arquivo
    os.rename(test_file, test_file_renamed)
    print("# Renomeando o arquivo")

    # Criando um diretório dentro do diretório de teste
    test_subdir = os.path.join(test_dir, "subdir")
    os.makedirs(test_subdir)
    print("# Criando um diretório dentro do diretório de teste")

    # Removendo o diretório criado (com o conteúdo)
    shutil.rmtree(test_subdir)
    print("# Removendo o diretório criado (com o conteúdo)")

    # Removendo o arquivo renomeado
    os.remove(test_file_renamed)
    print("# Removendo o arquivo renomeado")

    # Limpando o diretório de teste ao final
    os.rmdir(test_dir)
    print("Teste de monitoramento de syscalls concluído.")

    # Tentando mudar o UID (necessário permissões de root)
    try:
        os.setuid(1000)  # Muda o UID do processo se possível
        print("# Mudança de UID efetuada")
    except PermissionError:
        print("# Falha na mudança de UID, permissões insuficientes")

    # Simulando carregamento e descarregamento de módulo do kernel (root necessário)
    try:
        subprocess.run(['modprobe', 'dummy'], check=True)
        print("# Módulo dummy carregado")
        subprocess.run(['modprobe', '-r', 'dummy'], check=True)
        print("# Módulo dummy descarregado")
    except Exception as e:
        print(f"# Erro ao manipular módulos do kernel: {e}")

# Loop infinito para executar as operações de arquivo continuamente
while True:
    perform_file_operations()
    print("Aguardando próxima iteração...")
    time.sleep(10)  # Pausa de 10 segundos entre as iterações
