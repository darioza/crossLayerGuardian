import json
import pickle
from collections import Counter

def converter_para_string(sequencias):
    return [' '.join(seq) for seq in sequencias]

def engenharia_recursos(sequencias):
    frequencia_syscalls = []
    for seq in sequencias:
        contagem = Counter(seq)
        total = sum(contagem.values())
        frequencia = {syscall: count / total for syscall, count in contagem.items()}
        frequencia_syscalls.append(frequencia)
    
    return frequencia_syscalls

def main():
    # Carregar o modelo treinado
    with open('modelos_adfa_ld.pkl', 'rb') as file:
        modelo = pickle.load(file)

    # Carregar o vetorizador
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

    # Carregar o seletor de recursos
    selector = pickle.load(open('selector.pkl', 'rb'))

    # Carregar os dados do arquivo JSON
    with open('file_collected_data.json', 'r') as file:
        dados = [json.loads(line) for line in file]

    # Processar os dados em janelas de 500
    window_size = 100
    janelas = [dados[i:i+window_size] for i in range(0, len(dados), window_size)]

    # Processar cada janela
    for janela in janelas:
        sequencias = [evento['syscall'] for evento in janela]
        sequencias_string = converter_para_string(sequencias)
        recursos = engenharia_recursos(sequencias)

        # Vetorizar os recursos
        recursos_vetorizados = vectorizer.transform(recursos)

        # Aplicar seleção de recursos
        recursos_selecionados = selector.transform(recursos_vetorizados)

        # Fazer previsões usando o modelo
        previsoes = modelo.predict(recursos_selecionados)

        # Imprimir as previsões
        print("Previsões para a janela:")
        print(previsoes)

if __name__ == '__main__':
    main()