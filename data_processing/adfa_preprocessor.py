import pickle
from queue import Empty
import multiprocessing
import os

# Importar preprocess_data do arquivo file_monitor
from data_collection.file_monitor import preprocess_data as adfa_preprocess_data

# Fila compartilhada para receber dados do file_monitor
adfa_preprocessor_queue = multiprocessing.Queue()

# Fila compartilhada para enviar previsões de volta para o main.py
adfa_prediction_queue = multiprocessing.Queue()

# Obter o diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Carregando o modelo ADFA-LD treinado
adfa_model_path = os.path.join(current_dir, '../machine_learning/trained_models/modelos_adfa_ld.pkl')
with open(adfa_model_path, 'rb') as f:
    adfa_ld_model = pickle.load(f)

def predict_adfa(data):
    # Pré-processando os dados
    preprocessed_data = adfa_preprocess_data(data)

    # Realizando a previsão usando o modelo ADFA-LD
    predictions = adfa_ld_model.predict(preprocessed_data)

    # Enviar as previsões de volta para o main.py
    adfa_prediction_queue.put(predictions)

    return predictions  # Retornar as previsões diretamente

def consume_data():
    while True:
        try:
            data = adfa_preprocessor_queue.get(block=False)
            predict_adfa(data)
        except Empty:
            # A fila está vazia, não há dados para processar
            pass

if __name__ == '__main__':
    # Iniciando o loop para consumir dados da fila compartilhada
    consume_data()