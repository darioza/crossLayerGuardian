import pickle
from queue import Empty
import multiprocessing
import os

# Importar preprocess_data do arquivo net_monitor
from data_collection.net_monitor import preprocess_data as nsl_preprocess_data

# Fila compartilhada para receber dados do net_monitor
nsl_preprocessor_queue = multiprocessing.Queue()

# Fila compartilhada para enviar previsões de volta para o main.py
nsl_prediction_queue = multiprocessing.Queue()

# Obter o diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Carregando o modelo NSL-KDD treinado
nsl_model_path = os.path.join(current_dir, '../machine_learning/trained_models/modelos_nsl_kdd.pkl')
with open(nsl_model_path, 'rb') as f:
    nsl_kdd_model = pickle.load(f)

def predict_nsl(data):
    # Pré-processando os dados
    preprocessed_data = nsl_preprocess_data(data)

    # Realizando a previsão usando o modelo NSL-KDD
    predictions = nsl_kdd_model.predict(preprocessed_data)

    # Enviar as previsões de volta para o main.py
    nsl_prediction_queue.put(predictions)

    return predictions  # Retornar as previsões diretamente

def consume_data():
    while True:
        try:
            data = nsl_preprocessor_queue.get(block=False)
            predict_nsl(data)
        except Empty:
            # A fila está vazia, não há dados para processar
            pass

if __name__ == '__main__':
    # Iniciando o loop para consumir dados da fila compartilhada
    consume_data()