import pickle

# Caminho para o modelo treinado
model_path = '/home/darioza/eguardian/models/modelos_nsl_kdd.pkl'

# Carregar o modelo
with open(model_path, 'rb') as file:
    loaded_data = pickle.load(file)

# Imprimir o tipo do objeto carregado e sua representação se for um dicionário
print("Tipo do objeto carregado:", type(loaded_data))
if isinstance(loaded_data, dict):
    print("Chaves do dicionário carregado:", loaded_data.keys())

# Verificar se o modelo está em uma chave específica do dicionário
if 'model' in loaded_data:
    model = loaded_data['model']
    print("Modelo encontrado no dicionário sob a chave 'model'.")
else:
    print("Modelo não encontrado em uma chave esperada. Verifique as chaves listadas acima.")
