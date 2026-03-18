import json
import pandas as pd
import pickle
import socket
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Carregar os modelos
model_path = '/home/darioza/eguardian/models/modelos_nsl_kdd.pkl'
with open(model_path, 'rb') as file:
    models = pickle.load(file)

# Escolher o modelo baseado na categoria de ataque
model = models['DoS']

# Função para processar cada entrada JSON
def process_json_line(line):
    try:
        data = json.loads(line)
        return {
            "protocol_type": data.get("protocol"),
            "service": "http",  # Exemplo fixo, ajuste conforme necessário
            "flag": "SF",       # Exemplo fixo, ajuste conforme necessário
            "src_bytes": ip_to_bytes(data.get("saddr")),
            "dst_bytes": ip_to_bytes(data.get("daddr"))
        }
    except json.JSONDecodeError:
        return None

# Função para converter IP em bytes
def ip_to_bytes(ip):
    try:
        return int.from_bytes(socket.inet_aton(ip), 'big')
    except socket.error:
        return 0  # Retorna 0 para IPs inválidos

# Carregar dados JSON e preparar dados
log_path = '/home/darioza/eguardian/data_collection/data/net_collected_data.json'
data_list = []

with open(log_path, 'r') as file:
    for line in file:
        processed_data = process_json_line(line)
        if processed_data:
            data_list.append(processed_data)

# Criar DataFrame
df = pd.DataFrame(data_list)

# Codificar variáveis categóricas
label_encoders = {}
for column in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Adicionar colunas faltantes com valores padrão
default_values = {
    "duration": 0, "land": 0, "wrong_fragment": 0, "urgent": 0,
    "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
    "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
    "is_host_login": 0, "is_guest_login": 0, "count": 1, "srv_count": 1,
    "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0, "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0, "dst_host_count": 1, "dst_host_srv_count": 1,
    "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
}
for col, value in default_values.items():
    df[col] = df.get(col, pd.Series([value] * len(df)))

# Escalonar características numéricas
scaler = StandardScaler()
numeric_features = ["src_bytes", "dst_bytes"]
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Predição usando o modelo selecionado
predictions = model.predict(df.drop(['protocol_type', 'service', 'flag'], axis=1))

# Adicionar previsões ao DataFrame
df['prediction'] = predictions

# Exibir resultados
print(df)

# Salvar ou processar os resultados conforme necessário
df.to_csv('predictions.csv', index=False)
