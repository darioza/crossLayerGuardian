import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class ModelPredictor:
    def __init__(self, adfa_model_path, nsl_model_path):
        self.adfa_model = pickle.load(open(adfa_model_path, 'rb'))
        self.nsl_model = pickle.load(open(nsl_model_path, 'rb'))

    def predict_adfa(self, preprocessed_data):
        syscall_sequence = preprocessed_data['syscall_sequence']
        syscall_frequency = preprocessed_data['syscall_frequency']
        
        # Utilizar o modelo ADFA-LD para fazer a previsão
        prediction = self.adfa_model.predict([syscall_sequence, syscall_frequency])
        
        return prediction

    def predict_nsl(self, preprocessed_data):
        # Extrair os recursos relevantes do dicionário preprocessed_data
        protocol_type = preprocessed_data['protocol_type']
        service = preprocessed_data['service']
        flag = preprocessed_data['flag']
        src_bytes = preprocessed_data['src_bytes']
        dst_bytes = preprocessed_data['dst_bytes']
        land = preprocessed_data['land']
        wrong_fragment = preprocessed_data['wrong_fragment']
        urgent = preprocessed_data['urgent']
        hot = preprocessed_data['hot']
        num_failed_logins = preprocessed_data['num_failed_logins']
        logged_in = preprocessed_data['logged_in']
        num_compromised = preprocessed_data['num_compromised']
        root_shell = preprocessed_data['root_shell']
        su_attempted = preprocessed_data['su_attempted']
        num_root = preprocessed_data['num_root']
        num_file_creations = preprocessed_data['num_file_creations']
        num_shells = preprocessed_data['num_shells']
        num_access_files = preprocessed_data['num_access_files']
        num_outbound_cmds = preprocessed_data['num_outbound_cmds']
        is_host_login = preprocessed_data['is_host_login']
        is_guest_login = preprocessed_data['is_guest_login']
        count = preprocessed_data['count']
        srv_count = preprocessed_data['srv_count']
        serror_rate = preprocessed_data['serror_rate']
        srv_serror_rate = preprocessed_data['srv_serror_rate']
        rerror_rate = preprocessed_data['rerror_rate']
        srv_rerror_rate = preprocessed_data['srv_rerror_rate']
        same_srv_rate = preprocessed_data['same_srv_rate']
        diff_srv_rate = preprocessed_data['diff_srv_rate']
        srv_diff_host_rate = preprocessed_data['srv_diff_host_rate']
        dst_host_count = preprocessed_data['dst_host_count']
        dst_host_srv_count = preprocessed_data['dst_host_srv_count']
        dst_host_same_srv_rate = preprocessed_data['dst_host_same_srv_rate']
        dst_host_diff_srv_rate = preprocessed_data['dst_host_diff_srv_rate']
        dst_host_same_src_port_rate = preprocessed_data['dst_host_same_src_port_rate']
        dst_host_srv_diff_host_rate = preprocessed_data['dst_host_srv_diff_host_rate']
        dst_host_serror_rate = preprocessed_data['dst_host_serror_rate']
        dst_host_srv_serror_rate = preprocessed_data['dst_host_srv_serror_rate']
        dst_host_rerror_rate = preprocessed_data['dst_host_rerror_rate']
        dst_host_srv_rerror_rate = preprocessed_data['dst_host_srv_rerror_rate']

        # Criar uma lista com os recursos extraídos
        features = [protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent, hot,
                    num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_root,
                    num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_host_login,
                    is_guest_login, count, srv_count, serror_rate, srv_serror_rate, rerror_rate,
                    srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count,
                    dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate,
                    dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate,
                    dst_host_srv_rerror_rate]

        # Utilizar o modelo NSL-KDD para fazer a previsão
        prediction = self.nsl_model.predict([features])
        
        return prediction