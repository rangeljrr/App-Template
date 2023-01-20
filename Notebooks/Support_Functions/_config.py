# Environment type
env = 'dev'

# Paths
root = 'C:/Users/yrr8790/OneDrive - HMESRETL/Desktop/Project-Template-Prd/'
airfoil_path = root + 'Data/airfoil_self_noise.csv'
train_csv_path = root + 'Data/train_data.csv'
train_feature_name = 'target'

# Model Version
model_version = 'model2'
model_path = root + f'Models/{model_version}.pkl'

# Database
DRIVER = '{ODBC Driver 17 for SQL Server}'
DATABASE, SERVER = ['', '']
PASSWORD, USERNAME = ['','']

# Log Paths
performance_logs = root + 'Logs/Performance_Logs/log.csv'
prediction_logs = root + 'Logs/Prediction_Logs/{}.csv'