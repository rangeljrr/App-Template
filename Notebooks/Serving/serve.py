import pandas as pd
from datetime import datetime as dt

from Notebooks.Support_Functions._pkl_functions import load_model
from Notebooks.Support_Functions._config import model_path, model_version, prediction_logs, train_csv_path
from Notebooks.Support_Functions._data_functions import load_real_time_data
#from Notebooks.Support_Functions._log_functions import append_log

def run_serve():
    now = str(dt.now())
    # Need to load the model into memory
    model = load_model(model_path)

    # Need to load real time data
    real_time_data = load_real_time_data(train_csv_path)
    del real_time_data['target']

    # Predict on the data
    predictions = model.predict(real_time_data)

    # Add some additional inforation
    real_time_data['predictions'] = predictions
    real_time_data['timestamp'] = now
    real_time_data['model'] = model_version

    # Store predicitons as .csv logs
    real_time_data.to_csv(prediction_logs.format(now.replace(':','').replace('.','')), index=False)
