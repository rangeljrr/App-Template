import pandas as pd
import sys
from datetime import datetime as dt

from Notebooks.Support_Functions._config import train_csv_path, model_path, performance_logs
from Notebooks.Support_Functions._config import train_feature_name
from Notebooks.Support_Functions._data_functions import create_train_data
from Notebooks.Support_Functions._log_functions import append_log
from Notebooks.Support_Functions._pkl_functions import serialize_model
from Notebooks.Training._training_functions import reg_evaluate, train_model

def run_train(model_version):

    # Create the train data
    create_train_data()

    # Read in train data
    train_data = pd.read_csv(train_csv_path)

    # Fit the machine learning model
    X,y = train_data.drop([train_feature_name], axis=1), train_data[train_feature_name]
    model = train_model(X,y)
    
    # Need to serialize and dump model
    serialize_model(model, model_path)

    # Will also need to log accuracies --> Accuracy | R2 | RMSE | MAPE | MAE]
    valdation_metrics = reg_evaluate(model, X, y)
    valdation_metrics.append(str(dt.now()))
    valdation_metrics.insert(0,model_version)

    # Nede to store validation values into performance logs
    append_log(performance_logs, ','.join(map(str, valdation_metrics)))

