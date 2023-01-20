import pandas as pd
from Notebooks.Support_Functions._config import airfoil_path, train_csv_path

def load_real_time_data(path):
    """ This function will load real time data and return a dataframe"""

    return pd.read_csv(path)

def create_train_data():
    """ This function will create the train data and save it as a train_data.csv"""

    data = pd.read_csv(airfoil_path)
    data.to_csv(train_csv_path, index=False)