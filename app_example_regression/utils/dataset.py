# Example Data (replace with your real X and y)
from sklearn.datasets import load_diabetes
import pandas as pd
def load_historical():
    
    # Load historical (training-like) dataset
    master_hist = load_diabetes()
    X, y = [master_hist.data, master_hist.target]
    master_hist = pd.DataFrame(master_hist.data, columns=master_hist.feature_names)
    master_hist['target'] = y
    
    return master_hist


def load_live():
    
    # Simulate live (test-like) dataset
    master_live = load_diabetes()
    X, y = [master_live.data, master_live.target]
    master_live = pd.DataFrame(master_live.data, columns=master_live.feature_names)
    master_live['target'] = y
    
    return master_live