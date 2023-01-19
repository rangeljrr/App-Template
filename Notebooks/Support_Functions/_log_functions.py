from os.path import exists as file_exists
from Notebooks.Support_Functions._config import performance_logs

def append_txt(path, values):
    if file_exists(f'{performance_logs}'):
        pass
    else:
        with open(performance_logs,'w') as f:
            f.write('Model,Accuracy,R2,RMSE,MAPE,MAE,Timestamp')

    
    with open(path,'a') as f:
        f.write('\n' + values)
