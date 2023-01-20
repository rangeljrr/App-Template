from os.path import exists as file_exists
from Notebooks.Support_Functions._config import performance_logs

def append_log(path, values):
    """ This file will check if there is a file that already exists, if not then it 
        will create a new .txt file with heaters
        
        The function will then write the accuracy metrics to the file"""

    if file_exists(f'{performance_logs}'):
        pass
    else:
        with open(performance_logs,'w') as f:
            f.write('Model,Accuracy,R2,RMSE,MAPE,MAE,Timestamp')

    
    with open(path,'a') as f:
        f.write('\n' + values)
