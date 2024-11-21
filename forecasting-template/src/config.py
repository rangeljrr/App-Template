import sys
import pandas as pd
# Need to append system path
app_root = '/Users/rrangel/Downloads/app_example/'
sys.path.insert(0, app_root)

process_run_ts = pd.Timestamp.today().date()
process_run_ts_str = str(pd.Timestamp.today().date()).replace('-','')

run_settings = {
    'run_name' : f'run_{process_run_ts_str}',
    'validation_steps' : 30,
    'forecast_steps' : 12,
    'yearly_steps' : 12, # 12 = Monthly, 52 = Weekly, 365 = Daily
    'date_column' : 'Month',
    'target_column' : 'Passengers',
    'exog_variable_names' : [],
}



pyAF_settings = {
    'horizon_steps': [i for i in range(2,29,2)],
    #'horizon_steps': [2]
}