import sys

# Need to append system path
app_root = '/Users/rrangel/Downloads/app_example/'
sys.path.insert(0, app_root)

run_settings = {
    'ITERATION_NAME' : 'iteration 2',
    'VALIDATION_STEPS' : 30,
    'FORECAST_STEPS' : 12,
    'YEARLY_STEPS' : 12, # 12 = Monthly, 52 = Weekly, 365 = Daily
    'DATE_COLUMN_NAME' : 'Month',
    'TARGET_COLUMN_NAME' : 'Passengers',
    'EXOG_COLUMN_NAMES' : [],
}

selector_switch = {
        'pyAF' : True,
        'prophet' : False,
        'saimax' : False
    }

pyAF_settings = {
    'horizon_steps': [i for i in range(2,29,2)],
    #'horizon_steps': [2]
}