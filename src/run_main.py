import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

from config import run_settings, selector_switch

from app_functions.forecast_validation import create_validation_plots
from model_functions.pyAF_model import train_model as train_pyAF
from model_functions.pyAF_model import produce_forecast as prouce_forecast_pyAF


def check_or_create_dir(folder_path):

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        os.makedirs(folder_path + 'forecast/')
        os.makedirs(folder_path + 'scores/')
        os.makedirs(folder_path + 'test_period_forecast/')
        os.makedirs(folder_path + 'validation_plots/')
        os.makedirs(folder_path + 'validation_plots_YoY_growth/')

        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def create_forecast_dates(date_series,steps):
    """ Create Future Forecast Date Range """
    forecast_start_date = date_series.max() + relativedelta(months=1)
    forecast_dates = [forecast_start_date + relativedelta(months=i) for i in range(steps)]  

    return forecast_dates


def main():

    # Check if the directory exists
    iteration_name = run_settings['ITERATION_NAME']
    check_or_create_dir(f'../run_results/{iteration_name}/')

    # System Timestamp
    today = pd.Timestamp.today()

    # Steps
    validation_steps = run_settings['VALIDATION_STEPS']
    forecast_steps = run_settings['FORECAST_STEPS']
    
    # Names
    date_column_name = run_settings['DATE_COLUMN_NAME']
    target_column_name = run_settings['TARGET_COLUMN_NAME']
    exog_column_names = run_settings['EXOG_COLUMN_NAMES']

    # Read Data
    air_passengers = pd.read_csv('../data/air_passengers.csv')
    air_passengers[date_column_name] = pd.to_datetime(air_passengers[date_column_name]) + MonthEnd(0)
    
    train,test = [air_passengers.iloc[0:-validation_steps], air_passengers.iloc[-validation_steps:]]

    # pyAF
    if selector_switch['pyAF'] == True:
        print('Begin pyAF Model Train')
        
        pyAF_horizon, pyAF_mape, pyAF_test_period_forecast = train_pyAF(air_passengers)

        pyAF_forecast =  prouce_forecast_pyAF(pyAF_horizon,air_passengers)
    
    # Create Forecast Dataframe
    forecast_dates = create_forecast_dates(air_passengers[date_column_name], forecast_steps)
    forecast_dataframe = pd.DataFrame(forecast_dates, columns=[date_column_name])
    forecast_dataframe['pyAF_forecast'] = pyAF_forecast
    forecast_dataframe['model_run_ts'] = today
    forecast_dataframe.to_csv(f'../run_results/{iteration_name}/forecast/results.csv', index=False)

    # Create Score Dataframe    
    score_dataframe = pd.DataFrame()
    score_dataframe['pyAF_mape'] = [pyAF_mape]
    score_dataframe['model_run_ts'] = [today]
    score_dataframe['pyAF_params'] = [{'horizon':pyAF_horizon}]
    #print(pyAF_mape)
    score_dataframe.to_csv(f'../run_results/{iteration_name}/scores/results.csv', index=False)

    # Create Test Period Forecast Dataframe  
    forecast_dates = create_forecast_dates(air_passengers[date_column_name], validation_steps)
    test_period_forecast_dataframe = pd.DataFrame(forecast_dates, columns=[date_column_name])
    test_period_forecast_dataframe['pyAF_test_period_forecast'] = pyAF_test_period_forecast
    test_period_forecast_dataframe['model_run_ts'] = today
    test_period_forecast_dataframe.to_csv(f'../run_results/{iteration_name}/test_period_forecast/results.csv', index=False)

    create_validation_plots(air_passengers, forecast_dataframe.rename(columns={'pyAF_forecast':'Passengers'}))
    
if __name__ == "__main__":
    main()

