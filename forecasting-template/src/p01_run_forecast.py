import pandas as pd
import os
import itertools
from pandas.tseries.offsets import MonthEnd
from validation_functions.scoring_functions import mape

from config import run_settings, process_run_ts

from validation_functions.forecast_validation import create_validation_plots
from validation_functions.scoring_functions import run_all_metrics

# Prophet Untuned
from model_functions.prophet import train_model as train_prophet
from model_functions.prophet import produce_forecast as produce_forecast_prophet

# Prophet Tuned
from model_functions.prophet_tuned import train_model as train_prophet_tuned
from model_functions.prophet_tuned import produce_forecast as produce_forecast_prophet_tuned

# Arima 
from model_functions.arima import train_model as train_arima
from model_functions.arima import produce_forecast as produce_forecast_arima

# Sarima 
from model_functions.sarima import train_model as train_sarima
from model_functions.sarima import produce_forecast as produce_forecast_sarima

from app_setup.directories import check_or_create_dir
from data_preparation_functions.clean_data import clean_air_passengers
from data_preparation_functions.feature_engineering import create_features
from data_preparation_functions.feature_engineering import create_forecast_dates

def main():

    # Check if the directory exists
    iteration_name = run_settings['run_name']
    check_or_create_dir(f'../run_results/{iteration_name}/')
    check_or_create_dir(f'../run_results/latest/')
    
    # System Timestamp
    today = pd.Timestamp.today()
    
    # Steps
    validation_steps = run_settings['validation_steps']
    forecast_steps = run_settings['forecast_steps']
    forecast_steps = 30
    # Names
    date_column_name = run_settings['date_column']
    target_column_name = run_settings['target_column']
    exog_column_names = run_settings['exog_variable_names']
    
    # Read Data
    air_passengers = pd.read_csv('../data/air_passengers.csv')
    air_passengers[date_column_name] = pd.to_datetime(air_passengers[date_column_name]) + MonthEnd(0)
    
    # Clean dataset
    air_passengers = clean_air_passengers(air_passengers)
    
    # Create Features
    air_passengers = create_features(air_passengers)
    
    # Train Test Split
    train,test = [air_passengers.iloc[0:-validation_steps], air_passengers.iloc[-validation_steps:]]

    params = []
    forecast = []
    
    for i in range(1):# Prophet
    
        type_ = 'type'
    
        # Prophet Model
        prophet_params, prophet_error, prophet_test_period_forecast = train_prophet(train, test, validation_metric='mape')
        prophet_forecast = produce_forecast_prophet(air_passengers,prophet_params, forecast_steps)
        
        # Prophet Model Tuned
        prophet_tuned_params, prophet_tuned_error, prophet_tuned_test_period_forecast = train_prophet_tuned(train, test, validation_metric='mape')
        prophet_tuned_forecast =  produce_forecast_prophet_tuned(air_passengers,prophet_tuned_params, forecast_steps)
    
        # Arima Model
        arima_params, arima_error, arima_test_period_forecast = train_arima(train['Passengers'], test['Passengers'], 'mape')
        arima_forecast = produce_forecast_arima(air_passengers['Passengers'],arima_params, forecast_steps)
    
        # Sarima Model
        sarima_params, sarima_error, sarima_test_period_forecast = train_sarima(train['Passengers'], test['Passengers'], 'mape')
        sarima_forecast = produce_forecast_sarima(air_passengers['Passengers'],sarima_params, forecast_steps)
    
        # Recording params
        insert_params = pd.DataFrame()
        insert_params['label'] = 'type'
        insert_params['model_run_ts'] = today
        insert_params['prophet'] = [prophet_params]
        insert_params['prophet_tuned'] = [prophet_tuned_params]
        insert_params['arima'] = [arima_params]
        insert_params['sarima'] = [sarima_params]
        params.append(insert_params)
        
        # Train Period
        insert_train = train[['Month','Passengers']]
        insert_train['label'] = 'type'
        insert_train['model_run_ts'] = today
        insert_train['period'] = 'TRAIN'
        
        # Test Period
        insert_test_period_forecast = test[[date_column_name]]
        insert_test_period_forecast['label'] = 'type'
        insert_test_period_forecast['model_run_ts'] = today
        insert_test_period_forecast['period'] = 'TEST'
        insert_test_period_forecast['Passengers'] = test['Passengers'].values
        insert_test_period_forecast['prophet'] = prophet_test_period_forecast
        insert_test_period_forecast['prophet_tuned'] = prophet_tuned_test_period_forecast
        insert_test_period_forecast['arima'] = arima_test_period_forecast
        insert_test_period_forecast['sarima'] = sarima_test_period_forecast
    
        # Forecast Period
        insert_forecast = pd.DataFrame()
        insert_forecast['Month'] = create_forecast_dates(air_passengers['Month'], forecast_steps)
        insert_forecast['label'] = 'type'
        insert_forecast['model_run_ts'] = today
        insert_forecast['period'] = 'FCST'
        insert_forecast['Passengers'] = float('NaN')
        insert_forecast['prophet'] = prophet_forecast
        insert_forecast['prophet_tuned'] = prophet_tuned_forecast
        insert_forecast['arima'] = arima_forecast
        insert_forecast['sarima'] = sarima_forecast
        
        insert_all = pd.concat([
            insert_train,
            insert_test_period_forecast,
            insert_forecast
        ])
        
        forecast.append(insert_all)
    
    # Test Period Forecast Master
    params = pd.concat(params)
    params['model_run_ts'] = today
    params.to_csv(f'../run_results/{iteration_name}/params/results.csv', index=False)
    
    # Forecast Master
    forecast = pd.concat(forecast)
    forecast.to_csv(f'../run_results/{iteration_name}/forecast/results.csv', index=False)
    forecast.to_csv(f'../run_results/latest/forecast/results.csv', index=False)


    # Create Ensembles
    forecast_columns = forecast.columns[5:]
    
    all_combinations = []
    for r in range(2, len(forecast_columns) + 1):  # Start from combinations of size 2
        all_combinations.extend(itertools.combinations(forecast_columns, r))
    
    # Compute averages for each combination
    for combo in all_combinations:
        combo_name = "_".join(combo)
        forecast[f"ensemble_{combo_name}"] = forecast[list(combo)].mean(axis=1)
    
    # Computing scores
    scores = []
    for label in forecast['label'].unique():
        iter_data = forecast[(forecast['label'] == label) & (forecast['period'] == 'TEST')]
    
        insert_score = pd.DataFrame()
        insert_score['model_run_ts'] = [pd.to_datetime(forecast['model_run_ts'].iloc[0])]
        insert_score['label'] = [label]
        
        for c in iter_data.columns[5:]:
            insert_score[c] = mape(iter_data['Passengers'], iter_data[c])
    
        scores.append(insert_score)
    
    scores = pd.concat(scores)
    scores.to_csv(f'../run_results/latest/scores/results.csv', index=False)
    scores.to_csv(f'../run_results/{iteration_name}/scores/results.csv', index=False)

    forecast.to_csv(f'../run_results/latest/forecast/results.csv', index=False)
    forecast.to_csv(f'../run_results/{iteration_name}/forecast/results.csv', index=False)

if __name__ == "__main__":
    main()

