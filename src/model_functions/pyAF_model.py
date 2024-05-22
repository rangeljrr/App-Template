import numpy as np
import pandas as pd
import pyaf.ForecastEngine as autof

from app_functions.scoring_functions import mape
from config import pyAF_settings, run_settings
import matplotlib.pyplot as plt

def build_model_object():
    print('Build Model Object')
    """ Build the forecaster model object """
    forecaster = autof.cForecastEngine()

    # Lines below uses Cross Validation, K=5 Folds, Validate Using MAPE
    forecaster.mOptions.mCrossValidationOptions.mMethod = "TSCV"
    forecaster.mOptions.mCrossValidationOptions.mNbFolds = 5
    forecaster.mOptions.mModelSelection_Criterion = 'MAPE'
    
    return forecaster


def train_model(dataframe):
    print('Train Model Object')
    """ Build a model object and train the model to find the best horizon"""

    # Steps
    validation_steps = run_settings['VALIDATION_STEPS']
    yearly_steps = run_settings['YEARLY_STEPS']
    
    # Names = 
    date_column_name = run_settings['DATE_COLUMN_NAME']
    target_column_name = run_settings['TARGET_COLUMN_NAME']
    exog_column_names = run_settings['EXOG_COLUMN_NAMES']
    iteration_name = run_settings['ITERATION_NAME']

    # Column Names
    cv_scores = []
    cols = [date_column_name] + exog_column_names + [target_column_name]

    for h in pyAF_settings['horizon_steps']:

        # Initializing Model
        model = build_model_object()

        # Fitting Model
        model.train(dataframe[cols], date_column_name, target_column_name, h)
        forecast_results = model.forecast(dataframe[cols], validation_steps)

        # Test Period Forecast
        test_period_forecast = forecast_results[[target_column_name,target_column_name + '_Forecast']].dropna().iloc[-validation_steps:][target_column_name + '_Forecast'].values
        test_period_actuals = forecast_results[[target_column_name,target_column_name + '_Forecast']].dropna().iloc[-validation_steps:][target_column_name].values
        test_mape = mape(test_period_actuals, test_period_forecast)
        
        # Mape against seasonal (Horizon Selection)
        #current_year = pd.Timestamp.now().year
        current_year = 1960
        last_year_1 = dataframe[(dataframe[date_column_name] >= f'{current_year-1}-01-01') & 
                              (dataframe[date_column_name] <= f'{current_year-1}-12-31')]

        last_year_2 = dataframe[(dataframe[date_column_name] >= f'{current_year-2}-01-01') & 
                              (dataframe[date_column_name] <= f'{current_year-2}-12-31')]
        
        # Find Growth between last year and two years ago and setting floor, ceil
        avg_last_year_1 = np.mean(last_year_1[target_column_name].iloc[-validation_steps:])
        avg_last_year_2 = np.mean(last_year_2[target_column_name].iloc[-validation_steps:])
        
        growth = (avg_last_year_1 - avg_last_year_2) / avg_last_year_2
        #growth = max(-.10, min(growth, .10))
        
        # Offsetting 
        offset = validation_steps - yearly_steps

        if offset == 0:
            # If yearly steps line up to validation steps, no need to offset
            forecast_values = forecast_results.iloc[-validation_steps:][target_column_name + '_Forecast'].values
            yoy_seasonal_forecast = (dataframe.iloc[-validation_steps:][target_column_name] * (1 + growth)).values
            prev_year_values = dataframe.iloc[-validation_steps:][target_column_name].values

        elif offset > 0:
            # If yearly steps < validation steps will need to offset forecast
            forecast_values = forecast_results.iloc[-validation_steps-offset:-offset][target_column_name + '_Forecast'].values
            yoy_seasonal_forecast = (dataframe.iloc[-validation_steps:][target_column_name] * (1 + growth)).values
            prev_year_values = dataframe.iloc[-validation_steps:][target_column_name].values

        elif offset < 0:
            # If yearly steps > validation steps will need to offsets actuals
            forecast_values = forecast_results.iloc[-validation_steps:][target_column_name + '_Forecast'].values
            yoy_seasonal_forecast = (dataframe.iloc[-validation_steps-abs(offset):-abs(offset)][target_column_name] * (1 + growth)).values
            prev_year_values = dataframe.iloc[-validation_steps-abs(offset):-abs(offset)][target_column_name].values

        # Write Plots for YoY Growth vs Forecast
        figure, axis = plt.subplots(figsize=(16,8))
        plt.plot(forecast_values, label='Validation Forecast')
        plt.plot(yoy_seasonal_forecast, label='Prev Year + YoY Growth')
        plt.plot(prev_year_values, label='Prev Year')
        plt.legend()
        plt.savefig(f"../run_results/{iteration_name}/validation_plots_YoY_growth/{h}.png")

        # MAPE Growth Seasonal Data vs Forecast
        mape_seasonal_yoy = mape(yoy_seasonal_forecast, forecast_values)

        # MAPE CV Growth Seasonal Data vs Forecast
        seasonal_cv = np.std(yoy_seasonal_forecast) / np.mean(yoy_seasonal_forecast)
        forecast_cv = np.std(forecast_values) / np.mean(forecast_values)
        mape_cv = np.abs((forecast_cv - seasonal_cv)/seasonal_cv)
        
        # Selection MAPE
        selection_mape = (mape_seasonal_yoy +  mape_cv + test_mape) / 3

        # Sending To List
        cv_scores.append([h, selection_mape, test_mape, test_period_forecast])
        
    cv_scores = pd.DataFrame(cv_scores, columns=['horizon','selection_mape','mape','test_period_forecast'])

    best_score = cv_scores[cv_scores['selection_mape'] == cv_scores['selection_mape'].min()]['mape'].values[0]
    best_horizon = cv_scores[cv_scores['selection_mape'] == cv_scores['selection_mape'].min()]['horizon'].values[0]#
    best_test_period_forecast = cv_scores[cv_scores['selection_mape'] == cv_scores['selection_mape'].min()]['test_period_forecast'].values[0]

    return best_horizon, best_score, best_test_period_forecast

def produce_forecast(best_horizon, dataframe):

    # Steps
    forecast_steps = run_settings['FORECAST_STEPS']

    # Names

    date_column_name = run_settings['DATE_COLUMN_NAME']
    target_column_name = run_settings['TARGET_COLUMN_NAME']
    exog_column_names = run_settings['EXOG_COLUMN_NAMES']

    """ Use the best horizon metric to create a forecast"""
    
    cols = [date_column_name] + exog_column_names + [target_column_name]
    
    # Initializing Model
    model = build_model_object()

    # Fitting Model
    model.train(dataframe[cols], date_column_name, target_column_name, best_horizon)
    forecast_results = model.forecast(dataframe[cols], forecast_steps)
    forecast_values = forecast_results[target_column_name + '_Forecast'].iloc[-forecast_steps:].values

    return forecast_values