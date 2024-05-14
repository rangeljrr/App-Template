import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import ParameterGrid
from validation_functions.scoring_functions import mape
from training_functions._train_config import forecast_steps

def build_param_grid():
    param_settings = {
        'order': [(1, 0, 0), (0, 1, 1), (1, 1, 1)],
        'seasonal_order': [((1, 0, 0, 12)), ((0, 1, 1, 12)), ((1, 1, 1, 12))],
    }

    param_grid = ParameterGrid(param_settings)
    return param_grid

def build_model(params):
    # Initializing Model
    sarimax_model = SARIMAX(**params)
    return sarimax_model

def train_model(train, test):
    param_grid = build_param_grid()
    cv_scores = []

    for params in param_grid:
        # Initializing Model
        sarimax_model = build_model(params)
    
        # Fitting Model
        sarimax_result = sarimax_model.fit(disp=False)
    
        # Forecasting
        forecast_results = sarimax_result.get_forecast(steps=len(test))
        iter_score = mape(test, forecast_results.predicted_mean)
        iter_test_period_forecast = forecast_results.predicted_mean.values
        
        cv_scores.append([params, iter_score, iter_test_period_forecast])
        
    cv_scores = pd.DataFrame(cv_scores, columns=['parameters','mape','test_period_forecast'])

    best_score = cv_scores[cv_scores['mape'] == cv_scores['mape'].min()]['mape'].values[0]
    best_params = cv_scores[cv_scores['mape'] == cv_scores['mape'].min()]['parameters'].values[0]
    best_test_period_forecast = cv_scores[cv_scores['mape'] == cv_scores['mape'].min()]['test_period_forecast'].values[0]
    
    return best_params, best_score, best_test_period_forecast

def produce_forecast(parameters, train, test):
    """ Input-> parameters (dict), train (pd.Series), test (pd.Series)
        Output -> forecast (np.array)"""
    
    # Fitting the final model
    sarimax_model = build_model(parameters)
    sarimax_result = sarimax_model.fit(disp=False)
    
    # Forecasting future steps
    final_train_test = pd.concat([train, test])
    forecast_results = sarimax_result.get_forecast(steps=forecast_steps)
    
    return forecast_results.predicted_mean.values

def main(dataframe):
    train, test = dataframe[0], dataframe[1]

    best_params, best_score, hist_forecast = train_model(train, test)
    forecast = produce_forecast(best_params, train, test)

    # Need to save forecast, scores, test_forecast
