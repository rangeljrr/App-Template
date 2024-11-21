"""
Author: Rodrigo
Checks:
    1. Make sure you have the correct freq in produce_forecast, for example M vs MS
    This will cause shift issues. Also if you have M and your data is weekly, this 
    will cause forecast issues. 
    
"""

import pandas as pd
from prophet import Prophet

from sklearn.model_selection import ParameterGrid
from validation_functions.scoring_functions import mape, smape
from config import run_settings

def rename_to_ds_y(dataframe):
    
    # Need to make a couple changes for prophet to run
    return dataframe.rename(columns={'Month':'ds','Passengers':'y'})
    

def build_param_grid():

    param_settings = {
    }
    
    param_grid = ParameterGrid(param_settings)
    
    return param_grid


def build_model(params):
    
    # Initializing Model
    prophet_model = Prophet(**params)

    return prophet_model
    

def train_model(train, test, validation_metric):

    """ 
    Input -> train(pd.DataFrame), test (pd.DataFrame)
        Output -> [best_params(dict), best_score(flot)]
    """
    
    # Need to make a couple changes for prophet to run
    train = rename_to_ds_y(train)
    test = rename_to_ds_y(test)

    param_grid = build_param_grid()

    cv_scores = []

    if validation_metric == 'mape':
        scorer = mape

    elif validation_metric == 'smape':
        scorer = smape
        
    for params in param_grid:
        
        # Initializing Model
        prophet_model = build_model(params)
    
        # Fitting Model
        prophet_model.fit(train)
    
        forecast_results = prophet_model.predict(test.drop(['y'], axis=1))
        iter_score = scorer(test['y'].values, forecast_results['yhat'].values)
        iter_test_period_forecast = forecast_results['yhat'].values
        
        cv_scores.append([params, iter_score, iter_test_period_forecast])
        
    cv_scores = pd.DataFrame(cv_scores, columns=['parameters','error','test_period_forecast'])

    best_score = cv_scores[cv_scores['error'] == cv_scores['error'].min()]['error'].values[0]
    best_params = cv_scores[cv_scores['error'] == cv_scores['error'].min()]['parameters'].values[0]
    best_test_period_forecast = cv_scores[cv_scores['error'] == cv_scores['error'].min()]['test_period_forecast'].values[0]
    
    return best_params, best_score, best_test_period_forecast


def produce_forecast(dataframe, parameters, forecast_steps):
    """ Input-> parameters (dict), dataframe (pd.DataFrame)
        Output -> forecast[['forecast_date','forecasted_page_views']] (pd.DataFrame)"""
    
    # Need to make a couple changes for prophet to run
    dataframe = rename_to_ds_y(dataframe)
    prophet_model = build_model(parameters).fit(dataframe)

    forecast_dataframe = prophet_model.make_future_dataframe(periods=forecast_steps, freq='M')
    forecast = prophet_model.predict(forecast_dataframe)[['ds','yhat']]
    
    return forecast.iloc[-forecast_steps:]['yhat'].values
