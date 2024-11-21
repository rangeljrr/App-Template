"""
Author: Rodrigo
"""

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sys

sys.path.insert(1, '../')
from validation_functions.scoring_functions import mape, smape

def train_model(train, test, validation_metric):

    if validation_metric == 'mape':
        scorer = mape

    elif validation_metric == 'smape':
        scorer = smape

    auto_arima_model = auto_arima(
        y=train,
        seasonal=True,
        m=12,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    best_params = [auto_arima_model.order, auto_arima_model.seasonal_order]
    
    # Fit SARIMAX model with the best parameters
    sarimax_model = SARIMAX(
        train,
        order=best_params[0],
        seasonal_order=best_params[1],
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    
    best_test_period_forecast = sarimax_model.get_forecast(steps=len(test)).predicted_mean
    
    best_score = mape(test, best_test_period_forecast)

    return best_params, best_score, best_test_period_forecast.values

def produce_forecast(series, parameters, forecast_steps):

    # Fit SARIMAX model with the best parameters
    sarimax_model = SARIMAX(
        series,
        order=parameters[0],
        seasonal_order=parameters[1],
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
 
    forecast = sarimax_model.get_forecast(steps=forecast_steps).predicted_mean

    return forecast.values
