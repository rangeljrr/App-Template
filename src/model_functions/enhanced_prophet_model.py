import pandas as pd
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from data_functions.process_data import create_datetime_features
from validation_functions.scoring_functions import mape
from training_functions._train_config import forecast_steps, validation_steps

def rename_to_ds_y(dataframe):
    # Need to make a couple changes for prophet to run
    return dataframe.rename(columns={'week_start_date':'ds','page_views':'y'})
    

def build_param_grid():

    param_settings = {
       'seasonality_mode': ['additive', 'multiplicative'],
       'changepoint_prior_scale': [0.01, 0.1, 0.5],
       'seasonality_prior_scale': [0.01, 0.1, 1.0],
       'holidays_prior_scale': [0.01, 0.1, 1.0],
       'changepoint_range': [0.8, 0.9, 0.95]
    }

    param_grid = ParameterGrid(param_settings)
        
    return param_grid
    
def build_model(params):
    # Initializing Model
    prophet_model = Prophet(**params)
    
    # Adding additional regressors
    # prophet_model.add_country_holidays(country_name='US')
    # prophet_model.add_regressor('month')
    # prophet_model.add_regressor('dayofweek')
    # prophet_model.add_regressor('quarter')
    # prophet_model.add_regressor('dayofyear')
    # prophet_model.add_regressor('dayofmonth')
    # prophet_model.add_regressor('weekofyear')

    return prophet_model

def train_model(train, test):

    """ Input -> train(pd.DataFrame), test (pd.DataFrame)
        Output -> [best_params(dict), best_score(flot)]"""
    
    # Need to make a couple changes for prophet to run
    train = rename_to_ds_y(train)
    test = rename_to_ds_y(test)

    param_grid = build_param_grid()

    cv_scores = []

    for params in param_grid:
        
        # Initializing Model
        prophet_model = build_model(params)
    
    
        # Fitting Model
        prophet_model.fit(train)
    
        forecast_results = prophet_model.predict(test.drop(['y'], axis=1))
        iter_score = mape(test['y'].values, forecast_results['yhat'].values)
        iter_test_period_forecast = forecast_results['yhat'].values
        
        cv_scores.append([params, iter_score, iter_test_period_forecast])
        
    cv_scores = pd.DataFrame(cv_scores, columns=['parameters','mape','test_period_forecast'])

    best_score = cv_scores[cv_scores['mape'] == cv_scores['mape'].min()]['mape'].values[0]
    best_params = cv_scores[cv_scores['mape'] == cv_scores['mape'].min()]['parameters'].values[0]
    best_test_period_forecast = cv_scores[cv_scores['mape'] == cv_scores['mape'].min()]['test_period_forecast'].values[0]
    
    return best_params, best_score, best_test_period_forecast


def produce_forecast(parameters, dataframe):
    """ Input-> parameters (dict), dataframe (pd.DataFrame)
        Output -> forecast[['forecast_date','forecasted_page_views']] (pd.DataFrame)"""
    
    # Need to make a couple changes for prophet to run
    dataframe = rename_to_ds_y(dataframe)
    prophet_model = build_model(parameters).fit(dataframe)

    forecast_dataframe = prophet_model.make_future_dataframe(periods=forecast_steps, freq='W-Mon')
    forecast_dataframe =  create_datetime_features(forecast_dataframe,'ds')
    forecast = prophet_model.predict(forecast_dataframe)[['ds','yhat']]
    
    return forecast.iloc[-forecast_steps:]['yhat'].values


def main(dataframe):

    train,test = [dataframe[0], dataframe[1]]

    best_params, best_score, hist_forecast = train_model(train, test)

    forecast = produce_forecast(best_params, dataframe)

    # Need to save forecast, scores, test_forecast