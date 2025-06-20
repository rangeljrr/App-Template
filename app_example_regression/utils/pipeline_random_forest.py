import optuna
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score

from utils.evaluate import cross_validate_model
from utils._config import run_parameters
from utils.IO import save_to_json
#from utils.tuning import run_optuna_trial  # <- Ensure this exists
from utils.evaluate import run_optuna_trial

# Configuration
random_state = run_parameters['random_state']
n_splits = run_parameters['n_splits']
optuna_score = run_parameters['optuna_score']
cv_score = run_parameters['cv_score']
n_trials = run_parameters['n_trials']

def create_param_space():
    return {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 500),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': lambda trial: trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': lambda trial: trial.suggest_categorical('bootstrap', [True, False])
    }


def find_best_params(X, y, n_trials):
    
    param_space = create_param_space()
    model_object = RandomForestRegressor  # <- Pass class, not instance
    best_params = run_optuna_trial(X, y, model_object, param_space, optuna_score, n_trials, n_splits, cv_score, random_state)

    return best_params


def validate_model(X, y, params):
    
    model = RandomForestRegressor(**params)
    return cross_validate_model(X, y, model, n_splits, random_state)


def fit_best_model(X, y, params):
    
    return RandomForestRegressor(**params).fit(X, y)


def main(X, y):
    
    if n_trials > 0:
        base_model_validation = validate_model(X, y, {})
        best_params = find_best_params(X, y, n_trials)
        best_model_validation = validate_model(X, y, best_params)

        if base_model_validation.iloc[-1][cv_score] >= best_model_validation.iloc[-1][cv_score]:
            best_params = {}

        fitted_model = fit_best_model(X, y, best_params)
        
        return [best_model_validation, best_params, fitted_model]
        
    else:
        best_params = {}
        base_model_validation = validate_model(X, y, best_params)
        fitted_model = fit_best_model(X, y, best_params)
        
        return [base_model_validation, best_params, fitted_model]
