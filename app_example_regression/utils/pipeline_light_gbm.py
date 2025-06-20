import optuna
import numpy as np
import pandas as pd
import os
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score

from utils.evaluate import cross_validate_model, run_optuna_trial
from utils._config import run_parameters
from utils.IO import save_to_json

# Configuration
random_state = run_parameters['random_state']
n_splits = run_parameters['n_splits']
optuna_score = run_parameters['optuna_score']
cv_score = run_parameters['cv_score']
n_trials = run_parameters['n_trials']


def create_param_space():
    
    return {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 100, 500),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 20),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': lambda trial: trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': lambda trial: trial.suggest_int('min_child_samples', 5, 30),
        'subsample': lambda trial: trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': lambda trial: trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': lambda trial: trial.suggest_float('reg_lambda', 0.0, 1.0)
    }


def find_best_params(X, y, n_trials):
    
    param_space = create_param_space()
    model_object = LGBMRegressor
    best_params = run_optuna_trial(X, y, model_object, param_space, optuna_score, n_trials, n_splits, cv_score, random_state)

    return best_params


def validate_model(X, y, params):
    
    model = LGBMRegressor(**params, random_state=random_state, n_jobs=-1, verbosity=-1)
    return cross_validate_model(X, y, model, n_splits, random_state)


def fit_best_model(X, y, params):
    
    return LGBMRegressor(**params, random_state=random_state, n_jobs=-1, verbosity=-1).fit(X, y)


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
