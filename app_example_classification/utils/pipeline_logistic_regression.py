import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

from utils.evaluate import cross_validate_model
from utils._config import run_parameters
from utils.IO import save_to_json

# Configuration
random_state = run_parameters['random_state']
n_splits = run_parameters['n_splits']
cv_score = run_parameters['cv_score']
n_trials = run_parameters['n_trials']


def create_param_space():
    # If you want to later use Optuna for tuning, here’s a param space example
    return {
        # 'C': lambda trial: trial.suggest_float('C', 0.01, 10.0, log=True),
        # 'penalty': lambda trial: trial.suggest_categorical('penalty', ['l2']),
        # 'solver': lambda trial: trial.suggest_categorical('solver', ['lbfgs']),
    }


def find_best_params(X, y, n_trials):
    # No tuning by default
    return {}


def validate_model(X, y, params):
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        n_jobs=-1,
        **params
    )
    return cross_validate_model(X, y, model, n_splits, random_state)


def fit_best_model(X, y, params):
    return LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        n_jobs=-1,
        **params
    ).fit(X, y)


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
