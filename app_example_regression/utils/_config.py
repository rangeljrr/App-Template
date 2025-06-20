run_parameters = {
    
    # Run Params
    'development_iteration':'1.0',
    'root':'C:\\Users\\3283078\\OneDrive - Meijer365 (US)\\Documents\\app_example_regression\\',
    'feature_set':'test+train+regression+dummy',

    # Used for Optuna
    'n_trials':0,
    'optuna_score':'r2',
    
    # USed for Cross-Validation
    'n_splits':0, # Used for CV K-Fold
    'cv_score':'r2',

    # Replication
    'random_state':42,
}
