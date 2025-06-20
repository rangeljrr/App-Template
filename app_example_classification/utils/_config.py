run_parameters = {
    
    # Run Params
    'development_iteration':'1.0',
    'root':'C:\\Users\\3283078\\OneDrive - Meijer365 (US)\\Documents\\app_example_classification\\',
    'feature_set':'test+train+dummy',

    # Used for Optuna
    'n_trials':0,
    'optuna_score':'precision',
    
    # USed for Cross-Validation
    'n_splits':0, # Used for CV K-Fold
    'cv_score':'precision',

    # Replication
    'random_state':42,
}