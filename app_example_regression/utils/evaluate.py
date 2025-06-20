# Cleaned imports
import numpy as np
import pandas as pd
import plotly.express as px
import optuna

from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    explained_variance_score, max_error
)

def create_feature_importance_plot(model, feature_names, output_path):
    """
    Saves a feature importance plot using Plotly Express if the model supports it.
    
    Parameters:
    - model: A fitted model. Must have `feature_importances_` to plot.
    - feature_names: List or pandas Index of feature names.
    - output_path: File path to save the plot (supports .html, .png, etc.).
    """
    if not hasattr(model, 'feature_importances_'):
        print("Warning: Model has no 'feature_importances_' attribute. Skipping plot.")
        return

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance',
        width=1000,
        height=800
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})

    if output_path.endswith('.html'):
        fig.write_html(output_path)
    else:
        fig.write_image(output_path)


def evaluate_regression(y_true, y_pred, return_format="dataframe", verbose=False):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    max_err = max_error(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-10, None))) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

    results = {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "explained_variance": round(evs, 4),
        "max_error": round(max_err, 4),
        "mape": round(mape, 4),
        "smape": round(smape, 4),
        "total": len(y_true)
    }

    if verbose:
        print(f"Eval: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%, SMAPE={smape:.2f}%")

    if return_format == "dataframe":
        return pd.DataFrame([results])
    elif return_format == "dict":
        return results
    else:
        raise ValueError("return_format must be either 'dataframe' or 'dict'")


def cross_validate_model(X, y, model, n_splits, random_state):
    X = np.array(X)
    y = np.array(y)
    k_fold_scores = []

    if n_splits == 0:
        # Simple train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=random_state
        )

        model_fit = clone(model)
        model_fit.fit(X_train, y_train)
        y_pred = model_fit.predict(X_val)

        fold_score = evaluate_regression(
            y_val, y_pred, return_format="dataframe", verbose=False
        )
        fold_score["fold"] = 1
        k_fold_scores.append(fold_score)

    elif n_splits >= 2:
        # K-Fold CV logic (only now we create the KFold object)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model_fold = clone(model)
            model_fold.fit(X_train, y_train)
            y_pred = model_fold.predict(X_val)

            fold_score = evaluate_regression(
                y_val, y_pred, return_format="dataframe", verbose=False
            )
            fold_score["fold"] = fold + 1
            k_fold_scores.append(fold_score)

    else:
        raise ValueError("n_splits must be 0 (for train/test split) or >= 2 (for K-Fold CV)")

    # Aggregate results
    k_fold_scores = pd.concat(k_fold_scores, ignore_index=True)
    agg_row = k_fold_scores.drop(columns=["fold"]).mean()
    agg_row["fold"] = "final"
    k_fold_scores = pd.concat([k_fold_scores, pd.DataFrame([agg_row])], ignore_index=True)

    return k_fold_scores


def run_optuna_trial(X,y,model_object,param_space,optuna_score,n_trials,n_splits,cv_score,random_state):
    
    def objective(trial):
        # Sample hyperparameters from the provided space
        params = {key: fn(trial) for key, fn in param_space.items()}
        params['random_state'] = random_state  # enforce reproducibility

        # Suppress LightGBM warnings
        if model_object is LGBMRegressor:
            params['verbosity'] = -1
            
        model = model_object(**params)

        if n_splits <= 1:
            # Fallback to train/test split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=True, random_state=random_state
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        from sklearn.metrics import (
            r2_score, mean_absolute_error, mean_squared_error,
            explained_variance_score, max_error
        )
        import numpy as np
        
        if optuna_score == 'r2':
            return r2_score(y_val, y_pred)
        elif optuna_score == 'neg_mean_absolute_error':
            return -mean_absolute_error(y_val, y_pred)
        elif optuna_score == 'neg_mean_squared_error':
            return -mean_squared_error(y_val, y_pred)
        elif optuna_score == 'neg_root_mean_squared_error':
            return -np.sqrt(mean_squared_error(y_val, y_pred))
        elif optuna_score == 'explained_variance':
            return explained_variance_score(y_val, y_pred)
        elif optuna_score == 'neg_max_error':
            return -max_error(y_val, y_pred)
        else:
            raise ValueError(f"Unsupported optuna_score: {optuna_score}")
            
        if n_splits >= 2:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring=cv_score, n_jobs=-1)
            return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = 'maximize' if 'r2' in optuna_score else 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params