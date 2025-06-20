import pandas as pd
import numpy as np
import plotly.express as px
import optuna

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, KFold
)
from sklearn.base import clone


def create_feature_importance_plot(model, feature_names, output_path):
    if not hasattr(model, 'feature_importances_'):
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


def evaluate_classification(y_true, y_pred=None, y_proba=None, threshold=0.5, return_format="dataframe", verbose=False):
    if y_proba is not None and y_pred is None:
        y_pred = (y_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    total = len(y_true)

    results = {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "specificity": round(specificity, 4) if specificity is not None else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total": int(total)
    }

    if verbose:
        print(f"Eval: Acc={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    return pd.DataFrame([results]) if return_format == "dataframe" else results


def cross_validate_model(X, y, model, n_splits, random_state):
    X = np.array(X)
    y = np.array(y)
    k_fold_scores = []

    if n_splits == 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=random_state
        )
        model_fit = clone(model)
        model_fit.fit(X_train, y_train)
        y_pred = model_fit.predict(X_val)
        y_proba = model_fit.predict_proba(X_val)[:, 1] if hasattr(model_fit, "predict_proba") else None

        fold_score = evaluate_classification(y_val, y_pred, y_proba, return_format="dataframe", verbose=False)
        fold_score["fold"] = 1
        k_fold_scores.append(fold_score)

    elif n_splits >= 2:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model_fold = clone(model)
            model_fold.fit(X_train, y_train)
            y_pred = model_fold.predict(X_val)
            y_proba = model_fold.predict_proba(X_val)[:, 1] if hasattr(model_fold, "predict_proba") else None

            fold_score = evaluate_classification(y_val, y_pred, y_proba, return_format="dataframe", verbose=False)
            fold_score["fold"] = fold + 1
            k_fold_scores.append(fold_score)

    else:
        raise ValueError("n_splits must be 0 or >= 2")

    k_fold_scores = pd.concat(k_fold_scores, ignore_index=True)
    agg_row = k_fold_scores.drop(columns=["fold"]).mean()
    agg_row["fold"] = "final"
    return pd.concat([k_fold_scores, pd.DataFrame([agg_row])], ignore_index=True)


def run_optuna_trial_classification(
    X, y, model_object, param_space,
    optuna_score, n_trials, n_splits,
    cv_score, random_state
):
    def objective(trial):
        params = {key: fn(trial) for key, fn in param_space.items()}
        params['random_state'] = random_state

        from lightgbm import LGBMClassifier
        if model_object is LGBMClassifier:
            params['verbosity'] = -1

        model = model_object(**params)

        if n_splits <= 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=True, stratify=y, random_state=random_state
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

            if optuna_score == 'accuracy':
                return accuracy_score(y_val, y_pred)
            elif optuna_score == 'f1':
                return f1_score(y_val, y_pred)
            elif optuna_score == 'precision':
                return precision_score(y_val, y_pred)
            elif optuna_score == 'recall':
                return recall_score(y_val, y_pred)
            elif optuna_score == 'roc_auc' and y_proba is not None:
                return roc_auc_score(y_val, y_proba)
            else:
                raise ValueError(f"Unsupported optuna_score or missing predict_proba: {optuna_score}")
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring=cv_score, n_jobs=-1)
            return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params
