import numpy as np

def mse(actual, predicted):
    """
    Mean Squared Error (MSE).
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    errors = actual - predicted
    mse_value = np.mean(errors ** 2)

    return mse_value

def rmse(actual, predicted):
    """
    Root Mean Squared Error (RMSE).
    """
    mse_value = mse(actual, predicted)
    rmse_value = np.sqrt(mse_value)

    return rmse_value

def ae(actual, predicted):
    """
    Absolute Error (AE).
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    absolute_errors = np.abs(actual - predicted)
    ae_value = np.mean(absolute_errors)

    return ae_value

def mape(actual, predicted):
    """
    Mean Absolute Percentage Error (MAPE).
    Handles division by zero by ignoring zero actual values.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Mask to avoid division by zero
    mask = actual != 0
    actual_nonzero = actual[mask]
    predicted_nonzero = predicted[mask]
    
    percentage_errors = np.abs((actual_nonzero - predicted_nonzero) / actual_nonzero)
    mape_value = np.mean(percentage_errors)

    return mape_value

def smape(actual, predicted):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    smape_value = np.mean(numerator / denominator)

    return smape_value

def r_squared(actual, predicted):
    """
    R-squared (Coefficient of Determination).
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    ss_total = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
    ss_residual = np.sum((actual - predicted) ** 2)    # Residual sum of squares
    r2 = 1 - (ss_residual / ss_total)

    return r2

def adjusted_r_squared(actual, predicted, n, p):
    """
    Adjusted R-squared.
    n: Number of observations
    p: Number of predictors (independent variables)
    """
    r2 = r_squared(actual, predicted)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    return adj_r2
    
def run_all_metrics(actual, predicted, n, p):

    score_dict = {
        'mape': mape(actual, predicted), 
        # 'smape': smape(actual, predicted),
        # 'rmse': rmse(actual, predicted),
        # 'ae': ae(actual, predicted),
    }
    
    return score_dict


# def reg_evaluate(model, X, y, k=5):
#     """ Evaluate a regression model using Cross Validation (Default k=5) """
    
#     # Cross Validation metrics
#     R2 = round(np.mean(np.abs(cross_val_score(model, X, y, cv=5, scoring='r2'))),3)
#     RMSE = round(np.mean(np.abs(cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error'))),3)
#     MAPE = round(np.mean(np.abs(cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_percentage_error'))),3)
#     ACC = 1 - MAPE
#     MAE = round(np.mean(np.abs((cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')))),2)

#     return [ACC, R2, RMSE, MAPE , MAE]


# def cls_evaluate(model, X, y, k=5):
#     """ Evaluate a classification model using Cross Validation (Default k=5) """
    
#     # Cross Validation metrics
#     #R2 = round(np.mean(np.abs(cross_val_score(model, X, y, cv=5, scoring='r2'))),3)
#     #RMSE = round(np.mean(np.abs(cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error'))),3)
#     #MAPE = round(np.mean(np.abs(cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_percentage_error'))),3)
#     #ACC = 1 - MAPE
#     #MAE = round(np.mean(np.abs((cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')))),2)

#     #return [ACC, R2, RMSE, MAPE , MAE]
