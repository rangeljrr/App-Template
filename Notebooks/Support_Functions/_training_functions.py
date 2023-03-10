import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


def train_model(dataframe_X, series_y):
    """ This function will train an ML model and return the object"""

    model = LinearRegression()
    model.fit(dataframe_X, series_y)
    
    return model

def reg_evaluate(model, X, y, k=5):
    """ Evaluate a regression model using Cross Validation (Default k=5) """
    
    # Cross Validation metrics
    R2 = round(np.mean(np.abs(cross_val_score(model, X, y, cv=5, scoring='r2'))),3)
    RMSE = round(np.mean(np.abs(cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error'))),3)
    MAPE = round(np.mean(np.abs(cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_percentage_error'))),3)
    ACC = 1 - MAPE
    MAE = round(np.mean(np.abs((cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')))),2)

    return [ACC, R2, RMSE, MAPE , MAE]


