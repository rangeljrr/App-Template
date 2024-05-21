import numpy as np
#from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.model_selection import cross_val_score

def mape(y,yhat):
    return np.mean(np.abs(yhat-y) / y) 


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
