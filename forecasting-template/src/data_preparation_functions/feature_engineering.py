from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

def create_features(dataframe):
    return dataframe

def create_forecast_dates(date_series,steps):
    """ Create Future Forecast Date Range """
    forecast_start_date = date_series.max() + relativedelta(months=1)
    forecast_dates = [forecast_start_date + relativedelta(months=i) for i in range(steps)]  

    return forecast_dates