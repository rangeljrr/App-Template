
import pandas as pd
import matplotlib.pyplot as plt
from config import run_settings
from pandas.tseries.offsets import MonthEnd

def create_validation_plots(historical_data, forecast_data):

    date_column_name = run_settings['DATE_COLUMN_NAME']
    target_column_name = run_settings['TARGET_COLUMN_NAME']
    iteration_name = run_settings['ITERATION_NAME']

    forecast_data[date_column_name] = pd.to_datetime(forecast_data[date_column_name])
    forecast_data[date_column_name] = pd.to_datetime(forecast_data[date_column_name])
    historical_data[date_column_name] = pd.to_datetime(historical_data[date_column_name]) + MonthEnd(0)

    forecast_data['type']= 'forecast'
    historical_data['type']= 'historical'
    plot_data = pd.concat([historical_data,forecast_data])


    figure, axis = plt.subplots(2,1, figsize=(16,8))
    hist = plot_data[plot_data['type'] == 'historical']
    fcst = plot_data[plot_data['type'] == 'forecast']


    prev = plot_data[(plot_data['type'] == 'historical')&
                            (plot_data[date_column_name] >= pd.to_datetime('1960-01-01'))&
                            (plot_data[date_column_name] <= pd.to_datetime('1960-12-31'))].reset_index(drop=True)

    prev1 = plot_data[(plot_data['type'] == 'historical')&
                            (plot_data[date_column_name] >= pd.to_datetime('1959-01-01'))&
                            (plot_data[date_column_name] <= pd.to_datetime('1959-12-31'))].reset_index(drop=True)

    prev2 = plot_data[(plot_data['type'] == 'historical')&
                            (plot_data[date_column_name] >= pd.to_datetime('1958-01-01'))&
                            (plot_data[date_column_name] <= pd.to_datetime('1958-12-31'))].reset_index(drop=True)

    prev3 = plot_data[(plot_data['type'] == 'historical')&
                            (plot_data[date_column_name] >= pd.to_datetime('1957-01-01'))&
                            (plot_data[date_column_name] <= pd.to_datetime('1957-12-31'))].reset_index(drop=True)

    fcst_2024 = plot_data[plot_data['type'] == 'forecast']

    figure, axis = plt.subplots(2,1, figsize=(16,8))

    axis[0].plot(hist[date_column_name], hist[target_column_name], label = 'historical', linewidth=1)
    axis[0].plot(fcst[date_column_name], fcst[target_column_name], label = 'forecast', linewidth=1)
    axis[0].legend()


    axis[1].plot(prev[date_column_name].dt.month, prev[target_column_name], label = '1960', linewidth=1, linestyle=':')
    axis[1].plot(prev1[date_column_name].dt.month, prev1[target_column_name], label = '1959', linewidth=1, color='C0', linestyle='--')
    axis[1].plot(prev2[date_column_name].dt.month, prev2[target_column_name], label = '1958', linewidth=1, color='C0')
    axis[1].plot(fcst_2024[date_column_name].dt.month, fcst_2024[target_column_name], label = '1961 fcst', linewidth=1, color='darkorange')

    #axis[1].scatter(hist_2024['week_start_date'].dt.isocalendar()['week'], hist_2024['page_views'], s=5, color='C0')
    ##axis[1].scatter(fcst_2024['week_start_date'].dt.isocalendar()['week'], fcst_2024['page_views'], s=5, color='darkorange')

    axis[1].legend()

    axis[0].set_title(f"Passengers:")
    plt.savefig(f"../run_results/{iteration_name}/validation_plots/plot.png")