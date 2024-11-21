import pandas as pd
from config import run_settings

def main():
    
    forecast = pd.read_csv(f'../run_results/latest/forecast/results.csv')
    scores = pd.read_csv(f'../run_results/latest/scores/results.csv')
    iteration_name = run_settings['run_name']
    
    
    # Finding Best Score
    scores["winner_model"] = scores.iloc[:,2:].idxmin(axis=1)
    scores["winner_score"] =  scores.iloc[:,2:-1].min(axis=1)
    
    # Adding Winenr to forecast
    forecast = forecast.merge(scores[['label', 'winner_model']], how='left')
    forecast["winner_forecast"] = forecast.apply(lambda row: row[row["winner_model"]] if pd.notnull(row["winner_model"]) else None, axis=1)
    
    forecast.to_csv(f'../run_results/{iteration_name}/forecast/results.csv', index=False)
    forecast.to_csv(f'../run_results/latest/forecast/results.csv', index=False)
    
    scores.to_csv(f'../run_results/{iteration_name}/scores/results.csv', index=False)
    scores.to_csv(f'../run_results/latest/scores/results.csv', index=False)

if __name__ == "__main__":
    main()