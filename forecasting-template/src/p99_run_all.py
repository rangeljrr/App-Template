from p01_run_forecast import main as p01_run_forecast
from p02_champion_challenger import main as p02_champion_challenger
from p03_post_processing import main as p03_post_processing

def main():

    # Running process 1
    p01_run_forecast()
    
    # Running process 2
    p02_champion_challenger()
    
    # Running process 3
    p03_post_processing


if __name__ == "__main__":
    main()