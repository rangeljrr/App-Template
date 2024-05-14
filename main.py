from model_functions.enhanced_prophet_model import main as enhanced_prophet_main
from model_functions.prophet_model import main as prophet_main
from model_functions.sarimax_model import main as sarimax_main

selector_switch = {
    'prophet':True,
    'prophet_enhanced':False,
    'sarimax':True
}

# Read Data
data = []

# Run Model 1
if 'prophet' == True:
    check_directories('prophet')
    prophet_main(data)


# # Run Model 2
# if 'prophet_enhanced' == True:
#     check_directories('prophet_enhanced')
#     enhanced_prophet_main(data)


# # Run Model 3
# if 'sarimax' == True:
#     check_directories('prophet_enhanced')
#     sarimax_main(data)


import os

def create_directories_os(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Example usage
directories_to_create = ['dir1', 'dir2', 'dir3']
create_directories_os(directories_to_create)