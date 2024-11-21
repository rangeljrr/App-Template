import os
def check_or_create_dir(folder_path):

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        os.makedirs(folder_path + 'forecast/')
        os.makedirs(folder_path + 'scores/')
        os.makedirs(folder_path + 'params/')
        os.makedirs(folder_path + 'test_period_forecast/')
        os.makedirs(folder_path + 'validation_plots/')
        os.makedirs(folder_path + 'validation_plots_YoY_growth/')

        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")