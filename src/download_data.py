from d01_data.load_data import download_data

# Variables
DATASETS = ["IP", "SA", "PU"]
DATA_PATH = "./data/01_raw/"

# Download each dataset
for dataset in DATASETS:
    download_data(data_type=dataset, save_folder=DATA_PATH)
