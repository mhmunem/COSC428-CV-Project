from pathlib import Path

from loguru import logger
from tqdm import tqdm


from hsi_classifier.load_data import load_raw_data, create_traning_data, save_interim_data
# from hsi_classifier.dataset import load_data
# Global configuration
DATA_TYPE = "IP"
DEFAULT_TEST_RATIO = 0.80
DEFAULT_WINDOW_SIZE = 11

def main():
    data, label = load_raw_data('IP')
    X_train, y_train, X_test, y_test = create_traning_data(data, label)
    save_interim_data(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
