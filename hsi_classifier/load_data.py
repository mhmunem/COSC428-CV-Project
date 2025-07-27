import os
from pathlib import Path

from loguru import logger
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import typer

from hsi_classifier.config import INTERIM_DATA_DIR, LOG_DIR, RAW_DATA_DIR

app = typer.Typer()

DEFAULT_DATASET = "IP"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
RANDOM_STATE = 345


class HSIDataLoader:
    def __init__(
        self,
        data_path: Path,
        label_path: Path,
        save_path: Path,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        self.data_path = data_path
        self.label_path = label_path
        self.save_path = save_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.X = None
        self.y = None

    def load(self):
        logger.info(f"Loading data from {self.data_path}")
        data = loadmat(self.data_path)
        first_key = next(key for key in data.keys() if not key.startswith("__"))
        image = data[first_key]

        logger.info(f"Loading labels from {self.label_path}")
        labels = loadmat(self.label_path)
        first_key = next(key for key in labels.keys() if not key.startswith("__"))
        label_map = labels[first_key]

        self.X = image
        self.y = label_map

        logger.success(f"Loaded {self.X.shape[0]} labeled pixels with {self.X.shape[1]} bands")

    def return_data(self):
        return self.X

    def return_labels(self):
        return self.y

    def save(self, x, file_name: str):
        os.makedirs(self.save_path, exist_ok=True)
        save_file = self.save_path / f"{file_name}.npy"
        np.save(save_file, x)
        logger.info(f"Saved {file_name}.npy to {self.save_path}")

    def split(self):
        logger.info("Splitting data into train, validation, and test sets")
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
            # stratify=self.y
        )

        val_ratio = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_ratio),
            random_state=self.random_state,
            # stratify=y_temp
        )

        # Save the splits
        self.save(X_train, "X_train")
        self.save(y_train, "y_train")
        self.save(X_val, "X_val")
        self.save(y_val, "y_val")
        self.save(X_test, "X_test")
        self.save(y_test, "y_test")

        logger.success(
            f"Split complete: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test


@app.command()
def main(data_name: str = DEFAULT_DATASET):
    data_path = RAW_DATA_DIR / data_name / "data.mat"
    label_path = RAW_DATA_DIR / data_name / "labels.mat"
    save_path = INTERIM_DATA_DIR / data_name

    logger.add(f"{LOG_DIR}/pipeline.log", rotation="1 MB")
    logger.info(f"Loading dataset: {data_name}")

    loader = HSIDataLoader(data_path, label_path, save_path)
    loader.load()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.split()

    logger.info(
        f"Train shape: {X_train.shape}, \n Validation shape: {X_val.shape}, \nTest shape: {X_test.shape}"
    )
    logger.info(
        f"Train label shape: {y_train.shape}, \n Validation label shape: {y_val.shape}, \n Test label shape: {y_test.shape}"
    )
    logger.success(f"Load complete: {data_name}")


if __name__ == "__main__":
    app()
