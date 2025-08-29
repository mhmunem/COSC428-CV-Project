import os
from pathlib import Path
import time
from typing import List

from loguru import logger
import numpy as np
import requests
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from hsi_classifier.config import DATASET_URLS, INTERIM_DATA_DIR, RAW_DATA_DIR

# Global Variables
DEFAULT_TEST_RATIO = 0.80


def ensure_directory_exists(directory_path: Path | str) -> None:
    """Ensure the directory for the given path exists."""
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)


def download_file(
    file_url: str, output_file_path: str, max_retries: int = 3, delay: int = 5
) -> None:
    """Download a file with resume support and retries."""
    retries = 0
    while retries < max_retries:
        try:
            # Check if the file already exists and get its size
            if os.path.exists(output_file_path):
                file_size = os.path.getsize(output_file_path)
                headers = {"Range": f"bytes={file_size}-"}
            else:
                file_size = 0
                headers = {}

            # Start or resume the download
            response = requests.get(file_url, headers=headers, stream=True)
            response.raise_for_status()

            # Check if the server supports partial content (resume)
            if response.status_code == 206:  # Partial Content
                logger.info(
                    f"Resuming download of {os.path.basename(output_file_path)} from byte {file_size}..."
                )
            elif response.status_code == 200:  # Full Content
                logger.info(f"Starting new download of {os.path.basename(output_file_path)}...")
            else:
                raise requests.exceptions.RequestException(
                    f"Unexpected status code: {response.status_code}"
                )

            # Write to the file in append mode if resuming
            mode = "ab" if file_size > 0 else "wb"
            with open(output_file_path, mode) as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logger.success(f"File downloaded successfully: {output_file_path}")
            return
        except requests.exceptions.RequestException as error:
            retries += 1
            logger.warning(
                f"Error downloading {file_url} (attempt {retries}/{max_retries}): {error}"
            )
            if retries < max_retries:
                time.sleep(delay)
        except Exception as error:
            logger.error(f"Exception occurred while downloading {file_url}: {error}")
            return
    logger.error(f"Failed to download {file_url} after {max_retries} retries.")


def download_single_file(file_url: str, target_directory: Path | str) -> None:
    """Download a single file if it doesn't already exist."""
    file_name = os.path.basename(file_url)
    full_file_path = os.path.join(target_directory, file_name)
    if not os.path.exists(full_file_path):
        logger.info(f"Downloading {file_name}...")
        download_file(file_url, full_file_path)
    else:
        logger.info(f"{file_name} already exists. Skipping.")


def download_multiple_files(file_urls: List[str], target_directory: Path | str) -> None:
    """Download multiple files if they don't already exist."""
    for file_url in file_urls:
        download_single_file(file_url, target_directory)


def get_dataset_urls(dataset_type: str) -> List[str]:
    """Return the URLs for the specified dataset type."""
    if dataset_type not in DATASET_URLS:
        raise ValueError("Invalid dataset_type. Choose from 'IP', 'PU', or 'SA'.")
    return DATASET_URLS[dataset_type]


def validate_dataset_type(dataset_type: str) -> str:
    """Validate and normalize the dataset type."""
    dataset_type = dataset_type.upper()
    valid_dataset_types = {"IP", "PU", "SA"}
    if dataset_type not in valid_dataset_types:
        raise ValueError(f"Invalid dataset_type. Choose from {valid_dataset_types}.")
    return dataset_type


def download_dataset(dataset_type: str, target_directory: Path | str = RAW_DATA_DIR) -> None:
    """Download data files based on the specified dataset type."""
    dataset_type = validate_dataset_type(dataset_type)
    dataset_urls = get_dataset_urls(dataset_type)
    ensure_directory_exists(target_directory)
    download_multiple_files(dataset_urls, target_directory)


def load_raw_data(
    data_type: str, data_path: Path | str = RAW_DATA_DIR
) -> tuple[np.ndarray, np.ndarray]:
    if data_type == "IP":
        data = sio.loadmat(os.path.join(data_path, "Indian_pines_corrected.mat"))[
            "indian_pines_corrected"
        ]
        labels = sio.loadmat(os.path.join(data_path, "Indian_pines_gt.mat"))["indian_pines_gt"]
    elif data_type == "SA":
        data = sio.loadmat(os.path.join(data_path, "Salinas_corrected.mat"))["salinas_corrected"]
        labels = sio.loadmat(os.path.join(data_path, "Salinas_gt.mat"))["salinas_gt"]
    elif data_type == "PU":
        data = sio.loadmat(os.path.join(data_path, "PaviaU.mat"))["paviaU"]
        labels = sio.loadmat(os.path.join(data_path, "PaviaU_gt.mat"))["paviaU_gt"]
    else:
        raise ValueError("Invalid dataset name. Choose from 'IP', 'SA', or 'PU'.")
    return data, labels


def create_traning_data(
    data: np.ndarray, labels: np.ndarray, test_size: float = DEFAULT_TEST_RATIO
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates traing and testing data from data and labels"""
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)
    return X_train, y_train, X_test, y_test


def save_interim_data(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    # Step 1: create or check directory exist
    ensure_directory_exists(INTERIM_DATA_DIR)
    # Step 2: Save the files
    np.save(os.path.join(INTERIM_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(INTERIM_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(INTERIM_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(INTERIM_DATA_DIR, "y_test.npy"), y_test)
    print(f"Interim data saved successfully to {INTERIM_DATA_DIR}")


def load_image_data(file_path: Path | str):
    return np.load(file_path)


def load_label_data(file_path: Path | str):
    return np.load(file_path)


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def main():
    # Download Indian Pines dataset
    # download_dataset("IP")
    data_type = "IP"
    X, y = load_raw_data(data_type)
    X_train, y_train, X_test, y_test = create_traning_data(X, y)
    save_interim_data(X_train, y_train, X_test, y_test)
    print(X_train.shape)
    print(y_train.shape)


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin : r + margin + 1, c - margin : c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


if __name__ == "__main__":
    main()
