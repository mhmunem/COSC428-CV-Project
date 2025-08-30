import os
from pathlib import Path
import subprocess
from typing import Union

from loguru import logger
import numpy as np
from scipy.io import loadmat

from hsi_classifier.config import DATASET_METADATA


def get_metadata(data_name: str) -> dict:
    """
    Retrieve metadata for a specified dataset.

    This function looks up dataset metadata using a global configuration dictionary
    (e.g., `DATASET_METADATA`). If the dataset name is not found, an error is logged
    and the function returns None. Otherwise, it returns a structured dictionary
    containing key details such as URLs, class count, RGB band indices, and class names.

    Parameters
    ----------
    data_name : str
        The name of the dataset to retrieve metadata for (e.g., "IP", "SA", "PU").
        Must exactly match a key in the `DATASET_METADATA` dictionary.

    Returns
    -------
    dict or None
        A dictionary containing the following keys:
            - name (str): Dataset name.
            - data_url (str): URL to the hyperspectral data `.mat` file.
            - label_url (str): URL to the labels `.mat` file.
            - num_classes (int): Number of land cover or material classes.
            - rgb_bands (tuple or list): Indices of bands that approximate RGB channels.
            - class_names (list of str): List of class labels (e.g., ['Tree', 'Grass', 'Road']).
        Returns None if the dataset name is not found.
    """
    datasets = DATASET_METADATA
    if data_name not in datasets:
        logger.error(f"Dataset '{data_name}' not found in metadata.")
        return
    metadata = datasets[data_name]

    return {
        "name": data_name,
        "data_url": metadata.get("data_url"),
        "label_url": metadata.get("label_url"),
        "num_classes": metadata.get("classes"),
        "rgb_bands": metadata.get("rgb_bands"),
        "class_names": metadata.get("class_names"),
    }


def download_file(url: str, path: Path) -> bool:
    """
    Download a single file from a URL to a specified local path using `wget`.

    This function uses the system's `wget` command to download a file with progress
    display and resume capability (`-c` flag). It ensures the parent directory exists
    before starting the download. On failure, logs detailed error information.

    Parameters
    ----------
    url : str
        The full URL to the file to be downloaded. Must be accessible over HTTP/HTTPS.
    path : pathlib.Path
        The local filesystem path where the file should be saved. Parent directories
        will be created if they do not exist.

    Returns
    -------
    bool
        True if the download was successful (exit code 0 from `wget`).
        False if the command failed, was interrupted, or raised an exception.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        result = subprocess.run(
            ["wget", "-c", "--progress=bar:force", url, "-O", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:
            logger.info(f"File downloaded successfully: {path}")
            return True
        else:
            logger.error(f"Error downloading {url}:\n{result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Exception occurred while downloading {url}: {e}")
        return False


def get_mat_data(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load a MATLAB (.mat) file and return the first user-visible variable as a NumPy array.

    This function reads a .mat file using scipy.io.loadmat and extracts the first
    variable that does not start with '__', which are reserved for internal use.
    It is useful for loading hyperspectral datasets (e.g., Indian Pines, Salinas)
    where the data is stored as a single array in a .mat file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the .mat file to be loaded. Can be a string or Path object.

    Returns
    -------
    numpy.ndarray
        The data array from the .mat file. Typically a 2D or 3D array representing
        hyperspectral image data (e.g., height × width × bands).
    """
    data = loadmat(file_path)

    try:
        first_key = next(key for key in data.keys() if not key.startswith("__"))
    except StopIteration:
        raise ValueError(f"No valid data found in {file_path}. All keys are internal.")

    return data[first_key]
