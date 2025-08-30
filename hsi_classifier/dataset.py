import os
import subprocess
from pathlib import Path

from loguru import logger
import typer

from hsi_classifier.config import DATASET_METADATA, LOG_DIR, RAW_DATA_DIR

app = typer.Typer()


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

    See Also
    --------
    DATASET_METADATA : Global constant or config variable containing dataset definitions.
    download : Uses output of this function to initiate downloads.

    Examples
    --------
    >>> meta = get_metadata("IP")
    >>> if meta:
    ...     print(meta["num_classes"], "classes found")
    ... else:
    ...     print("Dataset not supported")

    Notes
    -----
    - This function logs an error via `loguru.logger` if the dataset is not found.
    - The returned dictionary is a copy of the metadata; mutating it does not affect config.
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


def download(
    data_name: str,
    data_url: str,
    label_url: str,
    force: bool = False,
) -> bool:
    """
    Download both the hyperspectral data and label files for a given dataset.

    This function checks whether the data and label `.mat` files already exist in the designated
    raw data directory. If they do not exist, or if `force` is True, the files are downloaded
    using `wget`. Progress is displayed during download. The files are saved as 'data.mat'
    and 'labels.mat' within a subdirectory named after the dataset.

    The download process is robust to interruptions and uses the `-c` (continue) flag in `wget`
    to resume partial downloads if possible.

    Parameters
    ----------
    data_name : str
        The name of the dataset. Used to create a subdirectory under `RAW_DATA_DIR`.
        Must be a valid directory name (e.g., no special characters or spaces).
    data_url : str
        The URL pointing to the hyperspectral data file (typically in `.mat` format).
    label_url : str
        The URL pointing to the corresponding label file (also in `.mat` format).
    force : bool, optional
        If True, forces re-download of both files even if they already exist locally.
        Default is False.

    Returns
    -------
    bool
        True if both the data and label files were successfully downloaded or already existed.
        False if any download failed (e.g., network error, invalid URL, permission issue).

    Side Effects
    ------------
    - Creates directories under `RAW_DATA_DIR` if they do not exist.
    - Writes two files (`data.mat` and `labels.mat`) to the dataset directory.
    - Logs progress and errors using the `loguru.logger`.

    See Also
    --------
    download_file : Performs the actual file download using subprocess and wget.
    get_metadata : Retrieves dataset metadata including URLs.

    Examples
    --------
    >>> success = download("IP", "https://example.com/data.mat", "https://example.com/labels.mat")
    >>> if success:
    ...     print("Dataset downloaded successfully.")
    ... else:
    ...     print("Download failed.")

    Notes
    -----
    - Requires `wget` to be available in the system's PATH. This function will fail if `wget`
      is not installed or not accessible.
    - Uses `subprocess.run()` to invoke `wget` with progress bar output forced via
      `--progress=bar:force`.
    - On failure, error details are logged via `logger.error`, including stderr output from `wget`.

    """

    dataset_dir = RAW_DATA_DIR / data_name
    data_file_path = dataset_dir / "data.mat"
    label_file_path = dataset_dir / "labels.mat"

    dataset_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # Download label file
    if not label_file_path.exists() or force:
        logger.info(f"Downloading labels for {data_name} from {label_url}")
        if not download_file(url=label_url, path=label_file_path):
            success = False
    else:
        logger.info(f"Label file already exists at {label_file_path}")

    # Download data file
    if not data_file_path.exists() or force:
        logger.info(f"Downloading data for {data_name} from {data_url}")
        if not download_file(url=data_url, path=data_file_path):
            success = False
    else:
        logger.info(f"Data file already exists at {data_file_path}")

    return success


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

    Raises
    ------
    subprocess.SubprocessError
        If there is an issue executing `wget` (e.g., command not found, network timeout).
        Exception is caught and logged internally; function returns False.

    Side Effects
    ------------
    - Executes external command `wget`.
    - Writes file to disk at `path`.
    - Creates parent directories if needed.
    - Logs progress, success, or error messages via `loguru.logger`.

    Notes
    -----
    - Requires `wget` to be installed and available in the system PATH.
    - Uses `--progress=bar:force` to ensure progress is visible even in non-TTY environments.
    - The `-c` flag allows resuming partial downloads.

    Examples
    --------
    >>> success = download_file("https://example.com/data.mat", Path("data/raw/IP/data.mat"))
    >>> if success:
    ...     print("File downloaded!")
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


@app.command()
def main(data_name: str = "IP", force: bool = False) -> None:
    logger.add(f"{LOG_DIR}/download.log", rotation="1 MB")
    logger.info("Starting dataset download...")

    meta = get_metadata(data_name)
    if not meta:
        raise typer.Exit(code=1)

    success = download(data_name, meta["data_url"], meta["label_url"], force=force)

    if success:
        logger.success(f"Download completed successfully: {data_name}")
    else:
        logger.error(f"Failed to download dataset: {data_name}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
