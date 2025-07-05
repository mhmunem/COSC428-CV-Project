import os
import requests
from typing import List
from loguru import logger
from pathlib import Path
from hsi_classifier.config import RAW_DATA_DIR, DATASET_URLS


def ensure_directory_exists(directory_path: Path | str) -> None:
    """Ensure the directory for the given path exists."""
    if isinstance(directory_path, str):
      directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)


import os
import requests
from loguru import logger
import time

def download_file(file_url: str, output_file_path: str, max_retries: int = 3, delay: int = 5) -> None:
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
                logger.info(f"Resuming download of {os.path.basename(output_file_path)} from byte {file_size}...")
            elif response.status_code == 200:  # Full Content
                logger.info(f"Starting new download of {os.path.basename(output_file_path)}...")
            else:
                raise requests.exceptions.RequestException(f"Unexpected status code: {response.status_code}")

            # Write to the file in append mode if resuming
            mode = "ab" if file_size > 0 else "wb"
            with open(output_file_path, mode) as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logger.success(f"File downloaded successfully: {output_file_path}")
            return
        except requests.exceptions.RequestException as error:
            retries += 1
            logger.warning(f"Error downloading {file_url} (attempt {retries}/{max_retries}): {error}")
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


def main():
    # Download Indian Pines dataset
    download_dataset("IP")
    download_dataset("SA")
    download_dataset("PU")


if __name__ == "__main__":
    main()
