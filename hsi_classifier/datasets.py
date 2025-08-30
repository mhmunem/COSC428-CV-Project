import os
from pathlib import Path
import subprocess

from loguru import logger
import typer

from hsi_classifier.config import DATASET_METADATA, LOG_DIR, RAW_DATA_DIR

app = typer.Typer()


DEFAULT_DATASET = "IP"


class HSIDownloader:
    def __init__(self, data_name: str, data_url: str, label_url: str, **metadata):
        self.data_name = data_name
        self.data_url = data_url
        self.label_url = label_url
        self.dataset_dir = RAW_DATA_DIR / data_name
        self.data_file_path = self.dataset_dir / "data.mat"
        self.label_file_path = self.dataset_dir / "labels.mat"
        self.rgb_bands = metadata.get("rgb_bands")
        self.num_classes = metadata.get("num_classes")
        self.class_names = metadata.get("class_names") or [
            f"Class {i}" for i in range(self.num_classes or 0)
        ]

    def download(self, force: bool = False) -> bool:
        """Download both data and label files. Returns (data_path, label_path, success)"""
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        success = True

        # Download label file
        if not self.label_file_path.exists() or force:
            logger.info(f"Downloading labels for {self.data_name} from {self.label_url}")
            if not self._download_file(url=self.label_url, path=self.label_file_path):
                success = False
        else:
            logger.info(f"Label file already exists at {self.label_file_path}")

        # Download data file
        if not self.data_file_path.exists() or force:
            logger.info(f"Downloading data for {self.data_name} from {self.data_url}")
            if not self._download_file(url=self.data_url, path=self.data_file_path):
                success = False
        else:
            logger.info(f"Data file already exists at {self.data_file_path}")

        return success

    def _download_file(self, url: str, path: Path) -> bool:
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

    def get_metadata(self) -> dict:
        return {
            "name": self.data_name,
            "num_classes": self.num_classes,
            "rgb_bands": self.rgb_bands,
            "class_names": self.class_names,
            "data_file": str(self.data_file_path),
            "label_file": str(self.label_file_path),
        }


@app.command()
def main(data_name: str = DEFAULT_DATASET):
    # Setup logging
    logger.add(f"{LOG_DIR}/download.log", rotation="1 MB")
    logger.info("Downloading dataset...")

    # Metadata dictionary
    datasets = DATASET_METADATA

    # Check if dataset exists
    if data_name not in datasets:
        logger.error(f"Dataset '{data_name}' not found in metadata.")
        return

    metadata = datasets[data_name]
    logger.info(f"Downloading dataset: {data_name}")

    downloader = HSIDownloader(
        data_name=data_name,
        data_url=metadata.get("data_url"),
        label_url=metadata.get("label_url"),
        num_classes=metadata.get("classes"),
        class_names=metadata.get("class_names"),
        rgb_bands=metadata.get("rgb_bands"),
    )

    success = downloader.download()

    if success:
        metadata = downloader.get_metadata()
        logger.success(f"Download complete: {metadata}")
    else:
        logger.error("Download failed")


if __name__ == "__main__":
    app()
