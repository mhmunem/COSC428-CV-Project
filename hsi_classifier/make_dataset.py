import os
from pathlib import Path
import subprocess

from loguru import logger

# from tqdm import tqdm
import typer

from hsi_classifier.config import DATASET_METADATA, LOG_DIR, RAW_DATA_DIR

app = typer.Typer()

DEFAULT_DATASET = "IP"


class HSIDataDownloader:
    def __init__(
        self,
        data_name: str,
        data_url: str,
        label_url: str,
        base_dir: str | Path = RAW_DATA_DIR,
        num_classes=None,
        class_names=None,
    ):
        self.data_name = data_name
        self.data_url = data_url
        self.label_url = label_url
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / data_name
        self.data_file_path = self.dataset_dir / "data.mat"
        self.label_file_path = self.dataset_dir / "labels.mat"
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes or 0)]

    def download(self, force: bool = False) -> tuple[Path, Path]:
        """Download both data and label files."""
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download label file
        if not self.label_file_path.exists() or force:
            logger.info(f"Downloading labels for {self.data_name} from {self.label_url}")
            self._download_file(url=self.label_url, path=self.label_file_path)
        else:
            logger.info(f"Label file already exists at {self.label_file_path}")

        # Download data file
        if not self.data_file_path.exists() or force:
            logger.info(f"Downloading data for {self.data_name} from {self.data_url}")
            self._download_file(url=self.data_url, path=self.data_file_path)
        else:
            logger.info(f"Data file already exists at {self.data_file_path}")

        return self.data_file_path, self.label_file_path

    def _download_file(self, url: str, path: Path) -> None:
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Use wget with -c to resume and -O to specify output file
            result = subprocess.run(
                ["wget", "-c", url, "-O", path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode == 0:
                logger.info(f"File downloaded successfully: {path}")
            else:
                logger.error(f"Error downloading {url}:\n{result.stderr}")

        except Exception as e:
            logger.error(f"Exception occurred while downloading {url}: {e}")

    def get_metadata(self) -> dict:
        return {
            "name": self.data_name,
            "num_classes": self.num_classes,
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

    info = datasets[data_name]
    logger.info(f"Downloading dataset: {data_name}")

    downloader = HSIDataDownloader(
        data_name=data_name,
        data_url=info["data_url"],
        label_url=info["label_url"],
        num_classes=info["classes"],
        class_names=info["class_names"],
    )

    downloader.download()
    # metadata = downloader.get_metadata()
    # logger.success(f"Download complete: {metadata}")


if __name__ == "__main__":
    app()
