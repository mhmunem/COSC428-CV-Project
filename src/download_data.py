import os
from pathlib import Path
import subprocess

from loguru import logger

import typer

app = typer.Typer()


# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

LOG_DIR = PROJ_ROOT / ".logs"
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"


DATASET_METADATA = {
    "IP": {
        "data_url": "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        "label_url": "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        "rgb_bands": (43, 21, 11),    
        "classes": 16,
        "class_names": [
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ],
    },
    "SA": {
        "data_url": "http://www.ehu.es/ccwintco/uploads/a/a3/Salinas_corrected.mat",
        "label_url": "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
        "rgb_bands": (43, 21, 11),    
        "classes": 16,
        "class_names": [
            "Broccoli_green_weeds_1",
            "Broccoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ],
    },
    "PU": {
        "data_url": "http://www.ehu.es/ccwintco/uploads/e/ee/PaviaU.mat",
        "label_url": "http://www.ehu.es/ccwintco/uploads/5/50/PaviaU_gt.mat",
        "rgb_bands": (55, 41, 12),    
        "classes": 9,
        "class_names": [
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ],
    },
    "BS": {
        "data_url":"http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat",
        "label_url":"http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat",
        "rgb_bands": (75, 33, 15),    
        "classes": 14,
        "class_names": [
            "Water",
            "Hippo grass",
            "Floodplain grasses 1",
            "Floodplain grasses 2",
            "Reeds",
            "Riparian",
            "Firescar",
            "Island interior",
            "Acacia woodlands",
            "Acacia shrublands",
            "Acacia grasslands",
            "Short mopane",
            "Mixed mopane",
            "Exposed soils",
        ],
    },
    
}





DEFAULT_DATASET = "IP"


class HSIDownloader:
    def __init__(
        self,
        data_name: str,
        data_url: str,
        label_url: str,
        **metadata
        ):
        self.data_name = data_name
        self.data_url = data_url
        self.label_url = label_url
        self.dataset_dir = RAW_DATA_DIR / data_name
        self.data_file_path = self.dataset_dir / "data.mat"
        self.label_file_path = self.dataset_dir / "labels.mat"
        self.rgb_bands =  metadata.get("rgb_bands")  
        self.num_classes = metadata.get("num_classes")
        self.class_names = metadata.get("class_names") or [f"Class {i}" for i in range(self.num_classes or 0)]

    def download(self, force: bool = False) ->  bool:
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

