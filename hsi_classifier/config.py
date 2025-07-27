from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

LOG_DIR = PROJ_ROOT / ".logs"
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_METADATA = {
    "IP": {
        "data_url": "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        "label_url": "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        "classes": 16,
        "class_names": [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
            "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
            "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
            "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
        ]
    },
    "SA": {
        "data_url": "http://www.ehu.es/ccwintco/uploads/a/a3/Salinas_corrected.mat",
        "label_url": "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
        "classes": 16,
        "class_names": [
            "Broccoli_green_weeds_1", "Broccoli_green_weeds_2", "Fallow",
            "Fallow_rough_plow", "Fallow_smooth", "Stubble", "Celery",
            "Grapes_untrained", "Soil_vinyard_develop", "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk", "Lettuce_romaine_5wk", "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk", "Vinyard_untrained", "Vinyard_vertical_trellis"
        ]
    },
    "PU": {
        "data_url": "http://www.ehu.es/ccwintco/uploads/e/ee/PaviaU.mat",
        "label_url": "http://www.ehu.es/ccwintco/uploads/5/50/PaviaU_gt.mat",
        "classes": 9,
        "class_names": [
            "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
            "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"
        ]
    }
}


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
