from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


DATASET_URLS: Dict[str, List[str]] = {
        "IP": [
            "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
            "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
        ],
        "SA": [
            "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
            "http://www.ehu.es/ccwintco/uploads/a/a3/Salinas_corrected.mat"
        ],
        "PU": [
            "http://www.ehu.es/ccwintco/uploads/5/50/PaviaU_gt.mat",
            "http://www.ehu.es/ccwintco/uploads/e/ee/PaviaU.mat"
        ]
    }

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
