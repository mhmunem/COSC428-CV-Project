from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from hsi_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    logger.info("Training some model...")

    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
