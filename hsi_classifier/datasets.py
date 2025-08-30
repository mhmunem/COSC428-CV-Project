from loguru import logger
import typer

from hsi_classifier.config import RAW_DATA_DIR
from hsi_classifier.utils import download_file, get_mat_data, get_metadata

app = typer.Typer()


class URLLoader:
    def __init__(self, data_name):
        self.data_name = data_name
        self.metadata = get_metadata(self.data_name)

        if not self.metadata:
            raise typer.Exit(code=1)

        self.dataset_dir = RAW_DATA_DIR / data_name
        self.data_file_path = self.dataset_dir / "data.mat"
        self.label_file_path = self.dataset_dir / "labels.mat"
        self.data_url = self.metadata["data_url"]
        self.label_url = self.metadata["label_url"]
        self.force = False

    def get_data(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        success = True

        # Download data file
        if not self.data_file_path.exists() or self.force:
            logger.info(f"Downloading data for {self.data_name} from {self.data_url}")
            if not download_file(url=self.data_url, path=self.data_file_path):
                success = False
        else:
            logger.info(f"Data file already exists at {self.data_file_path}")

        if success:
            logger.success(f"Data download complete: {self.data_name}")
        else:
            logger.error("Data download failed")

        data = get_mat_data(self.data_file_path)

        return data

    def get_labels(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        success = True

        # Download label file
        if not self.label_file_path.exists() or self.force:
            logger.info(f"Downloading labels for {self.data_name} from {self.label_url}")
            if not download_file(url=self.label_url, path=self.label_file_path):
                success = False
        else:
            logger.info(f"Label file already exists at {self.label_file_path}")

        if success:
            logger.success(f"Label download complete: {self.data_name}")
        else:
            logger.error("Label download failed")

        labels = get_mat_data(self.label_file_path)

        return labels


@app.command()
def main(
    data_name = "IP"
):
    url_loader = URLLoader(data_name=data_name)
    labels = url_loader.get_labels()
    data = url_loader.get_data()

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Label shape: {labels.shape}")


if __name__ == "__main__":
    app()
