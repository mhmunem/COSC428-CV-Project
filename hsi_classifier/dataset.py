from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from hsi_classifier.config import INTERIM_DATA_DIR
from hsi_classifier.load_data import load_image_data, load_label_data

# GLOBAL VARIABLES
DATA_DIR = INTERIM_DATA_DIR
TEST_DATA = Path(f"{DATA_DIR}/X_test.npy")
TEST_LABELS = Path(f"{DATA_DIR}/y_test.npy")
TRAIN_DATA = Path(f"{DATA_DIR}/X_train.npy")
TRAIN_LABELS = Path(f"{DATA_DIR}/y_train.npy")


class HSI_data(Dataset[Any]):
    """HSI Data class"""

    idx: int  # requested data index
    x: Tensor
    y: Tensor

    # CLASS VARIABLES
    # TRAIN_MAX = 255.0
    # TRAIN_NORMALIZED_MEAN = 0.1306604762738429
    # TRAIN_NORMALIZED_STDEV = 0.3081078038564622

    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        """Initias the class with data and labels"""
        if len(data) != len(targets):
            raise ValueError(
                f"data and targets must be the same length.{len(data)} != {len(targets)}"
            )

        self.data = data
        self.targets = targets

    def __len__(self):
        """Calculates length of the data"""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Gets x and y index from the dataset"""
        x = self.get_x(idx)
        y = self.get_y(idx)
        return x, y

    def get_x(self, idx: int) -> Tensor:
        """Gets index and preprocessed x from the dataset"""
        self.idx = idx
        self.preprocess_x()
        return self.x

    def preprocess_x(self) -> None:
        """Preprocesses x"""
        self.x = self.data[self.idx].copy().astype(np.float64)
        # self.x /= self.TRAIN_MAX
        # self.x -= self.TRAIN_NORMALIZED_MEAN
        # self.x /= self.TRAIN_NORMALIZED_STDEV
        # self.x = self.x.astype(np.float32)
        # self.x = torch.from_numpy(self.x)
        # self.x = self.x.unsqueeze(0)

    def get_y(self, idx: int) -> Tensor:
        self.idx = idx
        self.preprocess_y()
        return self.y

    def preprocess_y(self) -> None:
        """Preprocesses y"""
        self.y = self.targets[self.idx]
        # self.y = torch.tensor(self.y, dtype=torch.long)


# def apply_pca(X, num_components=75):
#     reshaped_X = np.reshape(X, (-1, X.shape[2]))
#     pca = PCA(n_components=num_components, whiten=True)
#     transformed_X = pca.fit_transform(reshaped_X)
#     reshaped_back = np.reshape(transformed_X, (X.shape[0], X.shape[1], num_components))
#     return reshaped_back

def create_dataloader(
    batch_size: int, data_path: Path, label_path: Path, shuffle: bool = True
) -> DataLoader[Any]:
    data = load_image_data(data_path)
    label_data = load_label_data(label_path)
    return DataLoader(
        dataset=HSI_data(data, label_data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )



def main():
    TRAIN_DATA = Path(f"{DATA_DIR}/X_train.npy")
    # TRAIN_LABELS = Path(f"{DATA_DIR}/y_train.npy")
    data = load_image_data(TRAIN_DATA)
    print(data.shape)
    # K = 100 if data.shape[1] == 145 else 40
    # data, _ = apply_pca(data, num_components=K)

    # dt=create_dataloader(64,TRAIN_DATA,TRAIN_LABELS,shuffle=False)
    # # Test the DataLoader
    # print("Testing DataLoader...\n")

    # # Iterate through a few batches
    # for batch_idx, (batch_data, batch_labels) in enumerate(dt):
    #     print(f"Batch {batch_idx + 1}")
    #     print(f"Batch data shape: {batch_data.shape}")  # Should be (batch_size, ...)
    #     print(f"Batch labels shape: {batch_labels.shape}")  # Should be (batch_size,)

    #     # Print the first sample in the batch
    #     print("\nFirst sample in batch:")
    #     print(f"Data: {batch_data[0]}")
    #     print(f"Label: {batch_labels[0]}")

    #     # Stop after printing a few batches
    #     if batch_idx >= 2:  # Print only 3 batches
    #         break

    # print("\nDataLoader test complete.")


if __name__ == "__main__":
    main()
