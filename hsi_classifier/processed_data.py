# from pathlib import Path

# from loguru import logger
# import numpy as np
# from scipy.io import loadmat
# from sklearn.model_selection import train_test_split
# import torch
# from torch.utils.data import Dataset
# import typer

# from hsi_classifier.config import LOG_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

# app = typer.Typer()

# DEFAULT_DATASET = "IP"
# DEFAULT_TEST_SIZE = 0.2
# DEFAULT_VAL_SIZE = 0.1
# RANDOM_STATE = 345
# PATCH_SIZE = 2
# IGNORED_LABELS = [0]
# SUPERVISION = "full"

# class HSIProcessor(Dataset):
#     """ Generic class for a hyperspectral scene """

#     def __init__(self,
#                  data_path: Path | str,
#                  label_path: Path | str,
#                  save_path: Path | str,
#                  **hyperparams):
#         super(HSIProcessor, self).__init__()
#         self.data_path = Path(data_path)
#         self.label_path = Path(label_path)
#         self.save_path = Path(save_path)
#         self.name = hyperparams.get("dataset", "unknown")
#         self.patch_size = hyperparams.get("patch_size", PATCH_SIZE)
#         self.ignored_labels = set(hyperparams.get("ignored_labels", IGNORED_LABELS))
#         self.flip_augmentation = hyperparams.get("flip_augmentation", False)
#         self.radiation_augmentation = hyperparams.get("radiation_augmentation", False)
#         self.mixture_augmentation = hyperparams.get("mixture_augmentation", False)
#         self.center_pixel = hyperparams.get("center_pixel", True)
#         supervision = hyperparams.get("supervision", SUPERVISION)

#         # Create save directory
#         self.save_path.mkdir(parents=True, exist_ok=True)

#         # Load data and labels
#         self._load()

#         # Build mask based on supervision
#         if supervision == "full":
#             mask = np.ones_like(self.label)
#             for l in self.ignored_labels:
#                 mask[self.label == l] = 0
#         elif supervision == "semi":
#             mask = np.ones_like(self.label)

#         x_pos, y_pos = np.nonzero(mask)
#         p = self.patch_size // 2
#         self.indices = np.array([
#             (x, y)
#             for x, y in zip(x_pos, y_pos)
#             if x > p and x < self.data.shape[0] - p and y > p and y < self.data.shape[1] - p
#         ])
#         self.labels = np.array([self.label[x, y] for x, y in self.indices])
#         np.random.shuffle(self.indices)

#     def _load(self) -> None:
#         logger.info(f"Loading data from {self.data_path}")
#         data = loadmat(self.data_path)
#         first_key = next(key for key in data.keys() if not key.startswith("__"))
#         self.data = data[first_key]

#         logger.info(f"Loading labels from {self.label_path}")
#         labels = loadmat(self.label_path)
#         first_key = next(key for key in labels.keys() if not key.startswith("__"))
#         self.label = labels[first_key]

#         logger.success(f"Loaded data with shape {self.data.shape} and labels with shape {self.label.shape}")

#     @staticmethod
#     def flip(*arrays):
#         horizontal = np.random.random() > 0.5
#         vertical = np.random.random() > 0.5
#         if horizontal:
#             arrays = [np.fliplr(arr) for arr in arrays]
#         if vertical:
#             arrays = [np.flipud(arr) for arr in arrays]
#         return arrays

#     @staticmethod
#     def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
#         alpha = np.random.uniform(*alpha_range)
#         noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
#         return alpha * data + beta * noise

#     def mixture_noise(self, data, label, beta=1 / 25):
#         alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
#         noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
#         data2 = np.zeros_like(data)
#         for idx, value in np.ndenumerate(label):
#             if value not in self.ignored_labels:
#                 l_indices = np.nonzero(self.labels == value)[0]
#                 if len(l_indices) > 0:
#                     l_indice = np.random.choice(l_indices)
#                     x, y = self.indices[l_indice]
#                     data2[idx] = self.data[x, y]
#         return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, i):
#         x, y = self.indices[i]
#         p = self.patch_size // 2
#         x1, y1 = x - p, y - p
#         x2, y2 = x1 + self.patch_size, y1 + self.patch_size

#         data = self.data[x1:x2, y1:y2]
#         label = self.label[x1:x2, y1:y2]

#         if self.flip_augmentation and self.patch_size > 1:
#             data, label = self.flip(data, label)
#         if self.radiation_augmentation and np.random.random() < 0.1:
#             data = self.radiation_noise(data)
#         if self.mixture_augmentation and np.random.random() < 0.2:
#             data = self.mixture_noise(data, label)

#         data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
#         label = np.asarray(np.copy(label), dtype="int64")

#         data = torch.from_numpy(data)
#         label = torch.from_numpy(label)

#         if self.center_pixel and self.patch_size > 1:
#             label = label[self.patch_size // 2, self.patch_size // 2]
#         elif self.patch_size == 1:
#             data = data[:, 0, 0]
#             label = label[0, 0]

#         if self.patch_size > 1:
#             data = data.unsqueeze(0)

#         return data, label

# def save_dataset_splits(train_dataset, val_dataset, test_dataset, save_path):
#     """Save the dataset splits to disk"""
#     save_path = Path(save_path)
#     save_path.mkdir(parents=True, exist_ok=True)

#     logger.info(f"Saving dataset splits to {save_path}")

#     # Save train dataset
#     train_data = []
#     train_labels = []
#     for i in range(len(train_dataset)):
#         data, label = train_dataset[i]
#         train_data.append(data.numpy())
#         train_labels.append(label.item())

#     np.save(save_path / "train_data.npy", np.array(train_data))
#     np.save(save_path / "train_labels.npy", np.array(train_labels))

#     # Save validation dataset
#     val_data = []
#     val_labels = []
#     for i in range(len(val_dataset)):
#         data, label = val_dataset[i]
#         val_data.append(data.numpy())
#         val_labels.append(label.item())

#     np.save(save_path / "val_data.npy", np.array(val_data))
#     np.save(save_path / "val_labels.npy", np.array(val_labels))

#     # Save test dataset
#     test_data = []
#     test_labels = []
#     for i in range(len(test_dataset)):
#         data, label = test_dataset[i]
#         test_data.append(data.numpy())
#         test_labels.append(label.item())

#     np.save(save_path / "test_data.npy", np.array(test_data))
#     np.save(save_path / "test_labels.npy", np.array(test_labels))

#     logger.success("Dataset splits saved successfully")

# def save_metadata(dataset, save_path):
#     """ Save dataset metadata """
#     save_path = Path(save_path)
#     metadata = {
#         'dataset_name': dataset.name,
#         'patch_size': dataset.patch_size,
#         'data_shape': dataset.data.shape,
#         'label_shape': dataset.label.shape,
#         'num_samples': len(dataset),
#         'ignored_labels': list(dataset.ignored_labels),
#         'indices_shape': dataset.indices.shape
#     }

#     np.save(save_path / "metadata.npy", metadata)
#     logger.info("Metadata saved")

# def split_dataset(dataset,
#                  test_size=DEFAULT_TEST_SIZE,
#                  val_size=DEFAULT_VAL_SIZE,
#                  random_state=RANDOM_STATE):
#     """Split dataset into train, validation, and test sets"""
#     total_size = len(dataset)
#     test_size_count = int(test_size * total_size)
#     val_size_count = int(val_size * total_size)
#     train_size_count = total_size - test_size_count - val_size_count

#     indices = list(range(total_size))
#     train_indices, temp_indices = train_test_split(
#         indices, test_size=test_size_count + val_size_count, random_state=random_state
#     )
#     val_indices, test_indices = train_test_split(
#         temp_indices, test_size=test_size_count, random_state=random_state
#     )

#     train_dataset = torch.utils.data.Subset(dataset, train_indices)
#     val_dataset = torch.utils.data.Subset(dataset, val_indices)
#     test_dataset = torch.utils.data.Subset(dataset, test_indices)

#     return train_dataset, val_dataset, test_dataset

# @app.command()
# def main(data_name: str = DEFAULT_DATASET):
#     data_path = RAW_DATA_DIR / data_name / "data.mat"
#     label_path = RAW_DATA_DIR / data_name / "labels.mat"
#     save_path = PROCESSED_DATA_DIR / data_name

#     logger.add(f"{LOG_DIR}/preprocessing.log", rotation="1 MB")
#     logger.info(f"Processing dataset: {data_name}")

#     # Create HSIProcessor dataset
#     dataset = HSIProcessor(
#         data_path=data_path,
#         label_path=label_path,
#         save_path=save_path,
#         dataset=data_name,
#         patch_size=PATCH_SIZE,
#         ignored_labels=[0],
#         flip_augmentation=False,
#         radiation_augmentation=False,
#         mixture_augmentation=False,
#         center_pixel=True,
#         supervision="full"
#     )

#     # Split dataset into train/val/test
#     train_dataset, val_dataset, test_dataset = split_dataset(dataset)

#     logger.info(f"Total samples: {len(dataset)}")
#     logger.info(f"Train samples: {len(train_dataset)}")
#     logger.info(f"Validation samples: {len(val_dataset)}")
#     logger.info(f"Test samples: {len(test_dataset)}")

#     # Example of how to use the datasets
#     logger.info("Example data shapes:")
#     sample_data, sample_label = dataset[0]
#     logger.info(f"Sample data shape: {sample_data.shape}")
#     logger.info(f"Sample label: {sample_label}")

#     # Save the processed data
#     save_dataset_splits(train_dataset, val_dataset, test_dataset, save_path)
#     save_metadata(dataset, save_path)

#     logger.success(f"Processing and saving complete: {data_name}")

# if __name__ == "__main__":
#     app()
