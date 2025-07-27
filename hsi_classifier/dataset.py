# from pathlib import Path
# from typing import Any

# import numpy as np
# from sklearn.decomposition import PCA
# import torch
# from torch.utils.data import DataLoader, Dataset

# from hsi_classifier.config import RAW_DATA_DIR

# # from hsi_classifier.load_data import load_image_data, load_label_data


# # GLOBAL VARIABLES
# TRAIN_DATA = Path(f"{INTERIM_DATA_DIR}/X_train.npy")
# TRAIN_LABELS = Path(f"{INTERIM_DATA_DIR}/y_train.npy")
# TEST_DATA = Path(f"{INTERIM_DATA_DIR}/X_test.npy")
# TEST_LABELS = Path(f"{INTERIM_DATA_DIR}/y_test.npy")
# PCA_COMPONENT = 15
# BATCH_SIZE = 64
# PATCH_SIZE = 25


# from scipy.io import loadmat
# from sklearn.model_selection import train_test_split


# class HSIDataset(Dataset):
#     def __init__(self, dataset_name, patch_size, train_rate=0.3, val_rate=0.1, n_pca=15):
#         self.data, self.labels = self.load_data(dataset_name)
#         self.data, self.pca = self.apply_pca(self.data, n_pca)
#         self.train_gt, self.test_gt = self.sample_gt(self.labels, train_rate)
#         self.val_gt, self.test_gt = self.sample_gt(self.test_gt, val_rate / (1 - train_rate))
#         self.patch_size = patch_size

#     def load_data(self, dataset_name):
#         # Load dataset-specific data
#         data_path = f"{RAW_DATA_DIR}/{dataset_name}/data.mat"
#         label_path = f"{RAW_DATA_DIR}/{dataset_name}/labels.mat"
#         data = loadmat(data_path)
#         first_key = next(key for key in data.keys() if not key.startswith('__'))
#         data = data[first_key]
#         labels = loadmat(label_path)
#         first_key = next(key for key in labels.keys() if not key.startswith('__'))
#         labels = labels[first_key]
#         return data, labels

#     def apply_pca(self, data, n_components):
#         # Apply PCA for dimensionality reduction
#         new_data = np.reshape(data, (-1, data.shape[2]))
#         pca = PCA(n_components=n_components, whiten=True)
#         new_data = pca.fit_transform(new_data)
#         return np.reshape(new_data, (data.shape[0], data.shape[1], n_components)), pca

#     def sample_gt(self, gt, train_rate):
#         # Split data into training and testing sets
#         indices = np.nonzero(gt)
#         X = list(zip(*indices))
#         y = gt[indices].ravel()
#         train_gt, test_gt = train_test_split(X, train_size=train_rate, stratify=y, random_state=100)
#         return train_gt, test_gt

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         # Generate 3D patches
#         x, y = self.indices[idx]
#         x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
#         x2, y2 = x1 + self.patch_size, y1 + self.patch_size
#         patch = self.data[x1:x2, y1:y2]
#         label = self.labels[x, y]
#         return patch, label


# def create_dataloader(
#     batch_size: int, data_path: Path, label_path: Path, shuffle: bool = True
# ) -> DataLoader[Any]:
#     data = load_image_data(data_path)
#     label_data = load_label_data(label_path)
#     return DataLoader(
#         dataset=HSI_data(data, label_data, patch_size=PATCH_SIZE),
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=0,
#     )


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load dataset
#     dataset = HSIDataset(dataset_name="SA", patch_size=25, train_rate=0.3, val_rate=0.1, n_pca=15)

#     # # Initialize model
#     # model = HybridSN(in_channels=15, patch_size=25, num_classes=16)

#     # # Train the model
#     # trainer = Trainer(model, dataset, device)
#     # trainer.train()

#     # # Evaluate the model
#     # trainer.evaluate()

