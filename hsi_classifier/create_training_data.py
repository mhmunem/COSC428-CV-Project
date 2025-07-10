import os

from keras.utils import to_categorical
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Global configuration
DEFAULT_DATASET = "IP"
DEFAULT_TEST_RATIO = 0.80
DEFAULT_WINDOW_SIZE = 11
DEFAULT_DATA_PATH = "./data/01_raw/"


def load_data(data_type, data_path=DEFAULT_DATA_PATH):
    if data_type == "IP":
        data = sio.loadmat(os.path.join(data_path, "Indian_pines_corrected.mat"))[
            "indian_pines_corrected"
        ]
        labels = sio.loadmat(os.path.join(data_path, "Indian_pines_gt.mat"))["indian_pines_gt"]
    elif data_type == "SA":
        data = sio.loadmat(os.path.join(data_path, "Salinas_corrected.mat"))["salinas_corrected"]
        labels = sio.loadmat(os.path.join(data_path, "Salinas_gt.mat"))["salinas_gt"]
    elif data_type == "PU":
        data = sio.loadmat(os.path.join(data_path, "PaviaU.mat"))["paviaU"]
        labels = sio.loadmat(os.path.join(data_path, "PaviaU_gt.mat"))["paviaU_gt"]
    else:
        raise ValueError("Invalid dataset name. Choose from 'IP', 'SA', or 'PU'.")
    return data, labels


def apply_pca(X, num_components=75):
    reshaped_X = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    transformed_X = pca.fit_transform(reshaped_X)
    reshaped_back = np.reshape(transformed_X, (X.shape[0], X.shape[1], num_components))
    return reshaped_back, pca


def pad_with_zeros(X, margin=2):
    padded_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    padded_X[margin:-margin, margin:-margin, :] = X
    return padded_X


def create_image_cubes(X, y, window_size=5, remove_zero_labels=True):
    margin = window_size // 2
    padded_X = pad_with_zeros(X, margin)
    patches_data = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]))
    patches_labels = np.zeros((X.shape[0] * X.shape[1]))
    patch_index = 0
    for r in range(margin, padded_X.shape[0] - margin):
        for c in range(margin, padded_X.shape[1] - margin):
            patch = padded_X[r - margin : r + margin + 1, c - margin : c + margin + 1]
            patches_data[patch_index] = patch
            patches_labels[patch_index] = y[r - margin, c - margin]
            patch_index += 1
    if remove_zero_labels:
        mask = patches_labels > 0
        patches_data = patches_data[mask]
        patches_labels = patches_labels[mask] - 1
    return patches_data, patches_labels


def oversample_weak_classes(X, y):
    unique_labels, label_counts = np.unique(y, return_counts=True)
    max_count = np.max(label_counts)
    new_X = []
    new_y = []
    for label, count in zip(unique_labels, label_counts):
        repeat_factor = round(max_count / count)
        class_X = X[y == label]
        class_y = y[y == label]
        new_X.append(np.repeat(class_X, repeat_factor, axis=0))
        new_y.append(np.repeat(class_y, repeat_factor, axis=0))
    new_X = np.concatenate(new_X)
    new_y = np.concatenate(new_y)
    np.random.seed(42)
    perm = np.random.permutation(len(new_y))
    return new_X[perm], new_y[perm]


def split_train_test(X, y, test_ratio, random_state=345):
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state, stratify=y)


def processed_data(dataset, test_ratio, window_size, data_path=DEFAULT_DATA_PATH):
    X, y = load_data(dataset, data_path=data_path)
    K = 100 if dataset == "IP" else 40
    X, pca = apply_pca(X, num_components=K)
    K = X.shape[2]
    X, y = create_image_cubes(X, y, window_size=window_size)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_ratio)
    X_train, y_train = oversample_weak_classes(X_train, y_train)
    X_train = X_train.reshape(-1, window_size, window_size, K, 1)
    y_train = to_categorical(y_train)
    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    return X_train, y_train, X_test, y_test
