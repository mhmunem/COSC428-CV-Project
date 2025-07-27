import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from hsi_classifier.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from hsi_classifier.utils import download_single_file, ensure_directory_exists


def split_train_test_set(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

RANDOM_STATE = 345

def save_interim_data( file_path: Path | str,
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    # Step 1: create or check directory exist
    ensure_directory_exists(file_path)
    # Step 2: Save the files
    np.save(os.path.join(file_path, "X_train.npy"), X_train)
    np.save(os.path.join(file_path, "y_train.npy"), y_train)
    np.save(os.path.join(file_path, "X_test.npy"), X_test)
    np.save(os.path.join(file_path, "y_test.npy"), y_test)
    print(f"Interim data saved successfully to {file_path}")

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

class IP_Data:
  """Downloading, Visuallizing, Spliting and Loading Indian Pines Data"""
  CLASS_NAMES = {
      0: 'Background',
      1: 'Alfalfa',
      2: 'Corn-notill',
      3: 'Corn-mintill',
      4: 'Corn',
      5: 'Grass-pasture',
      6: 'Grass-trees',
      7: 'Grass-pasture-mowed',
      8: 'Hay-windrowed',
      9: 'Oats',
      10: 'Soybean-notill',
      11: 'Soybean-mintill',
      12: 'Soybean-clean',
      13: 'Wheat',
      14: 'Woods',
      15: 'Buildings-Grass-Trees-Drives',
      16: 'Stone-Steel-Towers'
  }
  IP_DATA_DIR = RAW_DATA_DIR / "IP"
  DATA_URL = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
  LABELS_URL = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
  DATA_FILE = IP_DATA_DIR / "Indian_pines_corrected.mat"
  LABELS_FILE = IP_DATA_DIR / "Indian_pines_gt.mat"
  NUM_PCA = 15
  INTERIM_DIR = INTERIM_DATA_DIR / "IP"

  def __init__(self):
    self.data: np.ndarray
    self.labels:np.ndarray
    self.wavelengths:np.ndarray

  def download_dataset(self):
        """
        Download Indian Pines dataset if not already present
        """
        # Creating data directory if it doesn't exist
        ensure_directory_exists(self.IP_DATA_DIR)
        # Downloading data files if they don't exist
        download_single_file(self.DATA_URL, self.IP_DATA_DIR)
        download_single_file(self.LABELS_URL, self.IP_DATA_DIR)


  def load_data(self):
          """
          Load the Indian Pines dataset
          """
          # Loading the data and ground truth
          data = scipy.io.loadmat(self.DATA_FILE)
          labels = scipy.io.loadmat(self.LABELS_FILE)

          self.data = data['indian_pines_corrected']
          self.labels = labels['indian_pines_gt']

          # Generating wavelengths (approximately 400-2500 nm)
          self.wavelengths = np.linspace(400, 2500, self.data.shape[2])

          print(f"Data shape: {self.data.shape}")
          print(f"Labels shape: {self.labels.shape}")

  def split_data(self, test_ratio):
    self.data, _ = applyPCA(self.data, numComponents=self.NUM_PCA)
    X_train, X_test, y_train, y_test = train_test_split(self.data,
                                                        self.labels,
                                                        test_size=test_ratio,
                                                        random_state=RANDOM_STATE,
                                                        stratify=self.labels)
    save_interim_data(self.INTERIM_DIR,X_train, X_test, y_train, y_test)


  def create_false_color(self, bands=[30, 20, 10]):
          """
          Creating false color composite from specified bands
          """
          false_color = np.dstack([self.data[:,:,b] for b in bands])
          # Normalize to 0-1 range
          false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min())
          return false_color

  def plot_class_distribution(self):
          """
          Plotting distribution of classes in the dataset
          """
          if self.labels is None:
              raise ValueError("No data loaded")

          # Counting instances of each class
          unique, counts = np.unique(self.labels, return_counts=True)
          class_counts = dict(zip(unique, counts))

          # Creating a bar plot
          plt.figure(figsize=(15, 5))
          bars = plt.bar(
              [self.CLASS_NAMES[i] for i in unique],
              counts
          )
          plt.xticks(rotation=45, ha='right')
          plt.title('Distribution of Classes in Indian Pines Dataset')
          plt.xlabel('Class')
          plt.ylabel('Number of Pixels')

          # Adding value labels on top of bars
          for bar in bars:
              height = bar.get_height()
              plt.text(
                  bar.get_x() + bar.get_width()/2., height,
                  f'{int(height)}',
                  ha='center', va='bottom'
              )

          plt.tight_layout()
          plt.show()

  def plot_sample_spectra(self):
          """
          Plotting sample spectra from different classes
          """
          plt.figure(figsize=(15, 5))

          # Select a few major classes
          selected_classes = [2, 3, 10, 11, 14]  # Corn and Soybean varieties, Woods

          for class_id in selected_classes:
              # Get pixels belonging to this class
              mask = self.labels == class_id
              if np.any(mask):
                  # Get mean spectrum for this class
                  mean_spectrum = np.mean(self.data[mask], axis=0)
                  plt.plot(self.wavelengths, mean_spectrum, label=self.CLASS_NAMES[class_id])

          plt.title('Average Spectra for Major Classes')
          plt.xlabel('Wavelength (nm)')
          plt.ylabel('Reflectance')
          plt.legend()
          plt.grid(True)
          plt.show()

  def plot_comprehensive_view(self):
          """
          Create a comprehensive visualization of the dataset
          """
          fig, axes = plt.subplots(2, 2, figsize=(15, 15))

          # False color composite
          false_color = self.create_false_color()
          axes[0, 0].imshow(false_color)
          axes[0, 0].set_title('False Color Composite')

          # Ground truth
          im = axes[0, 1].imshow(self.labels)
          axes[0, 1].set_title('Ground Truth Classes')
          plt.colorbar(im, ax=axes[0, 1])

          # Single band visualization
          mid_band = self.data.shape[2] // 2
          im = axes[1, 0].imshow(self.data[:, :, mid_band])
          axes[1, 0].set_title(f'Single Band ({int(self.wavelengths[mid_band])}nm)')
          plt.colorbar(im, ax=axes[1, 0])

          # Spectral variance
          spectral_variance = np.std(self.data, axis=2)
          im = axes[1, 1].imshow(spectral_variance)
          axes[1, 1].set_title('Spectral Variance')
          plt.colorbar(im, ax=axes[1, 1])

          plt.tight_layout()
          plt.show()

  def visualize_dataset(self):
          """
          Create all visualizations
          """
          print("Creating comprehensive visualization...")
          self.plot_comprehensive_view()

          print("\nPlotting class distribution...")
          self.plot_class_distribution()

          print("\nPlotting sample spectra...")
          self.plot_sample_spectra()


def main():
    ip_data = IP_Data()
    # ip_data.download_dataset()
    ip_data.load_data()
    # ip_data.visualize_dataset()
    ip_data.split_data(0.8)


if __name__ == "__main__":
    main()
