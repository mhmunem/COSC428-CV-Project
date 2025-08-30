from scipy.io import loadmat

file_path = '/home/munem/01-Work/01-github/COSC428-CV-Project/data/raw/PU/label.mat'
data = loadmat(file_path)
print(data)