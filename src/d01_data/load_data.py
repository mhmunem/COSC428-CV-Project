import os
import subprocess

def download_file(url, save_path):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Use wget with -c to resume and -O to specify output file
        result = subprocess.run(
            ["wget", "-c", url, "-O", save_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            print(f"File downloaded successfully: {save_path}")
        else:
            print(f"Error downloading {url}:\n{result.stderr}")

    except Exception as e:
        print(f"Exception occurred while downloading {url}: {e}")

def download_multiple_files(urls, save_folder):
    for url in urls:
        try:
            file_name = url.split("/")[-1]
            save_path = os.path.join(save_folder, file_name)
            download_file(url, save_path)
        except Exception as e:
            print(f"Error downloading {url}: {e}")

import os
import urllib.request

def download_multiple_files(urls, save_folder):
    for url in urls:
        filename = os.path.basename(url)
        filepath = os.path.join(save_folder, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"{filename} already exists. Skipping.")

def download_data(data_type, save_folder="./data/01_raw/"):
    data_type = data_type.upper()

    urls_dict = {
        "IP": [
            "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
            "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
        ],
        "SA": [
            "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
            "http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat"
        ],
        "PU": [
            "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
            "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
        ]
    }

    if data_type not in urls_dict:
        raise ValueError("Invalid data_type. Choose from 'IP', 'PU', or 'SA'.")

    os.makedirs(save_folder, exist_ok=True)
    download_multiple_files(urls_dict[data_type], save_folder)

if __name__ == "__main__":
    download_data(data_type="IP")
