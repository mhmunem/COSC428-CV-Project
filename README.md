# Hyperspectral Image Classification

## Project Overview

This project involves the classification of hyperspectral images (HSI) using a 3D Convolutional Neural Network (CNN). The provided Jupyter Notebook (`20250628MM-IP-HSI.ipynb`) contains the code necessary to train and test the model, leveraging various Python libraries listed in the `requirements.txt` file.

---

## Setup Instructions

### Prerequisites

Ensure you have the following installed on your local machine:

- Python 3.9 or later
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mhmunem/COSC428-CV-Project.git
   cd COSC428-CV-Project
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv/Scripts/activate
    ```

3. **Install Required Libraries**

    ```bash
    pip install -r requirements.txt
    ```
---

## Downloadnig Data
Run the `download_data.py` file on src to download all the data.
```python 
python ./src/download_data.py
```

---

## Running the Notebooks

1. **Launch Jupyter Notebook**

    ```bash
    jupyter notebook
    ```

2. **Open the Notebook**

    Navigate to the project directory in the find the Jupyter Notebook interface and open the `20250628MM-IP-HSI.ipynb`.

3. **Execute the Notebook**

    Run the notebook cells sequentially to train and evaluate the CNN models on the provided hyperspectral image data.

---

## Notes

- **Data**: Ensure that the hyperspectral image data is correctly formatted and placed in the appropriate directory as specified in the notebook. Uncomment the specific data download links in the first cell. Store the data in a data folder.

- **Dependencies**: If you encounter any issues with specific versions of libraries, you might need to install them individually or adjust the `requirements.txt` file.

- **Dataset naming convention**: For ease of use chnage the `DATASET` variable while running a specific data (i.e. use `IP` when running indian pines data, `SA` for Salinas Valley data and `PU` for Pavia University data)

- **Testing other models**: These models can be compared with other models present in the [DeepHyperX](https://github.com/nshaud/DeepHyperX) github page.

---

## Contact

For any questions or issues, please contact the project contributors:

- **Mohammad Munem**: [mohammad.munem@canterbury.ac.nz](mailto:mohammad.munem@canterbury.ac.nz)

- **Richard Green**:  [richard.green@canterbury.ac.nz](mailto:richard.green@canterbury.ac.nz)
