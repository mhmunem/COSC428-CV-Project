# Hyperspectral Image Classification

## Project Overview

This project involves the classification of hyperspectral images (HSI) using a 3D Convolutional Neural Network (CNN). The provided Jupyter Notebook (`hsi-test-3d-cnn.ipynb`) contains the code necessary to train and test the model, leveraging various Python libraries listed in the `requirements.txt` file.

## Setup Instructions

### Prerequisites

Ensure you have the following installed on your local machine:

- Python 3.9 or later
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory> 
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
   
3. **Install Required Libraries**

    ```bash
    pip install -r requirements.txt
    ```
   
## Running the Notebook

1. **Launch Jupyter Notebook**

    ```bash
    jupyter notebook
    ```

2. **Open the Notebook**

    Navigate to the project directory in the Jupyter Notebook interface and open `hsi-test-3d-cnn.ipynb`.

3. **Execute the Notebook**

    Run the notebook cells sequentially to train and evaluate the CNN model on the provided hyperspectral image data.

## Notes

- **Data**: Ensure that the hyperspectral image data is correctly formatted and placed in the appropriate directory as specified in the notebook. Uncomment the specific data download links in the first cell. Store the data in a data folder.

- **Dependencies**: If you encounter any issues with specific versions of libraries, you might need to install them individually or adjust the `requirements.txt` file.

- **Dataset naming convention**: For ease of use chnage the `dataset` variable while running a specific data (i.e. use `IP` when running indian pines data, `SA` for Salinas Valley data and `PU` for Pavia University data)

- **Testing other models**: The compariison models can be found at the [DeepHyperX](https://github.com/nshaud/DeepHyperX) github page.


## Contact

For any questions or issues, please contact the project contributors:

- **Mohammad Munem**: [mohammad.munem@pg.canterbury.ac.nz](mailto:mohammad.munem@pg.canterbury.ac.nz)

- **Richard Green**:  [richard.green@canterbury.ac.nz](mailto:richard.green@canterbury.ac.nz)
