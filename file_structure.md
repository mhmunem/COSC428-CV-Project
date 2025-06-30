## Project File Structure
The principles for this file structure is given in this [link](https://github.com/dssg/hitchhikers-guide/tree/master/sources/curriculum/0_before_you_start/pipelines-and-project-workflow).


```
├── README.md          <- The top-level README for developers.
├── conf               <- Space for credentials
│
├── data
│   ├── 01_raw         <- Immutable input data
│   ├── 02_intermediate<- Cleaned version of raw
│   ├── 03_processed   <- Data used to develop models
│   ├── 04_models      <- trained models
│   ├── 05_model_output<- model output
│   └── 06_reporting   <- Reports and input to frontend
│
├── docs               <- Space for Sphinx documentation
│
├── notebooks          <- Jupyter notebooks. Naming convention is
|                         date YYYYMMDD (for ordering),
│                         the creator's initials, and a short `-`
|                         delimited description.
│
├── references         <- Data dictionaries, manuals, etc.
│
├── results            <- Final analysis docs.
│
├── requirements.txt   <- The requirements file for reproducing the
|                         analysis environment.
│
├── .gitignore         <- Avoids uploading data, credentials,
|                         outputs, system files etc
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── d00_utils      <- Functions used across the project
    │
    │
    ├── d01_data       <- Scripts to reading and writing data etc
    │   └── load_data.py
    │
    ├── d02_intermediate<- Scripts to transform data from raw to
    |                      intermediate(data is already cleaned so this is not needed.)
    │
    │
    ├── d03_processing <- Scripts to turn intermediate data into
    |   |                 modelling input
    │   └── create_training_data.py
    │
    ├── d04_modelling  <- Scripts to train models and then use
    |   |                  trained models to make predictions.
    │   └── train_model.py
    │
    ├── d05_model_evaluation<- Scripts that analyse model
    |   |                      performance and model selection.
    │   └── calculate_performance_metrics.py
    │
    ├── d06_reporting  <- Scripts to produce report.
    │   └── report.py
    │
    └── d07_visualisation<- Scripts to create frequently used plots
    │   └── visualise.py
    │
    └── download_data.py<- downloads the IP SA PU datasets
