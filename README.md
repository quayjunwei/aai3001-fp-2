# aai3001_fp_2

This repository contains the AAI3001 project focused on dataset collection, model development, and deployment. It features a user-friendly interface and emphasizes scalability for future enhancements. The project is fully replicable for others to reproduce the results.

## Getting started

### We'll be using venv as our virtual environment
```
pip install virtualenv
```

1. **Create virtual environment**

Mac
```
python3 -m venv myenv
```

Windows
```
python -m venv myenv
```

2. Activate virtual environment

Mac
```
source myenv/bin/activate
```

Windows
```
source myenv/Scripts/activate
```



3. Install required libraries & dependancies 
```
pip install -r requirements.txt
```

### STEPS TO RUN THE PROJECT

#### Load CheXNet pre-trained weights

Download the file and place it in `models/`

The pre-trained weights required to run this project can be downloaded from the following link:

[Link to pretrained weights (model.pth.tar)](https://sitsingaporetechedu-my.sharepoint.com/personal/2302675_sit_singaporetech_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F2302675%5Fsit%5Fsingaporetech%5Fedu%5Fsg%2FDocuments%2FAAI3001%20Deep%20Learning%20%26%20Computer%20Vision%2Fmodels)

#### Step 1: Run the ETL Pipeline

The ETL pipeline processes raw data, filters it based on allowed labels, organizes it into class-specific folders, and performs data augmentation for underrepresented classes.

Command:
``` 
python src/etl/etl_pipeline.py
```

#### Step 2: Split the Dataset

Split the processed dataset into training, validation, and test sets. Ensure the necessary directories for splits exist (train, val, test).

Command: 
```
python src/etl/data_splitting.py
```

#### Step 3: Train the Model

Train the DenseNet model using the prepared dataset. The script will save the trained model in the models directory.

Command: 
```
python src/modelling/train.py
```

#### Step 4: Make Predictions

Use the trained model to make predictions on the test dataset. Modify the test_dir path in the script if needed.

Command: 
```
python src/modelling/predict.py
```

## Trained model's Pickle File

Download the file and place it in `models/`
The trained model's Pickle File required to run this project can be downloaded from the following link:

[Link to pretrained weights (densenet121_epoch55.pth)](https://sitsingaporetechedu-my.sharepoint.com/personal/2302675_sit_singaporetech_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F2302675%5Fsit%5Fsingaporetech%5Fedu%5Fsg%2FDocuments%2FAAI3001%20Deep%20Learning%20%26%20Computer%20Vision%2Fmodels)

Save the file in the `models` directory before running the application.

### Runnig the app
```
streamlit run src/main.py
```
## Project Organization

```
├── .gitignore                   <- Specifies files and directories that Git should ignore.
├── README.md                    <- The top-level README for developers using this project.
├── requirements.txt             <- The list of dependencies required for the project.
│
├── data                         <- Contains various stages of the data lifecycle.
│   ├── external                 <- Data from third-party sources.
│   ├── interim                  <- Intermediate data that has been transformed.
│   ├── processed                <- Final, canonical datasets ready for modeling.
│   └── raw                      <- The original, immutable data dump.
│
├── models                       <- Trained and serialized models, model predictions, or model summaries.
│   └── .gitkeep                 <- Keeps the folder in version control even if it's empty.
│
├── notebooks                    <- Jupyter notebooks for data exploration and model training.
│
├── references                   <- Data dictionaries, manuals, and other explanatory materials.
│
├── reports                      <- Generated reports and analysis.
│   └── figures                  <- Generated graphics and figures for reporting.
│
└── src                          <- Source code for use in this project.
    ├── __init__.py              <- Makes `src` a Python module.
    ├── main.py                  <- Main script to run the project end-to-end.
    │
    ├── etl                      <- Code for the ETL (Extract, Transform, Load) process.
    │   ├── data_cleaning.py      <- Code to clean and preprocess data.
    │   ├── data_loading.py       <- Code to load raw data from sources.
    │   ├── data_splitting.py     <- Code to split data into train, validation, and test sets.
    │   ├── data_transformation.py<- Code to transform data (e.g., scaling, encoding features).
    │   └── etl_pipeline.py       <- Orchestrates the entire ETL process.
    │
    ├── modelling                <- Code related to model training and evaluation.
    │   ├── __init__.py           <- Makes `modelling` a Python module.
    │   ├── predict.py            <- Code to make predictions using trained models.
    │   └── train.py              <- Code to train machine learning models.
    │
    └── visualisation            <- Code to generate visualizations.
        ├── __init__.py           <- Makes `visualisation` a Python module.
        └── plot.py               <- Code to create plots and visualizations.

```

--------

