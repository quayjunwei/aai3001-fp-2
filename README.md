# aai3001_fp_2

This repository contains the AAI3001 project focused on dataset collection, model development, and deployment. It features a user-friendly interface and emphasizes scalability for future enhancements. The project is fully replicable for others to reproduce the results.

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
