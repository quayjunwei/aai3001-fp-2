# aai3001_fp_2

This repository contains the AAI3001 project focused on dataset collection, model development, and deployment. It features a user-friendly interface and emphasises scalability for future enhancements. The project is fully replicable for others to reproduce the results. The app is successfully deployed and is accessible at the following link: https://jjaskscan.streamlit.app/


## Contributions
- Leong Tuck Ming Kenneth
- Justin Tan Yong An
- Liew Xi Jun Adrian
- Sherwyn Chan Yin Kit
- Quay Jun Wei

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
## Model Loading and Good Coding Practices

### Trained Model File Management
To run this project, the trained model file (`densenet121_epoch55.pth`) is **not included in the GitHub repository**. Instead, it is available for download from the [GitHub Releases section](https://github.com/quayjunwei/aai3001-fp-2/releases/tag/v1.0) or [Onedrive (densenet121_epoch55.pth)](https://sitsingaporetechedu-my.sharepoint.com/:f:/g/personal/2302675_sit_singaporetech_edu_sg/EndudpPedUlBlZYnWYBbiGsB2X4TxlXXXq_nJikX4AQVOw?e=avVkyn). This approach adheres to **good coding practices**, including:

- **Minimising Repository Size**: Large files like models or datasets should not bloat the repository.
- **Version Control**: Releases allow precise versioning for easier replication and debugging.

When the application is running, the script will check for the model file in the `models/` directory. If it is not found, the application will automatically download it from the specified release. 

### Good Coding Practices
The project adheres to several best practices to maintain readability, scalability, and usability:

1. **Modularization**:
   - The code is split into meaningful modules (e.g., `etl`, `modelling`, `visualisation`), ensuring clear separation of concerns.
   - Each module encapsulates specific functionality, making the project easier to maintain and extend.

2. **PEP 8 Compliance**:
   - The code follows the [PEP 8](https://peps.python.org/pep-0008/) style guide for Python, ensuring consistent and readable formatting.

3. **Docstrings**:
   - Each function and class includes a descriptive docstring, explaining its purpose, inputs, and outputs.
   - These docstrings assist developers in understanding the code without requiring external documentation.

4. **Caching**:
   - Model loading uses `st.cache_resource` in Streamlit to optimize performance by avoiding redundant downloads or initialisations.

By adhering to these principles, this project is designed to be user-friendly, replicable, and scalable, making it accessible for fellow developers.

### Runnig the app
```
streamlit run src/main.py
```

### Prerequisite to run pipeline: Dataset & CheXNet pre-trained weights 

[Download the dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data/data) and place it in `data/raw`

[Download the the pre-trained CheXNet weights (model.pth.tar)](https://sitsingaporetechedu-my.sharepoint.com/:f:/g/personal/2302675_sit_singaporetech_edu_sg/EndudpPedUlBlZYnWYBbiGsB2X4TxlXXXq_nJikX4AQVOw?e=avVkyn) and place it in `models/`


Data after ETL can also be found [here](https://sitsingaporetechedu-my.sharepoint.com/:f:/g/personal/2302675_sit_singaporetech_edu_sg/EmBJrHsqQRRNoutqXWeKZX8B-GnWXdQ4TsfdSfwK6rD9vQ?e=bAcRq7)
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

