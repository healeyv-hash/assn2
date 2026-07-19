# Assignment 2
Assignment 2 for MLOps - Tori Healey

## Repo structure
├── src/
│   ├── init.py
│   └── pipeline_helper_functions.py   # helper functions for main file (feature cleaning, model training, etc)
├── notebooks/
│   └── Assignment 2 Healey.py                 # runs the full pipeline
├── requirements.txt
└── README.md

## Prerequisites
- A Databricks workspace with Unity Catalog enabled
- See requirements.txt file for specific dependencies

## Setup

1. Clone the repo into DataBricks
2. Upload the athletes.csv data set to DataBricks. The file path you upload it to should take this sort of form: /Volumes/<CATALOG>/default/<SCHEMA>/athletes.csv
3. Configure paths: open /notebooks/Assignment 2 Healey.ipynb and upload the configuration cell to reflect your file paths and MLflow experiment name
4. Install dependencies: the notebook's first cell installs everything from requirements.txt

## Running the pipeline
Run `notebooks/Assignment 2 Healey.ipynb` top to bottom. It will:
1. Load `athletes.csv`
2. Perform an EDA (including tables and graphs)
3. Preprocess the data
4. Build two feature versions and register them as Unity Catalog feature tables
5. Train XGBoost across across 2 feature versions and 2 hyperparameter configurations
9. Log all runs, parameterss, and models to MLflow


