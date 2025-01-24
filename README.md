# Masterthesis-DRF

## Overview
This repository contains the work conducted for my master's thesis, focusing on the **Distributional Random Forests (DRF)**. The thesis explores advanced forecasting methods, evaluates their performance, and highlights their distributional characteristics.

---

## Repository Structure

```plaintext
Masterthesis-DRF/
├── Data/               # Datasets used for experiments and analysis
├── Plots/              # Generated visualizations 
├── Results/            # Processed results and summaries
├── Wrapper/            # Wrapper scripts for the drf and the mlp
├── scripts/            # Python scripts used 
├── main.ipynb          # Jupyter Notebook merging the scripts to obtain the results.
├── requirements.txt    # List of Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignored files
```

---

## Usage

To reproduce the results or explore the analysis:

1. **Choose the dataset**: Update the dataset configuration in the appropriate `config.py` file
2. **Run the main notebook**: Open `main.ipynb` in Jupyter Notebook and execute the cells sequentially to follow the workflow and obtain the results.

---


## Scripts Overview

### General Workflow

After the initial preprocessing, we process each target variable individually, column by column. Each variable is then scaled, and the results for each target (ID) are calculated. First, the DDOP models are applied, together with point forecaster tuning for the LSx models, followed by the DRF and LSx models.

This process is executed twice: first, we train the models using only the target variable and then on the entire dataset. During this, all results are collected in global tables.

---

### The `scripts/` directory contains the following Python scripts:

- **get_data.py**: Loads and prepares the data.
- **globals.py**: Defines the global tables.
- **process_target.py**: Contains the function that defines the workflow.
- **shared_imports.py**: Includes common imports for other scripts.
- **train_and_evaluate_alldata.py**: Executes hyperparameter search and outputs the results for the entire dataset.
- **train_and_evaluate_singleID.py**: Executes hyperparameter search and outputs the results for the "singleID" target variable.
- **utils.py**: Contains utility functions used across various scripts for common tasks.

These scripts work together to create a seamless and modular workflow for the project.

---