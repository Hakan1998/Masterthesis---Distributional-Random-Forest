import pandas as pd

global_cv_results = []
global_fold_scores = []
drf_cv_results = []


 # Replace this with the actual dataset name

dataset_name = "bakery" 


# file for best params of levelset estimators - LGBM and MLP Regressor

filename = f"results_basic_Models_{dataset_name}.csv"

#result_table = pd.read_csv(filename)