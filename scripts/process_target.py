import sys
import os
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scripts.cv_and_evaluation import preprocess_per_instance, create_cv_folds, append_result, evaluate_and_append_models, pinball_loss, get_grid, train_and_evaluate_model
# Standard library imports
import os
import random
import warnings
import logging

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import gdown
import rpy2.robjects as ro
from rpy2.rinterface import RRuntimeWarning
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from pulp import LpSolverDefault
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
# Custom or external package imports
from ddop2.newsvendor import (
    DecisionTreeWeightedNewsvendor, KNeighborsWeightedNewsvendor, 
    SampleAverageApproximationNewsvendor, DeepLearningNewsvendor, 
    RandomForestWeightedNewsvendor, GaussianWeightedNewsvendor, 
    LinearRegressionNewsvendor
)

from dddex.levelSetKDEx_univariate import LevelSetKDEx
from dddex.loadData import loadDataYaz
from dddex.crossValidation import QuantileCrossValidation, groupedTimeSeriesSplit
from joblib import Parallel, delayed
import pandas as pd
from threadpoolctl import threadpool_limits  # Importiere threadpool_limits
from Wrapper.wrapper import MLPRegressorWrapper, DeepLearningNewsvendorWrapper
from scripts.config import n_jobs, timeseries
import scripts.config as config  # Import the entire config module

# Add the WorkingFolder to the sys.path
sys.path.append('/root/WorkingFolder')

n_jobs = config.n_jobs
n_points = config.n_points
n_initial_points = config.n_initial_points

# Now you can import your module
from scripts.cv_and_evaluation import preprocess_per_instance, create_cv_folds, append_result, evaluate_and_append_models
# Function to process one column within a combination of cu and co
def process_column(column, cu, co, tau, y_train, X_train_features, X_test_features, y_test, random_state):
    table_rows = []
    n_jobs = config.n_jobs
    # Preprocess data for this specific column
    X_train_scaled, X_test_scaled, y_train_col, y_test_col, X_train_scaled_withID = preprocess_per_instance(
        column, X_train_features, X_test_features, y_train, y_test
    )

    create_cv_folds(X_train_scaled_withID)

    saa_model = SampleAverageApproximationNewsvendor(cu, co)
    saa_pred = saa_model.fit(y_train_col).predict(X_test_scaled.shape[0])

    saa_pinball_loss = pinball_loss(y_test_col.values.flatten(), saa_pred, tau)
    append_result(table_rows, column, cu, co, 'SAA', saa_pinball_loss, 'N/A', np.nan, tau)

    n_features = X_train_scaled.shape[1]

    other_models = [
        ('MLP', MLPRegressorWrapper(random_state=random_state, early_stopping=True), get_grid('MLP', n_features)),
        ('LGBM', LGBMRegressor(random_state=random_state, n_jobs=1, verbosity=-1), get_grid('LGBM', n_features)),
        ('RFW', RandomForestWeightedNewsvendor(random_state=random_state, cu=cu, co=co, n_jobs=1), get_grid('RFW', n_features)),
        ('KNNW', KNeighborsWeightedNewsvendor(cu=cu, co=co, n_jobs=1), get_grid('KNNW', n_features)),
        ('DTW', DecisionTreeWeightedNewsvendor(cu=cu, co=co, criterion='squared_error', random_state=random_state), get_grid('DTW', n_features)),
        ('GKW', GaussianWeightedNewsvendor(cu=cu, co=co), get_grid('GKW', n_features)),
    ]

    # Begrenze die Anzahl der Threads für numerische Bibliotheken
    with threadpool_limits(limits=1):
        # Loop over each model and print the name before running it
        for model_name, model, param_grid in other_models:
            print(f"Running model {model_name} for column {column}, cu={cu}, co={co}")
            # Evaluate and append the model
            evaluate_and_append_models([(model_name, model, param_grid)], X_train_scaled, X_test_scaled, y_train_col, y_test_col, saa_pinball_loss, tau, cu, co, column, table_rows, timeseries)

    result_table = pd.DataFrame(
        table_rows,
        columns=['Variable', 'cu', 'co', 'Model', 'Pinball Loss', 'Best Params', 'delta C', 'sl'])

    print(result_table.tail(7))
    return table_rows