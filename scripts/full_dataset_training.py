# full_dataset_training.py

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV
from joblib import parallel
from sklearn.preprocessing import StandardScaler
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
)
from drf import drf
from dddex.levelSetKDEx_univariate import LevelSetKDEx
from dddex.crossValidation import QuantileCrossValidation, groupedTimeSeriesSplit
from joblib import Parallel, delayed
import pandas as pd
from threadpoolctl import threadpool_limits  # Importiere threadpool_limits

from Wrapper.wrapper import DeepLearningNewsvendorWrapper, MLPRegressorWrapper
from scripts.cv_and_evaluation import pinball_loss, pinball_loss_scorer, get_pre_tuned_params, bayesian_search_model
from scripts.get_grids import get_grid

import scripts.globals as globals
from scripts.config import *
from scripts.get_data import create_cv_folds


# Anfang der Datei
cvFolds = None  # Globale Initialisierung

def preprocess_per_instance(column, X_train_features, X_test_features, y_train, y_test):
    """Preprocess training and test data for the given column."""
    # Drop irrelevant columns
    drop_columns = ['label', 'id_for_CV', 'demand', 'dayIndex', 'dummyID']

    # Now safely drop the columns (without errors)
    X_train = X_train_features.get_group(column).drop(columns=drop_columns, errors='ignore')
    X_test = X_test_features.get_group(column).drop(columns=drop_columns, errors='ignore')

    # Extract corresponding target data
    y_train_col, y_test_col = y_train[column], y_test[column]
    y_train_col = y_train_col.dropna()
    y_test_col = y_test_col.dropna()

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Retain the id and dayIndex columns for later (for both train and test)
    X_train_scaled_withID = pd.DataFrame(
        np.hstack((X_train_scaled, X_train_features.get_group(column)[['id_for_CV', 'dayIndex']].values)),
        columns=list(X_train.columns) + ['id_for_CV', 'dayIndex'])

    X_test_scaled_withID = pd.DataFrame(
        np.hstack((X_test_scaled, X_test_features.get_group(column)[['id_for_CV', 'dayIndex']].values)),
        columns=list(X_test.columns) + ['id_for_CV', 'dayIndex'])

    return X_train_scaled, X_test_scaled, y_train_col, y_test_col, X_train_scaled_withID, X_test_scaled_withID


def append_result(table_rows, variable, cu, co, model_name, pinball_loss_value, best_params, delta, tau):
    """Append model evaluation results to the table_rows."""
    result_row = [variable, cu, co, model_name, pinball_loss_value, best_params, delta, tau]
    table_rows.append(result_row)





def evaluate_and_append_models(models, X_train_scaled, X_test_scaled, y_train_col, y_test_col,
                               saa_pinball_losses_per_id, tau, cu, co, column, table_rows, timeseries, X_test_scaled_withID):
    
    global cvFolds
    for model_name, model, param_grid in models:
        print(f"Evaluating model: {model_name}, cu: {cu}, co: {co}")
        pinball_loss_value, best_params, predictions = train_and_evaluate_model(
            model_name, model, param_grid, X_train_scaled, X_test_scaled,
            y_train_col, y_test_col, tau, cu, co, timeseries, column
        )

        # Create DataFrame for model predictions
        predictions_df = pd.DataFrame({
            'id_for_CV': X_test_scaled_withID['id_for_CV'].values,
            'y_true': y_test_col.values,
            'y_pred': predictions
        })

        # Compute pinball loss per ID
        grouped_predictions = predictions_df.groupby('id_for_CV')
        for id_val, group in grouped_predictions:
            y_true_id = group['y_true'].values
            y_pred_id = group['y_pred'].values
            pinball_loss_id = pinball_loss(y_true_id, y_pred_id, tau)
            # Calculate delta per ID
            if id_val in saa_pinball_losses_per_id:
                delta_id = 1 - (pinball_loss_id / saa_pinball_losses_per_id[id_val])
            else:
                delta_id = np.nan
            # Append result per ID
            append_result(table_rows, id_val, cu, co, model_name, pinball_loss_id, best_params, delta_id, tau)

def train_and_evaluate_model(model_name, model, param_grid, X_train_scaled, X_test_scaled,
                             y_train, y_test, tau, cu, co, timeseries, column):
    if timeseries and ("LS_KDEx_MLP" in model_name or "LS_KDEx_LGBM" in model_name):
        # Perform LevelSetKDEx time series cross-validation
        point_forecast_model = model.estimator
        if isinstance(point_forecast_model, LGBMRegressor):
            lgbm_params = get_pre_tuned_params(column, cu, co, 'LGBM')
            lgbm_params['n_jobs'] = n_jobs  # Set n_jobs for LightGBM
            point_forecast_model.set_params(**lgbm_params)
        elif isinstance(point_forecast_model, MLPRegressor):
            mlp_params = get_pre_tuned_params(column, cu, co, 'MLP')
            point_forecast_model.set_params(**mlp_params)

        point_forecast_model.fit(X_train_scaled, y_train)
        model.estimator = point_forecast_model

        CV = QuantileCrossValidation(estimator=model, parameterGrid=param_grid, cvFolds=cvFolds,
                                     probs=[tau], refitPerProb=True, n_jobs=n_jobs)
        CV.fit(X=X_train_scaled, y=y_train.to_numpy())

        fold_scores_raw = CV.cvResults_raw
        for fold_idx, fold_scores in enumerate(fold_scores_raw):
            fold_scores['fold'] = fold_idx
            fold_scores['model_name'] = model_name
            fold_scores['cu'] = cu
            fold_scores['co'] = co
            fold_scores['tau'] = tau
            fold_scores['variable'] = column
            fold_scores['dataset_name'] = dataset_name

            fold_scores_df = pd.DataFrame(fold_scores)
            global_fold_scores.append(fold_scores_df)

        best_model = CV.bestEstimator_perProb[tau]
        predictions = best_model.predict(X_test_scaled, probs=[tau])[tau]
        best_params = CV.bestParams_perProb[tau]
    else:
        # Perform Bayesian search for other models
        model, best_params = bayesian_search_model(
            model_name, model, param_grid, X_train_scaled, y_train, tau, cu, co, n_points, n_initial_points, n_jobs, column
        )
        if model_name == 'DRF':
            predictions = model.predict(X_test_scaled, quantile=tau).flatten()
        else:
            predictions = model.predict(X_test_scaled).flatten()

    # Calculate pinball loss
    pinball_loss_value = pinball_loss(y_test.values.flatten(), predictions, tau)

    return pinball_loss_value, best_params, predictions  # Return all three values


def process_column(column, cu, co, tau, y_train, X_train_features, X_test_features, y_test, random_state):
    table_rows = []
    global cvFolds
    # Preprocess data for this specific column
    X_train_scaled, X_test_scaled, y_train_col, y_test_col, X_train_scaled_withID, X_test_scaled_withID = preprocess_per_instance(
        column, X_train_features, X_test_features, y_train, y_test
    )

    create_cv_folds(X_train_scaled_withID)

    # SAA model
    saa_model = SampleAverageApproximationNewsvendor(cu, co)
    saa_pred = saa_model.fit(y_train_col).predict(X_test_scaled.shape[0])


    # Ensure id_for_CV, y_true, and y_pred are 1-D arrays
    id_for_CV = X_test_scaled_withID['id_for_CV'].values.flatten()
    y_true = y_test_col.values.flatten()
    y_pred = saa_pred.flatten()  # Flatten y_pred to ensure it's 1-D

    # Create DataFrame for SAA predictions
    saa_predictions_df = pd.DataFrame({
        'id_for_CV': id_for_CV,
        'y_true': y_true,
        'y_pred': y_pred  # Use the flattened y_pred here
    })

    saa_pinball_losses_per_id = {}
    grouped_saa = saa_predictions_df.groupby('id_for_CV')
    for id_val, group in grouped_saa:
        y_true_id = group['y_true'].values
        y_pred_id = group['y_pred'].values
        pinball_loss_id = pinball_loss(y_true_id, y_pred_id, tau)
        saa_pinball_losses_per_id[id_val] = pinball_loss_id
        append_result(table_rows, id_val, cu, co, 'SAA', pinball_loss_id, 'N/A', np.nan, tau)

    n_features = X_train_scaled.shape[1]

    other_models = [
        ('MLP', MLPRegressorWrapper(random_state=random_state, early_stopping=True), get_grid('MLP', n_features)),
        ('LGBM', LGBMRegressor(random_state=random_state, n_jobs=5, verbosity=-1), get_grid('LGBM', n_features)),
        ('RFW', RandomForestWeightedNewsvendor(random_state=random_state, cu=cu, co=co, n_jobs=32), get_grid('RFW', n_features)),
        ('KNNW', KNeighborsWeightedNewsvendor(cu=cu, co=co, n_jobs=n_jobs), get_grid('KNNW', n_features)),
        ('DTW', DecisionTreeWeightedNewsvendor(cu=cu, co=co, criterion='squared_error', random_state=random_state), get_grid('DTW', n_features)),
        ('GKW', GaussianWeightedNewsvendor(cu=cu, co=co), get_grid('GKW', n_features)),
    ]

    # Limit the number of threads for numerical libraries
    with threadpool_limits(limits=1):
        for model_name, model, param_grid in other_models:
            print(f"Running model {model_name} for column {column}, cu={cu}, co={co}")
            evaluate_and_append_models([(model_name, model, param_grid)], X_train_scaled, X_test_scaled,
                                       y_train_col, y_test_col, saa_pinball_losses_per_id,
                                       tau, cu, co, column, table_rows, timeseries, X_test_scaled_withID)

    result_table = pd.DataFrame(
        table_rows,
        columns=['Variable', 'cu', 'co', 'Model', 'Pinball Loss', 'Best Params', 'delta C', 'sl']
    )

    print(result_table.tail(7))
    return table_rows


# Funktion zur dynamischen Erstellung von Dataset-Einstellungen basierend auf den geladenen Daten
def get_dataset_settings(data):
    return {
        'bakery': {
            'file_id': '1r_bDn9Z3Q_XgeTTkJL7352nUG3jkUM0z',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': ['is_schoolholiday', 'is_holiday', 'is_holiday_next2days'],
            'drop_columns': ['date']
        },
        'yaz': {
            'file_id': '1xrY3Uv5F9F9ofgSM7dVoSK4bE0gPMg36',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': [],
            'drop_columns': []
        },
        'm5': {
            'file_id': '1tCBaxOgE5HHllvLVeRC18zvALBz6B-6w',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': [],
            'drop_columns': []
        },
        'sid': {
            'file_id': '1J9bPCfeLDH-mbSnvTHRoCva7pl6cXD3_',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': [],
            'drop_columns': []
        },
        'air': {
            'file_id': '1SKPpNxulcusNTjRwCC0p3C_XW7aNBNJZ',
            'backscaling_columns': [] ,
            'bool_columns': [],
            'drop_columns': ["counts", "location", "target"]
        },
                'copula': {
            'file_id': '1H5wdJgmxdhbzeS17w0NkRlHRCESEAd-e',
            'backscaling_columns': [] ,
            'bool_columns': [],
            'drop_columns': []
        },
                'wage': {
            'file_id': '1bn7E7NOoRzE4NwXXs1MYhRSKZHC13qYU',
            'backscaling_columns': [] ,
            'bool_columns': [],
            'drop_columns': []
        }
    }


def preprocess_data(data, demand_columns, bool_columns, drop_columns):
    # 1. Rückskalierung der 'demand_'-Spalten und der Target-Spalte 'demand'




    data = data.reset_index()
    data.drop(columns=drop_columns, inplace=True, errors='ignore')


    data['id_for_CV'] = data['id']                                        ########################
    data["dummyID"] = "dummyID"                                           #########################
    data.drop(columns=['id'], inplace=True)                               #########################

    data[bool_columns] = data[bool_columns].astype(int)


    y = data[["demand", "label", "id_for_CV"]].set_index('id_for_CV')                      ####################
    y.rename(columns={'demand': 'dummyID'}, inplace=True)                 ##################

    train_data = data[data['label'] == 'train']
    test_data = data[data['label'] == 'test']



    # 6. Aufteilen der Zielvariablen in Trainings- und Testdaten
    y_train = y[y['label'] == 'train'].drop(columns=['label'])
    y_test = y[y['label'] == 'test'].drop(columns=['label'])

    # 7. Gruppierung der Daten nach 'id' für die Trainings- und Testdatensätze
    X_train_features = train_data.groupby('dummyID')                      #####
    X_test_features = test_data.groupby('dummyID')                        #####
    


    return y, train_data, test_data, X_train_features, X_test_features, y_train, y_test