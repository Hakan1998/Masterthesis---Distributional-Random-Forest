from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from dddex.crossValidation import QuantileCrossValidation, groupedTimeSeriesSplit
import scripts.config as config
from Wrapper.wrapper import MLPRegressorWrapper, DeepLearningNewsvendorWrapper
import scripts.globals as globals  # Import the globals module

bin_sizes = config.bin_sizes 
n_splits = config.n_splits

n_jobs = config.n_jobs
n_iter = config.n_iter
n_points = config.n_points
n_initial_points = config.n_initial_points



# Pinball loss function for quantile regression
def pinball_loss(y_true, y_pred, tau):
    y_true = y_true.flatten() if y_true.ndim > 1 else y_true
    y_pred = y_pred.flatten() if y_pred.ndim > 1 else y_pred
    loss = (y_true - y_pred) * ((y_true > y_pred) * tau - (y_true <= y_pred) * (1 - tau))
    return loss.mean()

# Scorer for GridSearchCV using pinball loss
def pinball_loss_scorer(tau):
    return make_scorer(lambda y_true, y_pred: pinball_loss(y_true, y_pred, tau), greater_is_better=False)




def get_grid(estimator_name, n_features):

    bin_sizes = config.bin_sizes

    kernel_bandwidth_values = np.arange(1.0, np.sqrt(n_features / 2) + 0.5, 0.5).tolist()

    layer1_values = [
        int(0.5 * n_features),
        int(1 * n_features),
        int(2 * n_features),
        int(3 * n_features)
    ]
    
    layer2_values = [
        int(0.25 * n_features),
        int(0.5 * n_features),
        int(1 * n_features),
        int(2 * n_features)
    ]

    grids = {

        "DTW": {
            "max_depth": Categorical([None, 2, 4, 6, 8, 10]),  # Diskrete Werte bleiben unverändert
            "min_samples_split": Categorical([2, 4, 8, 16, 32, 64])  # Diskrete Werte mit kleineren Abstufungen
        },
        "RFW": {
            "max_depth": Categorical([None, 2, 4, 8, 10]),  # Diskrete Werte bleiben unverändert
            "min_samples_split": Categorical ([2, 4 ,8, 16, 32, 64]),  # Diskrete Werte mit feineren Abstufungen
            "n_estimators": Categorical([10, 20, 50, 100]),  # Diskrete Werte für n_estimators mit kleineren Schritten
            "max_features": Categorical([None, 'sqrt'])  # Diskrete Wahl bleibt unverändert
        },
        "KNNW": {
            "n_neighbors": Categorical([1, 2, 4, 8, 16, 32, 64, 128])  # Kontinuierlicher Bereich für Anzahl der Nachbarn
        },
        "GKW": {
            "kernel_bandwidth": Categorical(kernel_bandwidth_values)  # Kontinuierlicher Bereich für Kernel-Bandbreite
        },
        "MLP": {    
            'layer1': Categorical(layer1_values),  # Kontinuierlicher Bereich für Layergrößen
            'layer2': Categorical(layer2_values),  # Kontinuierlicher Bereich für Layergrößen
            'solver': Categorical(['adam']),  # Solver bleibt diskret
            'alpha': Categorical([0.0001, 0.001]),  # Kontinuierlicher Bereich für alpha
            'learning_rate_init': Categorical([0.0005, 0.001]),  # Lernrate als kontinuierlicher Bereich
            'max_iter': Categorical([500, 1000]),  # Diskrete Werte für max_iter
            'early_stopping': Categorical([True])  # Diskret, da True oder False
        },
        
        "DRF": {
            "min_node_size": Categorical([1, 2, 4, 8, 16, 32, 64, 128]),  # Kontinuierlicher Bereich für min_node_size
            "num_trees": Categorical([50, 100, 250, 500]),  # Kontinuierlicher Bereich für Anzahl der Bäume
            "num_features": Categorical([5, 15, 30, 50])  # Kontinuierlicher Bereich für num_features
        },
        "LevelSetKDEx_groupsplit": {"binSize": bin_sizes, "weightsByDistance": [True, False]},
        "LGBM": {
            'num_leaves': Categorical([31, 63, 127]),  # Diskrete Werte mit kleineren Schritten für num_leaves
            'min_data_in_leaf': Categorical([20, 50, 100, 500]),  # Diskrete Werte für min_data_in_leaf
            'max_depth': Categorical([3,5,7, -1]),  # Diskrete Werte für max_depth
            'learning_rate': Categorical([0.01, 0.05, 0.1]),  # Diskrete Werte für learning_rate
            'n_estimators': Categorical([100, 200, 500])  # Diskrete Werte für n_estimators
        }
    }

    return grids.get(estimator_name, None)



# Modify the train_and_evaluate_model function
def train_and_evaluate_model(model_name, model, param_grid, X_train_scaled, X_test_scaled, 
                             y_train, y_test, tau, cu, co, timeseries, column):
    
    
    
    if timeseries and ("LS_KDEx_MLP" in model_name or "LS_KDEx_LGBM" in model_name):

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

        # Step 2: Continue with LevelSetKDEx time series cross-validation
        print("Performing LevelSetKDEx TimeSeries cross-validation...")
        # Perform time series split using groupedTimeSeriesSplit

        # Perform cross-validation for LevelSetKDEx
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

            # Convert fold_scores to DataFrame and append to global results
            fold_scores_df = pd.DataFrame(fold_scores)

            globals.global_fold_scores.append(fold_scores_df)
         
        # Get best estimator and make predictions
        best_model = CV.bestEstimator_perProb[tau]
        predictions = best_model.predict(X_test_scaled, probs=[tau])[tau]
        best_params = CV.bestParams_perProb[tau]


    else:
        # Perform Bayesian search for all models since param_grid is always available
        model, best_params = bayesian_search_model(model_name, model, param_grid, X_train_scaled, y_train, tau, cu, co, n_points, n_initial_points, n_jobs, column)

        # Conditionally pass 'quantile' only for DRF
        if model_name == 'DRF':
            predictions = model.predict(X_test_scaled, quantile=tau).flatten()
        else:
            predictions = model.predict(X_test_scaled).flatten()

        # Calculate pinball loss
    pinball_loss_value = pinball_loss(y_test.values.flatten(), predictions, tau)
    
    return pinball_loss_value, best_params




from itertools import product

# Function to calculate the number of hyperparameter combinations
def calculate_n_iter(param_grid):
    param_values = []
    # Convert Categorical objects to lists
    for key, value in param_grid.items():
        if isinstance(value, Categorical):
            param_values.append(value.categories)  # Use .categories to get the list of values from Categorical
        else:
            param_values.append(value)
    
    # Calculate the total number of combinations in the grid
    total_combinations = len(list(product(*param_values)))
    return total_combinations


def bayesian_search_model(model_name, model, param_grid, X_train, y_train, tau, cu, co, n_points, n_initial_points, n_jobs, column):


    scorer = pinball_loss_scorer(tau)

    max_combinations = calculate_n_iter(param_grid)

    if model_name == "LGBM":
        n_iter = 80  # Higher number of iterations for LGBM
    else: 
        n_iter = min(max_combinations, 50)  # Dynamically set n_iter to the smaller of max_combinations or 50

    # Create Bayesian search object with optimized settings
    bayes_search = BayesSearchCV(
        estimator=model,
        random_state=42,
        search_spaces=param_grid,
        n_iter=n_iter,  # Number of iterations
        cv=cvFolds,  # Cross-validation folds
        n_jobs=n_jobs,  # Use all available CPU cores
        n_points=n_points,  # Number of hyperparameter sets to evaluate in parallel
        scoring=scorer,
        verbose=1,
        iid = False,
        optimizer_kwargs={
            'n_initial_points': n_initial_points  # Number of initial random points
        }
    )
    
    # Fit the model with Bayesian optimization
    bayes_search.fit(X_train, y_train)
    
    best_model = bayes_search.best_estimator_

    # Only set cu and co for models that accept these parameters
    if hasattr(best_model, 'cu') and hasattr(best_model, 'co'):
        best_model.set_params(cu=cu, co=co)
    
    best_model.fit(X_train, y_train)

    # Extract cv_results_ and append to global list
    cv_results = bayes_search.cv_results_
    cv_results_df = pd.DataFrame(cv_results)

    # Add metadata for identification
    cv_results_df['model_name'] = model_name
    cv_results_df['cu'] = cu
    cv_results_df['co'] = co
    cv_results_df['tau'] = tau
    cv_results_df['variable'] = column


    print(f"Cross-validation results for {model_name} with cu={cu}, co={co}:")

    # Append the results to the global list for further aggregation
    if model_name == 'DRF':
        globals.drf_cv_results.append(cv_results_df)
    else:
        globals.global_cv_results.append(cv_results_df)

    return best_model, bayes_search.best_params_


    
    # Define the preprocessing function
def preprocess_per_instance(column, X_train_features, X_test_features, y_train, y_test):
    """Preprocess training and test data for the given column."""
    # Drop irrelevant columns
    drop_columns = ['label', 'id', 'demand', 'dayIndex']

    # Check if 'scalingValue' exists in the DataFrame for this column and add it to drop_columns if it exists
    if 'scalingValue' in X_train_features.get_group(column).columns:
        drop_columns.append('scalingValue')

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



    # Retain the id and dayIndex columns for later
    X_train_scaled_withID = pd.DataFrame(
        np.hstack((X_train_scaled, X_train_features.get_group(column)[['id', 'dayIndex']].values)),
        columns=list(X_train.columns) + ['id', 'dayIndex'])
    
    
    return X_train_scaled, X_test_scaled, y_train_col, y_test_col, X_train_scaled_withID

def append_result(table_rows, column, cu, co, model_name, pinball_loss_value, best_params, delta, tau):
    """Append model evaluation results to the table_rows and print the result."""
    result_row = [column, cu, co, model_name, pinball_loss_value, best_params, delta, tau]
    table_rows.append(result_row)
    # Print the result row after appending


def evaluate_and_append_models(models, X_train_scaled, X_test_scaled, y_train_col, y_test_col, 
                               saa_pinball_loss, tau, cu, co, column, table_rows, timeseries):
    """Evaluate models, calculate pinball loss, and append results to table_rows, with debug prints."""
    for model_name, model, param_grid in models:
        # Check if model is LevelSetKDEx_DL to compare different versions
        print(f"Evaluating model: {model_name}, cu: {cu}, co: {co}")
        if ("LevelSetKDEx_DL" in model_name or "LevelSetKDEx_RF" in model_name) and get_grid("LevelSetKDEx_manual", X_train_scaled.shape[1]):
            # Evaluate models for time series datasets
            pinball_loss_value, best_params = train_and_evaluate_model(
                model_name, model, param_grid, X_train_scaled, X_test_scaled, 
                y_train_col, y_test_col, tau, cu, co, timeseries, column
            )
            # Calculate delta (improvement) compared to SAA
            delta = 1 - (pinball_loss_value / saa_pinball_loss)
            # Append result to table_rows
            append_result(table_rows, column, cu, co, model_name, pinball_loss_value, best_params, delta, tau)

        else:
            # Evaluate models for time series datasets
            pinball_loss_value, best_params = train_and_evaluate_model(
                model_name, model, param_grid, X_train_scaled, X_test_scaled, 
                y_train_col, y_test_col, tau, cu, co, timeseries, column
            )
            # Calculate delta (improvement) compared to SAA
            delta = 1 - (pinball_loss_value / saa_pinball_loss)
            # Append result to table_rows
            append_result(table_rows, column, cu, co, model_name, pinball_loss_value, best_params, delta, tau)

def create_cv_folds(X_train_scaled_withID, kFolds=5, testLength=None, groupFeature='id', timeFeature='dayIndex'):
    global cvFolds
    if testLength is None:
        testLength = int(0.2 * (len(X_train_scaled_withID)/kFolds ))
    print(f"Test length for column: {testLength} (20% of {len(X_train_scaled_withID)/kFolds})")
    cvFolds = groupedTimeSeriesSplit(
        data=X_train_scaled_withID, 
        kFolds=kFolds, 
        testLength=testLength, 
        groupFeature=groupFeature, 
        timeFeature=timeFeature
    )

