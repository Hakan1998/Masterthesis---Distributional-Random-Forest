# train_and_evaluation_.py

from scripts.shared_imports import *
import scripts.config as config
from scripts.config import *
from scripts.utils import *



cvFolds_FULLDATA = None  # Globale Initialisierung

def preprocess_per_instance_alldata(column, X_train_features, X_test_features, y_train, y_test):
    """Preprocess training and test data for the given column."""
    # Drop irrelevant columns
    drop_columns = ['label', 'id_for_CV', 'demand', 'dayIndex', 'dummyID']

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

    # Retain the id and dayIndex , 
    # 1. train data for CV Folds building and 
    # 2. the IDs of Test Data to predict & calculate the metrics on each ID after weve trained the models on the full datasets
    X_train_scaled_withID = pd.DataFrame(
        np.hstack((X_train_scaled, X_train_features.get_group(column)[['id_for_CV', 'dayIndex']].values)),
        columns=list(X_train.columns) + ['id_for_CV', 'dayIndex'])

    X_test_scaled_withID = pd.DataFrame(
        np.hstack((X_test_scaled, X_test_features.get_group(column)[['id_for_CV', 'dayIndex']].values)),
        columns=list(X_test.columns) + ['id_for_CV', 'dayIndex'])

    return X_train_scaled, X_test_scaled, y_train_col, y_test_col, X_train_scaled_withID, X_test_scaled_withID



def evaluate_and_append_models_alldata(models, X_train_scaled, X_test_scaled, y_train_col, y_test_col,
                               saa_pinball_losses_per_id, tau, cu, co, column, table_rows, timeseries, X_test_scaled_withID):
    
    global cvFolds_FULLDATA
    for model_name, model, param_grid in models:
        print(f"Evaluating model: {model_name}, cu: {cu}, co: {co}")
        pinball_loss_value, best_params, predictions = train_and_evaluate_model_alldata(
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

def train_and_evaluate_model_alldata(model_name, model, param_grid, X_train_scaled, X_test_scaled,
                             y_train, y_test, tau, cu, co, timeseries, column):
    if timeseries and ("LS_KDEx_MLP" in model_name or "LS_KDEx_LGBM" in model_name):
        # Perform LevelSetKDEx time series cross-validation
        point_forecast_model = model.estimator
        if isinstance(point_forecast_model, LGBMRegressor):
            lgbm_params = get_pre_tuned_params_alldata(column, cu, co, 'LGBM')
            lgbm_params['n_jobs'] = n_jobs  # Set n_jobs for LightGBM
            point_forecast_model.set_params(**lgbm_params)
        elif isinstance(point_forecast_model, MLPRegressor):
            mlp_params = get_pre_tuned_params_alldata(column, cu, co, 'MLP')
            point_forecast_model.set_params(**mlp_params)

        point_forecast_model.fit(X_train_scaled, y_train)
        model.estimator = point_forecast_model

        CV = QuantileCrossValidation(estimator=model, parameterGrid=param_grid, cvFolds=cvFolds_FULLDATA,
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
            globals.global_fold_scores.append(fold_scores_df)

        best_model = CV.bestEstimator_perProb[tau]
        predictions = best_model.predict(X_test_scaled, probs=[tau])[tau]
        best_params = CV.bestParams_perProb[tau]
    else:
        # Perform Bayesian search for other models
        model, best_params = bayesian_search_model_alldata(
            model_name, model, param_grid, X_train_scaled, y_train, tau, cu, co, n_points, n_initial_points, n_jobs, column
        )
        if model_name == 'DRF':
            predictions = model.predict(X_test_scaled, quantile=tau).flatten()
        else:
            predictions = model.predict(X_test_scaled).flatten()

    # Calculate pinball loss
    pinball_loss_value = pinball_loss(y_test.values.flatten(), predictions, tau)

    return pinball_loss_value, best_params, predictions  # Return all three values




def create_cv_folds_alldata(X_train_scaled_withID, kFolds=5, testLength=None, groupFeature='id_for_CV', timeFeature='dayIndex'):   ###########
    global cvFolds_FULLDATA

        # Prüfe, ob die Zahl 16 in der groupFeature-Spalte vorhanden ist
    if 16 in X_train_scaled_withID[groupFeature].values:
        print("Wage Dataset, no time series split using basic KFold Cross Validation")
        kf = KFold(n_splits=5)  # Anzahl der gewünschten Folds
        cvFolds_FULLDATA = list(kf.split(X_train_scaled_withID))  # list() um die Folds zu erstellen
    else:
        amount_groups = X_train_scaled_withID[groupFeature].nunique()
        datapoints_per_group = len(X_train_scaled_withID) / amount_groups

        # Wenn keine Testlänge angegeben wurde, setze sie auf 6% der Datenpunkte pro Gruppe
        if testLength is None:
            testLength = int(0.06 * datapoints_per_group)

        print(f"Test length for column: {testLength} (6% of {int(datapoints_per_group)} Datapoints per Group)")


        cvFolds_FULLDATA = groupedTimeSeriesSplit(
            data=X_train_scaled_withID,
            kFolds=kFolds,
            testLength=testLength,
            groupFeature=groupFeature,
            timeFeature=timeFeature
    )


def bayesian_search_model_alldata(model_name, model, param_grid, X_train, y_train, tau, cu, co, n_points, n_initial_points, n_jobs, column):


    scorer = pinball_loss_scorer(tau)

    max_combinations = calculate_n_iter(param_grid)

    if model_name == "LGBM":
        n_iter = 80  # Higher number of iterations for LGBM since we have more params here
    else: 
        n_iter = min(max_combinations, 50)  # Dynamically set n_iter to the smaller of max_combinations or 50


    # Create Bayesian search object with optimized settings
    bayes_search = BayesSearchCV(
        estimator=model,
        random_state=42,
        search_spaces=param_grid,
        n_iter=n_iter,  # Number of iterations
        cv=cvFolds_FULLDATA,  # Cross-validation folds
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
    try:
        bayes_search.fit(X_train, y_train)
    except Exception as e:
        print(f"Bayesian search failed: {e}")
        raise
    
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

levelset_calculations = config.levelset_calculations
if levelset_calculations == True :
# Define the folder where results are stored
    results_folder = "results"
    dataset_name = config.dataset_name  # Get the dataset name from config.py
    filename = os.path.join(results_folder, f"results_basic_Models_{dataset_name}.csv")
    result_table = pd.read_csv(filename)

# Function to fetch the pre-tuned parameters for a given model from the result table
def get_pre_tuned_params_alldata(column, cu, co, model_name):
    """Fetch the pre-tuned parameters for DLNW or RF models from the result table."""
    params_row = result_table[
       # (result_table['Variable'] == column) &
        (result_table['cu'] == cu) &
        (result_table['co'] == co) &
        (result_table['Model'] == model_name)
    ]['Best Params'].values
    if params_row.size > 0:
        return eval(params_row[0])  # Convert the string representation of the dictionary into a Python dict
    return None