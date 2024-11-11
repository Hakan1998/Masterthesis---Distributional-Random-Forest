# process_target.py

from scripts.shared_imports import *
from scripts.utils import *

import scripts.globals as globals  # Import the globals module
from scripts.config import *

from scripts.train_and_evaluate_singleID import preprocess_per_instance_singleID, create_cv_folds_singleID, evaluate_and_append_models_singleID
from scripts.train_and_evaluate_alldata import preprocess_per_instance_alldata, create_cv_folds_alldata,evaluate_and_append_models_alldata


# Function to process one column within a combination of cu and co
def process_target_singleID(column, cu, co, tau, y_train, X_train_features, X_test_features, y_test, random_state):
    table_rows = []

    # Preprocess data for this specific column
    X_train_scaled, X_test_scaled, y_train_col, y_test_col, X_train_scaled_withID = preprocess_per_instance_singleID(
        column, X_train_features, X_test_features, y_train, y_test
    )

    create_cv_folds_singleID(X_train_scaled_withID)

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
            evaluate_and_append_models_singleID([(model_name, model, param_grid)], X_train_scaled, X_test_scaled, y_train_col, y_test_col, saa_pinball_loss, tau, cu, co, column, table_rows, timeseries)

    result_table = pd.DataFrame(
        table_rows,
        columns=['Variable', 'cu', 'co', 'Model', 'Pinball Loss', 'Best Params', 'delta C', 'sl'])

    print(result_table.tail(7))
    return table_rows

def process_target_alldata(column, cu, co, tau, y_train, X_train_features, X_test_features, y_test, random_state):
    table_rows = []
    global cvFolds_FULLDATA
    # Preprocess data for this specific column
    X_train_scaled, X_test_scaled, y_train_col, y_test_col, X_train_scaled_withID, X_test_scaled_withID = preprocess_per_instance_alldata(
        column, X_train_features, X_test_features, y_train, y_test
    )

    create_cv_folds_alldata(X_train_scaled_withID)

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
            evaluate_and_append_models_alldata([(model_name, model, param_grid)], X_train_scaled, X_test_scaled,
                                       y_train_col, y_test_col, saa_pinball_losses_per_id,
                                       tau, cu, co, column, table_rows, timeseries, X_test_scaled_withID)

    result_table = pd.DataFrame(
        table_rows,
        columns=['Variable', 'cu', 'co', 'Model', 'Pinball Loss', 'Best Params', 'delta C', 'sl']
    )

    print(result_table.tail(7))
    return table_rows