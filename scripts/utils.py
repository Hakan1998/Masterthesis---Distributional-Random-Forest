from scripts.shared_imports import *
import scripts.config as config
bin_sizes = config.bin_sizes


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

    kernel_bandwidth_values = np.arange(1.0, np.sqrt(n_features / 2) + 0.5, 0.5).tolist()

    sqrt_n_features = math.sqrt(n_features)

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
            "max_depth": Categorical([None, 2, 4, 6, 8, 10]),  
            "min_samples_split": Categorical([2, 4, 8, 16, 32, 64])  
        },
        "RFW": {
            "max_depth": Categorical([None, 2, 4, 8, 10]),  
            "min_samples_split": Categorical ([2, 4 ,8, 16, 32, 64]),  
            "n_estimators": Categorical([10, 20, 50, 100, 250, 500]),  
            "max_features": Categorical([None, 'sqrt'])  
        },
        "KNNW": {
            "n_neighbors": Categorical([1, 2, 4, 8, 16, 32, 64, 128])  
        },
        "GKW": {
            "kernel_bandwidth": Categorical(kernel_bandwidth_values)  
        },
        "MLP": {    
            'layer1': Categorical(layer1_values),  
            'layer2': Categorical(layer2_values),  
            'solver': Categorical(['adam']),  
            'alpha': Categorical([0.0001, 0.001]),  
            'learning_rate_init': Categorical([0.0005, 0.001]),  
            'max_iter': Categorical([500, 1000]),  
            'early_stopping': Categorical([True])  
        },
        "DRF": {
            "min_node_size": Categorical([2, 4 ,8, 16, 32, 64]),  
            "num_trees": Categorical([10, 20, 50, 100, 250, 500]),  
            "num_features": Categorical([n_features, int(sqrt_n_features)]) 
        },
        "LevelSetKDEx_groupsplit": {"binSize": bin_sizes, "weightsByDistance": [True, False]},
        "LGBM": {
            'num_leaves': Categorical([31, 63, 127]),  #
            'min_data_in_leaf': Categorical([20, 50, 100, 500]),  
            'max_depth': Categorical([ 2, 4, 8, 10 -1]),  
            'learning_rate': Categorical([0.01, 0.05, 0.1]),  
            'n_estimators': Categorical([10, 20, 50, 100, 250, 500])  
        }
    }

    return grids.get(estimator_name, None)

def append_result(table_rows, column, cu, co, model_name, pinball_loss_value, best_params, delta, tau):
    """Append model evaluation results to the table_rows."""
    result_row = [column, cu, co, model_name, pinball_loss_value, best_params, delta, tau]
    table_rows.append(result_row)
    

# Function to calculate the number of hyperparameter combinations
# to set dynamically the iterations for bayesopt if we have less than than our aimed iterations (50)
# otherweise models with less 50 param combinations run iteration on same params double /e.g. knnw would still run 50 iteration although it has less

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