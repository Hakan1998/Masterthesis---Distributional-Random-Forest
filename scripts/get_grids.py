from skopt.space import Real, Integer, Categorical
import math
import numpy as np
import sys
import os
import scripts.config as config

def get_grid(estimator_name, n_features):

    kernel_bandwidth_values = np.arange(1.0, np.sqrt(n_features / 2) + 0.5, 0.5).tolist()

    sqrt_n_features = math.sqrt(n_features)

    bin_sizes = config.bin_sizes 

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
            "num_features": Categorical([n_features, int(sqrt_n_features)])  # Quadratwurzel von n_features als Option
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