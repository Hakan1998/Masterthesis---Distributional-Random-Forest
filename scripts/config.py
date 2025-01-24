# config.py

# ==========================
# Fixed Parameters
# ==========================
n_splits = 5                      # Number of splits for cross-validation
n_iter = 50                        # Number of iterations for model training/optimization
n_points = 10                      # Number of points for certain models
n_initial_points = 10              # Initial number of points for optimization
timeseries = True                  # Flag indicating if dataset is time series

# ==========================
# Variable Parameters
# ==========================
bin_sizes = [20, 100, 400]   # Bin sizes to use for training. Add 800 for full dataset
dataset_name = "subset_bakery"               # Name of the dataset to use

# Available dataset names: 
# for ID-based Training:
# air, bakery, m5, wage, yaz
#subset_air,, subset_bakery,subset_m5, wage, yaz
# ==========================
# Model Settings
# ==========================
levelset_calculations = True        # Whether to load previous results from LS estimator hyperparameter tuning
n_jobs = 1                          # Number of parallel jobs for computations. Adjust based on available CPU resources


