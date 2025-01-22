#config.py

# fix parameters

n_splits = 5
n_iter= 50
n_points = 10
n_initial_points = 10
timeseries = True

######################################################################

# variable parameter

bin_sizes = [20,100,400]    # here we add bin_sizes 800 when we train on the full dataset
dataset_name = "yaz"
levelset_calculations = True # set true to load the previous results from the estimator hyperparameter tuning
n_jobs = 1 