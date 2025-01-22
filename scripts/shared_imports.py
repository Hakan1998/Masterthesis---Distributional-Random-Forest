# shared_imports.py

# Standard library imports
import os
import sys
import random
import warnings
import logging
import math
from itertools import product
from collections import OrderedDict

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import tensorflow as tf
import gdown
import rpy2.robjects as ro
from rpy2.rinterface import RRuntimeWarning
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from pulp import LpSolverDefault
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits


# Custom or external package imports
from ddop2.newsvendor import (
    DecisionTreeWeightedNewsvendor, KNeighborsWeightedNewsvendor, 
    SampleAverageApproximationNewsvendor, DeepLearningNewsvendor, 
    RandomForestWeightedNewsvendor, GaussianWeightedNewsvendor, 
    LinearRegressionNewsvendor
)
from drf import drf
from dddex.levelSetKDEx_univariate import LevelSetKDEx
from dddex.loadData import loadDataYaz
from dddex.crossValidation import QuantileCrossValidation, groupedTimeSeriesSplit

# Wrapper imports
from Wrapper.wrapper import MLPRegressorWrapper, DRFWrapper


import scripts.config as config
import scripts.globals as globals

# Environment configuration
# Set pandas display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full column width
pd.set_option('display.max_rows', 10)  # Limit the number of displayed rows
pd.set_option('display.width', 1000)  # Set high enough width to show all columns in a line

# Suppress warnings and logging
warnings.filterwarnings("ignore")  # Suppress all Python warnings
rpy2_logger.setLevel(logging.CRITICAL)  # Only show critical messages from R

# Set R options to suppress warnings and messages
ro.r('while (sink.number() > 0) sink(NULL)') 
ro.r('options(warn=-1)')  
ro.r('suppressMessages(suppressWarnings(library("drf")))')  

# Set environment variables for R libraries
os.environ['R_LIBS_USER'] = '/usr/lib/R/site-library'
os.environ['R_HOME'] = '/usr/lib/R'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set random seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.get_logger().setLevel(logging.ERROR)

# Deactivate CBC Solver output
LpSolverDefault.msg = False  # Deactivates the CBC Solver output
