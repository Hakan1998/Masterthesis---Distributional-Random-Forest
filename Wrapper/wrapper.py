from scripts.shared_imports import *

############################

# # The DRFWrapper class is a wrapper for the DRF model, which allows it to be used in Scikit-learn pipelines and GridSearchCV.
# Since the DRF is not compatible with Scikit-learn, we need to create a wrapper class to use it in our Bayesian optimization framework.
# Further: DRF doesnt run with n_jobs, so we need to set the number of threads in the model itself!

# important:
#  
# the number of threads should be set to the number of threads should be uses similiar to the n_jobs parameter in GridSearchCV etc.
# if dont set and pass the num_threads argument in the initialisating it will use the default value num_thread = NULL 
# which means it will use all available threads and all available kernels

# The python package doesnt contain the passed arguments
# for an overview of all arguments see the source R Package: https://github.com/lorismichel/drf/blob/master/r-package/drf/R/drf.R
############################

class DRFWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, min_node_size=10, num_trees=100, num_features=None, splitting_rule="FourierMMD", seed=42, num_threads=1):
        self.min_node_size = min_node_size
        self.num_trees = num_trees
        self.num_features = num_features  # Replace mtry with num_features
        self.splitting_rule = splitting_rule
        self.seed = seed
        self.num_threads = num_threads  # Number of threads for parallel processing
        self.model = None

    def fit(self, X, y):
        # Ensure X and y are in compatible formats
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"
        
        # Pass the num_threads argument to the model if applicable
        self.model = drf(
            min_node_size=self.min_node_size, 
            num_trees=self.num_trees, 
            num_features=self.num_features,  # Corrected syntax here
            splitting_rule=self.splitting_rule, 
            seed=self.seed,
            num_threads=self.num_threads  # Use the number of threads if specified
        )
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X, quantile=0.9):  # Default quantile 0.9 if none is provided
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        predictions = self.model.predict(X, functional="quantile", quantiles=[quantile])
        return predictions.quantile[:, :, 0]  # Return predicted quantile values
    
    def get_params(self, deep=True):
        return {
            'min_node_size': self.min_node_size, 
            'num_trees': self.num_trees, 
            'num_features': self.num_features,  # Return num.features in get_params
            'splitting_rule': self.splitting_rule, 
            'seed': self.seed,
            'num_threads': self.num_threads  # Return num_threads in get_params
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    



    # BayesSearchCV requires parameters to be Categorical, Integer, or Real types,
    # typically defined by bounds or a list of values
    # However, hidden_layer_sizes must be an array, which is not directly supported.

class MLPRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, layer1=50, layer2=25, layer3=10, activation='relu',
                 solver='adam', alpha=0.0001, learning_rate_init=0.001,
                 max_iter=200, early_stopping=False, validation_fraction=0.1,
                 random_state=None):
        # Initialize all hyperparameters including early_stopping and validation_fraction
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X, y):
        # Construct hidden_layer_sizes tuple based on layer parameters
        hidden_layers = [self.layer1]
        if self.layer2 > 0:
            hidden_layers.append(self.layer2)
        if self.layer3 > 0:
            hidden_layers.append(self.layer3)
        hidden_layer_sizes = tuple(hidden_layers)

        # Initialize and fit the MLPRegressor with early_stopping and validation_fraction
        self.model_ = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        return self.model_.score(X, y)



