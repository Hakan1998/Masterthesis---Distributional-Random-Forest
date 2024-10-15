from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from drf import drf
from ddop2.newsvendor import (
    DeepLearningNewsvendor
)

class DRFWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, min_node_size=10, num_trees=100, num_features=None, splitting_rule="FourierMMD", seed=42):
        self.min_node_size = min_node_size
        self.num_trees = num_trees
        self.num_features = num_features  # Replace mtry with num_features
        self.splitting_rule = splitting_rule
        self.seed = seed
        self.model = None

    def fit(self, X, y):
        # Ensure X and y are in compatible formats
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"
        self.model = drf(
            min_node_size=self.min_node_size, 
            num_trees=self.num_trees, 
            num_features=self.num_features,  # Corrected syntax here
            splitting_rule=self.splitting_rule, 
            seed=self.seed
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
            'seed': self.seed
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

from sklearn.base import BaseEstimator, RegressorMixin
# Import your DeepLearningNewsvendor class from the appropriate module
# from ddop2.newsvendor import DeepLearningNewsvendor

class DeepLearningNewsvendorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, layer1=50, layer2=25, layer3=10, activations=[ 'ReLU', 'ReLU', 'ReLU'], optimizer='adam',
                 epochs=100, batch_size=32, cu=1.0, co=1.0, random_state=None):
        # Initialize all hyperparameters including cost parameters cu and co
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.activations = activations
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.cu = cu  # Underage cost
        self.co = co  # Overage cost
        self.random_state = random_state

    def fit(self, X, y):
        # Build the list of neurons per layer based on layer parameters
        layers = []
        if self.layer1 > 0:
            layers.append(self.layer1)
        if self.layer2 > 0:
            layers.append(self.layer2)
        if self.layer3 > 0:
            layers.append(self.layer3)
        neurons = tuple(layers)

        # Initialize the DeepLearningNewsvendor model with the specified parameters
        self.model_ = DeepLearningNewsvendor(
            cu=self.cu,
            co=self.co,
            neurons=neurons,
            activations=self.activations,
            optimizer=self.optimizer,
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_state=self.random_state
        )
        # Fit the model to the data
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        # Use the trained model to make predictions
        return self.model_.predict(X)

    def get_params(self, deep=True):
        # Return a dictionary of the model's parameters
        return {
            'layer1': self.layer1,
            'layer2': self.layer2,
            'layer3': self.layer3,
            'activations': self.activations,
            'optimizer': self.optimizer,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'cu': self.cu,
            'co': self.co,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        # Set the model's parameters
        for key, value in params.items():
            setattr(self, key, value)
        return self



class LevelSetKDExWrapper(BaseEstimator):
    """
    This wrapper is needed to make the LevelSetKDEx estimator compatible with GridSearchCV, especially when using 
    non-Scikit-learn models like DeepLearningNewsvendor (built with Keras). 

    Why the wrapper is necessary:
    -----------------------------
    Keras models, such as DeepLearningNewsvendor, aren't directly compatible with Scikit-learn's cloning mechanism, 
    which is required for GridSearchCV. Without this wrapper, models with custom loss functions or stateful components 
    could fail during grid search or deep copying.

    The wrapper ensures:
    --------------------
    - The estimator (like DeepLearningNewsvendor) is passed immutably to GridSearchCV.
    - Safe handling of cloning, even for models that involve custom components like Keras-based models.

    Why RandomForest doesn't need a wrapper:
    ----------------------------------------
    RandomForestRegressor, being a native Scikit-learn model, is fully compatible with GridSearchCV and doesn't require 
    extra handling since it adheres to Scikit-learn's design for cloning.

    """

    def __init__(self, estimator, binSize=100, weightsByDistance=False):
        self.estimator = estimator
        self.binSize = binSize
        self.weightsByDistance = weightsByDistance

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, probs=None):
        if probs is not None:
            return self.estimator.predict(X, probs=probs)
        return self.estimator.predict(X)

    def get_params(self, deep=True):
        return {
            'estimator': self.estimator,
            'binSize': self.binSize,
            'weightsByDistance': self.weightsByDistance
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


