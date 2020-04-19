"""
Author      : Jeremy Jess
Description : This file is adapted from Prof. Wu
              in CS158
"""

# python libraries
from abc import ABC

# numpy libraries
import numpy as np

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor


######################################################################
# classes
######################################################################

class Classifier(ABC):
    """Base class for classifier with hyper-parameter optimization.
    See sklearn.model_selection._search.

    Attributes
    -------
    estimator_ : estimator object
        This is assumed to implement the scikit-learn estimator interface.

    param_grid_ : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    Parameters
    -------
    n : int
        Number of samples.

    d : int
        Number of features.
    """

    def __init__(self, n, d):
        self.estimator_ = None
        self.param_grid_ = None

class Dummy(Classifier):
    """A Dummy classifier."""

    def __init__(self,n,d):
        self.estimator_ = DummyClassifier(strategy='stratified')
        self.param_grid_ = {}

class LinearRegressor(Classifier):
    """Linear Regression Classifier"""

    def __init__(self,n,d):
        self.estimator_ = LinearRegression()
        self.param_grid_ = {}

"""
class LinearRegressionBagging(Classifier):
    Linear Regression w/ Bagging

    def __init__(self,n,d):
        self.estimator_ = BaggingRegressor(
            LinearRegression(),
            n_jobs=-1
        )
        self.param_grid_ = {
            'n_estimators':np.arange(10,500,5),
            'max_features':np.arange(1,10,1)
        }
"""

"""
class LinearRegressionBoosting(Classifier):
    Linear Regression w/ Boosting
    def __init__(self,n,d):
        self.estimator_ = AdaBoostRegressor(LinearRegression())
        self.param_grid_ = {
            'n_estimators':np.arange(20,500,5)
        }
"""

class KNN(Classifier):
    """K-Nearest Neighbor Classifier"""
    def __init__(self,n,d):
        self.estimator_ = KNeighborsRegressor()
        self.param_grid_ = {
            'n_neighbors':np.arange(1,20,1),
            'weights':['distance','uniform']
        }

class GradientBoosting(Classifier):
    """Gradient Boosting Classifier"""
    def __init__(self,n,d):
        self.estimator_ = GradientBoostingRegressor()
        self.param_grid_ = {
            'n_estimators':np.arange(10,500,5),
            'max_features':np.arange(1,10,1)
        }

class NeuralNet(Classifier):
    """Neural Net"""
    def __init__(self,n,d):
        self.estimator_ = MLPRegressor(
                            max_iter=5000,
                            learning_rate = 'adaptive',
                            solver='sgd',
        )
        self.param_grid_ = {
            'hidden_layer_sizes':[(x,) for x in np.arange(1,50,1)],
            'activation':['logistic','relu']
        }

"""
class LinearSVM(Classifier):
    A SVM classifier.

    def __init__(self, n, d):
        self.estimator_ = SVC(kernel='linear', class_weight='balanced')
        self.param_grid_ = {'C': np.logspace(-3, 3, 7)}


class RbfSVM(Classifier):
    A SVM classifier.

    def __init__(self, n, d):
        self.estimator_ = SVC(kernel='rbf', class_weight='balanced',
                              tol=1e-3, max_iter=1e6)
        self.param_grid_ = {'gamma': np.logspace(-3, 3, 7), 'C': np.logspace(-3, 3, 7)}
"""

######################################################################
# globals
######################################################################

CLASSIFIERS = [c.__name__ for c in Classifier.__subclasses__()]
