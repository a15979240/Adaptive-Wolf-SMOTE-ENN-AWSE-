#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 11:15:24 2018

@author: gykovacs
"""
 
# import system packages
import os
import pickle
import itertools
import logging
import re
import time
import glob
import inspect
import random
import statistics

# used to parallelize evaluation
from joblib import Parallel, delayed

# numerical methods and arrays
import numpy as np
import pandas as pd

# import packages used for the implementation of sampling methods
from sklearn.model_selection import (RepeatedStratifiedKFold, KFold,
                                     cross_val_score, StratifiedKFold)
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone, BaseEstimator, ClassifierMixin

# some statistical methods
from scipy.stats import skew
import scipy.signal as ssignal
import scipy.spatial as sspatial
import scipy.optimize as soptimize
import scipy.special as sspecial
from scipy.stats.mstats import gmean
import hdbscan as hdb

from _version import __version__

__author__ = "György Kovács"
__license__ = "MIT"
__email__ = "gyuriofkovacs@gmail.com"

# for handler in _logger.root.handlers[:]:
#    _logger.root.removeHandler(handler)

# setting the _logger format
_logger = logging.getLogger('smote_variants')
_logger.setLevel(logging.DEBUG)
_logger_ch = logging.StreamHandler()
_logger_ch.setFormatter(logging.Formatter(
    "%(asctime)s:%(levelname)s:%(message)s"))
_logger.addHandler(_logger_ch)

# exported names
__all__ = ['__author__',
           '__license__',
           '__version__',
           '__email__',
           'get_all_oversamplers',
           'get_all_noisefilters',
           'get_n_quickest_oversamplers',
           'get_all_oversamplers_multiclass',
           'get_n_quickest_oversamplers_multiclass',
           'evaluate_oversamplers',
           'read_oversampling_results',
           'model_selection',
           'cross_validate',
           'MLPClassifierWrapper',
           'OverSampling',
           'NoiseFilter',
           'TomekLinkRemoval',
           'CondensedNearestNeighbors',
           'OneSidedSelection',
           'CNNTomekLinks',
           'NeighborhoodCleaningRule',
           'EditedNearestNeighbors',
           'SMOTE',
           'SMOTE_TomekLinks',
           'SMOTE_ENN',
           'Borderline_SMOTE1',
           'Borderline_SMOTE2',
           'ADASYN',
           'AHC',
           'LLE_SMOTE',
           'distance_SMOTE',
           'SMMO',
           'polynom_fit_SMOTE',
           'Stefanowski',
           'ADOMS',
           'Safe_Level_SMOTE',
           'MSMOTE',
           'DE_oversampling',
           'SMOBD',
           'SUNDO',
           'MSYN',
           'SVM_balance',
           'TRIM_SMOTE',
           'SMOTE_RSB',
           'ProWSyn',
           'SL_graph_SMOTE',
           'NRSBoundary_SMOTE',
           'LVQ_SMOTE',
           'SOI_CJ',
           'ROSE',
           'SMOTE_OUT',
           'SMOTE_Cosine',
           'Selected_SMOTE',
           'LN_SMOTE',
           'MWMOTE',
           'PDFOS',
           'IPADE_ID',
           'RWO_sampling',
           'NEATER',
           'DEAGO',
           'Gazzah',
           'MCT',
           'ADG',
           'SMOTE_IPF',
           'KernelADASYN',
           'MOT2LD',
           'V_SYNTH',
           'OUPS',
           'SMOTE_D',
           'SMOTE_PSO',
           'CURE_SMOTE',
           'SOMO',
           'ISOMAP_Hybrid',
           'CE_SMOTE',
           'Edge_Det_SMOTE',
           'CBSO',
           'E_SMOTE',
           'DBSMOTE',
           'ASMOBD',
           'Assembled_SMOTE',
           'SDSMOTE',
           'DSMOTE',
           'G_SMOTE',
           'NT_SMOTE',
           'Lee',
           'SPY',
           'SMOTE_PSOBAT',
           'MDO',
           'Random_SMOTE',
           'ISMOTE',
           'VIS_RST',
           'GASMOTE',
           'A_SUWO',
           'SMOTE_FRST_2T',
           'AND_SMOTE',
           'NRAS',
           'AMSCO',
           'SSO',
           'NDO_sampling',
           'DSRBF',
           'Gaussian_SMOTE',
           'kmeans_SMOTE',
           'Supervised_SMOTE',
           'SN_SMOTE',
           'CCR',
           'ANS',
           'cluster_SMOTE',
           'NoSMOTE',
           'MulticlassOversampling',
           'OversamplingClassifier']


def get_all_oversamplers():
    """
    Returns all oversampling classes

    Returns:
        list(OverSampling): list of all oversampling classes

    Example::

        import smote_variants as sv

        oversamplers= sv.get_all_oversamplers()
    """

    return OverSampling.__subclasses__()


def get_n_quickest_oversamplers(n=10):
    """
    Returns the n quickest oversamplers based on testing on the datasets of
    the imbalanced_databases package.

    Args:
        n (int): number of oversamplers to return

    Returns:
        list(OverSampling): list of the n quickest oversampling classes

    Example::

        import smote_variants as sv

        oversamplers= sv.get_n_quickest_oversamplers(10)
    """

    runtimes = {'SPY': 0.11, 'OUPS': 0.16, 'SMOTE_D': 0.20, 'NT_SMOTE': 0.20,
                'Gazzah': 0.21, 'ROSE': 0.25, 'NDO_sampling': 0.27,
                'Borderline_SMOTE1': 0.28, 'SMOTE': 0.28,
                'Borderline_SMOTE2': 0.29, 'ISMOTE': 0.30, 'SMMO': 0.31,
                'SMOTE_OUT': 0.37, 'SN_SMOTE': 0.44, 'Selected_SMOTE': 0.47,
                'distance_SMOTE': 0.47, 'Gaussian_SMOTE': 0.48, 'MCT': 0.51,
                'Random_SMOTE': 0.57, 'ADASYN': 0.58, 'SL_graph_SMOTE': 0.58,
                'CURE_SMOTE': 0.59, 'ANS': 0.63, 'MSMOTE': 0.72,
                'Safe_Level_SMOTE': 0.79, 'SMOBD': 0.80, 'CBSO': 0.81,
                'Assembled_SMOTE': 0.82, 'SDSMOTE': 0.88,
                'SMOTE_TomekLinks': 0.91, 'Edge_Det_SMOTE': 0.94,
                'ProWSyn': 1.00, 'Stefanowski': 1.04, 'NRAS': 1.06,
                'AND_SMOTE': 1.13, 'DBSMOTE': 1.17, 'polynom_fit_SMOTE': 1.18,
                'ASMOBD': 1.18, 'MDO': 1.18, 'SOI_CJ': 1.24, 'LN_SMOTE': 1.26,
                'VIS_RST': 1.34, 'TRIM_SMOTE': 1.36, 'LLE_SMOTE': 1.62,
                'SMOTE_ENN': 1.86, 'SMOTE_Cosine': 2.00, 'kmeans_SMOTE': 2.43,
                'MWMOTE': 2.45, 'V_SYNTH': 2.59, 'A_SUWO': 2.81,
                'RWO_sampling': 2.91, 'SMOTE_RSB': 3.88, 'ADOMS': 3.89,
                'SMOTE_IPF': 4.10, 'Lee': 4.16, 'SMOTE_FRST_2T': 4.18,
                'cluster_SMOTE': 4.19, 'SOMO': 4.30, 'DE_oversampling': 4.67,
                'CCR': 4.72, 'NRSBoundary_SMOTE': 5.26, 'AHC': 5.27,
                'ISOMAP_Hybrid': 6.11, 'LVQ_SMOTE': 6.99, 'CE_SMOTE': 7.45,
                'MSYN': 11.92, 'PDFOS': 15.14, 'KernelADASYN': 17.87,
                'G_SMOTE': 19.23, 'E_SMOTE': 19.50, 'SVM_balance': 24.05,
                'SUNDO': 26.21, 'GASMOTE': 31.38, 'DEAGO': 33.39,
                'NEATER': 41.39, 'SMOTE_PSO': 45.12, 'IPADE_ID': 90.01,
                'DSMOTE': 146.73, 'MOT2LD': 149.42, 'Supervised_SMOTE': 195.74,
                'SSO': 215.27, 'DSRBF': 272.11, 'SMOTE_PSOBAT': 324.31,
                'ADG': 493.64, 'AMSCO': 1502.36}

    samplers = get_all_oversamplers()
    samplers = sorted(
        samplers, key=lambda x: runtimes.get(x.__name__, 1e8))

    return samplers[:n]


def get_all_oversamplers_multiclass(strategy="eq_1_vs_many_successive"):
    """
    Returns all oversampling classes which can be used with the multiclass
    strategy specified

    Args:
        strategy (str): the multiclass oversampling strategy -
                        'eq_1_vs_many_successive'/'equalize_1_vs_many'

    Returns:
        list(OverSampling): list of all oversampling classes which can be used
                            with the multiclass strategy specified

    Example::

        import smote_variants as sv

        oversamplers= sv.get_all_oversamplers_multiclass()
    """

    oversamplers = get_all_oversamplers()

    if (strategy == 'eq_1_vs_many_successive' or
            strategy == 'equalize_1_vs_many'):

        def multiclass_filter(o):
            return ((OverSampling.cat_changes_majority not in o.categories) or
                    ('proportion' in o().get_params()))

        return [o for o in oversamplers if multiclass_filter(o)]
    else:
        raise ValueError(("It is not known which oversamplers work with the"
                          " strategy %s") % strategy)


def get_n_quickest_oversamplers_multiclass(n,
                                           strategy="eq_1_vs_many_successive"):
    """
    Returns the n quickest oversamplers based on testing on the datasets of
    the imbalanced_databases package, and suitable for using the multiclass
    strategy specified.

    Args:
        n (int): number of oversamplers to return
        strategy (str): the multiclass oversampling strategy -
                        'eq_1_vs_many_successive'/'equalize_1_vs_many'

    Returns:
        list(OverSampling): list of n quickest oversampling classes which can
                    be used with the multiclass strategy specified

    Example::

        import smote_variants as sv

        oversamplers= sv.get_n_quickest_oversamplers_multiclass()
    """

    oversamplers = get_all_oversamplers()
    quickest_oversamplers = get_n_quickest_oversamplers(len(oversamplers))

    if (strategy == 'eq_1_vs_many_successive'
            or strategy == 'equalize_1_vs_many'):

        def multiclass_filter(o):
            return ((OverSampling.cat_changes_majority not in o.categories) or
                    ('proportion' in o().get_params()))

        return [o for o in quickest_oversamplers if multiclass_filter(o)][:n]
    else:
        raise ValueError("It is not known which oversamplers work with the"
                         " strategy %s" % strategy)


def get_all_noisefilters():
    """
    Returns all noise filters
    Returns:
        list(NoiseFilter): list of all noise filter classes
    """
    return NoiseFilter.__subclasses__()


def mode(data):
    values, counts = np.unique(data, return_counts=True)
    # print(counts)
    return values[np.where(counts == max(counts))[0][0]]


class StatisticsMixin:
    """
    Mixin to compute class statistics and determine minority/majority labels
    """

    def class_label_statistics(self, X, y):
        """
        determines class sizes and minority and majority labels
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        unique, counts = np.unique(y, return_counts=True)
        self.class_stats = dict(zip(unique, counts))
        # cs=self.class_stats
        # print(cs)
        self.min_label = unique[0] if counts[0] < counts[1] else unique[1]
        self.maj_label = unique[1] if counts[0] < counts[1] else unique[0]
        # shorthands
        self.min_label = self.min_label
        self.maj_label = self.maj_label

    def check_enough_min_samples_for_sampling(self, threshold=2):
        if self.class_stats[self.min_label] < threshold:
            m = ("The number of minority samples (%d) is not enough "
                 "for sampling")
            m = m % self.class_stats[self.min_label]
            _logger.warning(self.__class__.__name__ + ": " + m)
            return False
        return True


class RandomStateMixin:
    """
    Mixin to set random state
    """

    def set_random_state(self, random_state):
        """
        sets the random_state member of the object

        Args:
            random_state (int/np.random.RandomState/None): the random state
                                                                initializer
        """

        self._random_state_init = random_state

        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is np.random:
            self.random_state = random_state
        else:
            raise ValueError(
                "random state cannot be initialized by " + str(random_state))


class ParameterCheckingMixin:
    """
    Mixin to check if parameters come from a valid range
    """

    def check_in_range(self, x, name, r):
        """
        Check if parameter is in range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x < r[0] or x > r[1]:
            m = ("Value for parameter %s outside the range [%f,%f] not"
                 " allowed: %f")
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_out_range(self, x, name, r):
        """
        Check if parameter is outside of range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x >= r[0] and x <= r[1]:
            m = "Value for parameter %s in the range [%f,%f] not allowed: %f"
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal(self, x, name, val):
        """
        Check if parameter is less than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x > val:
            m = "Value for parameter %s greater than %f not allowed: %f > %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x > y:
            m = ("Value for parameter %s greater than parameter %s not"
                 " allowed: %f > %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less(self, x, name, val):
        """
        Check if parameter is less than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x >= val:
            m = ("Value for parameter %s greater than or equal to %f"
                 " not allowed: %f >= %f")
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x >= y:
            m = ("Value for parameter %s greater than or equal to parameter"
                 " %s not allowed: %f >= %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal(self, x, name, val):
        """
        Check if parameter is greater than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x < val:
            m = "Value for parameter %s less than %f is not allowed: %f < %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x < y:
            m = ("Value for parameter %s less than parameter %s is not"
                 " allowed: %f < %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater(self, x, name, val):
        """
        Check if parameter is greater than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x <= val:
            m = ("Value for parameter %s less than or equal to %f not allowed"
                 " %f < %f")
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_par(self, x, name_x, y, name_y):
        """
        Check if parameter is greater than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x <= y:
            m = ("Value for parameter %s less than or equal to parameter %s"
                 " not allowed: %f <= %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal(self, x, name, val):
        """
        Check if parameter is equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x == val:
            m = ("Value for parameter %s equal to parameter %f is not allowed:"
                 " %f == %f")
            m = m % (name, val, x, val)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x == y:
            m = ("Value for parameter %s equal to parameter %s is not "
                 "allowed: %f == %f")
            m = m % (name_x, name_y, x, y)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_isin(self, x, name, li):
        """
        Check if parameter is in list
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            li (list): list to check if parameter is in it
        Throws:
            ValueError
        """
        if x not in li:
            m = "Value for parameter %s not in list %s is not allowed: %s"
            m = m % (name, str(li), str(x))
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_n_jobs(self, x, name):
        """
        Check n_jobs parameter
        Args:
            x (int/None): number of jobs
            name (str): the parameter name
        Throws:
            ValueError
        """
        if not ((x is None)
                or (x is not None and isinstance(x, int) and not x == 0)):
            m = "Value for parameter n_jobs is not allowed: %s" % str(x)
            raise ValueError(self.__class__.__name__ + ": " + m)


class ParameterCombinationsMixin:
    """
    Mixin to generate parameter combinations
    """

    @classmethod
    def generate_parameter_combinations(cls, dictionary, raw):
        """
        Generates reasonable paramter combinations
        Args:
            dictionary (dict): dictionary of paramter ranges
            num (int): maximum number of combinations to generate
        """
        if raw:
            return dictionary
        keys = sorted(list(dictionary.keys()))
        values = [dictionary[k] for k in keys]
        combinations = [dict(zip(keys, p))
                        for p in list(itertools.product(*values))]
        return combinations


class NoiseFilter(StatisticsMixin,
                  ParameterCheckingMixin,
                  ParameterCombinationsMixin):
    """
    Parent class of noise filtering methods
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    def remove_noise(self, X, y):
        """
        Removes noise
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        pass

    def get_params(self, deep=False):
        """
        Return parameters

        Returns:
            dict: dictionary of parameters
        """

        return {}

    def set_params(self, **params):
        """
        Set parameters

        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self



class EditedNearestNeighbors(NoiseFilter):
    """
    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    }
    """

    def __init__(self, remove='both', n_jobs=1):
        """
        Constructor of the noise removal object

        Args:
            remove (str): class to remove from 'both'/'min'/'maj'
            n_jobs (int): number of parallel jobs
        """
        super().__init__()

        self.check_isin(remove, 'remove', ['both', 'min', 'maj'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.remove = remove
        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        if len(X) < 4:
            _logger.info(self.__class__.__name__ + ': ' +
                         "Not enough samples for noise removal")
            return X.copy(), y.copy()

        nn = NearestNeighbors(n_neighbors=4, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        to_remove = []
        for i in range(len(X)):
            if not y[i] == mode(y[indices[i][1:]]):
                if (self.remove == 'both' or
                    (self.remove == 'min' and y[i] == self.min_label) or
                        (self.remove == 'maj' and y[i] == self.maj_label)):
                    to_remove.append(i)

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)

    def get_params(self):
        """
        Get noise removal parameters

        Returns:
            dict: dictionary of parameters
        """
        return {'remove': self.remove}


class OverSampling(StatisticsMixin,
                   ParameterCheckingMixin,
                   ParameterCombinationsMixin,
                   RandomStateMixin):
    """
    Base class of oversampling methods
    """

    categories = []

    cat_noise_removal = 'NR'
    cat_dim_reduction = 'DR'
    cat_uses_classifier = 'Clas'
    cat_sample_componentwise = 'SCmp'
    cat_sample_ordinary = 'SO'
    cat_sample_copy = 'SCpy'
    cat_memetic = 'M'
    cat_density_estimation = 'DE'
    cat_density_based = 'DB'
    cat_extensive = 'Ex'
    cat_changes_majority = 'CM'
    cat_uses_clustering = 'Clus'
    cat_borderline = 'BL'
    cat_application = 'A'

    def __init__(self):
        pass

    def det_n_to_sample(self, strategy, n_maj, n_min):
        """
        Determines the number of samples to generate
        Args:
            strategy (str/float): if float, the fraction of the difference
                                    of the minority and majority numbers to
                                    generate, like 0.1 means that 10% of the
                                    difference will be generated if str,
                                    like 'min2maj', the minority class will
                                    be upsampled to match the cardinality
                                    of the majority class
        """
        if isinstance(strategy, float) or isinstance(strategy, int):
            return max([0, int((n_maj - n_min)*strategy)])
        else:
            m = "Value %s for parameter strategy is not supported" % strategy
            raise ValueError(self.__class__.__name__ + ": " + m)

    def sample_between_points(self, x, y):
        """
        Sample randomly along the line between two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
        Returns:
            np.array: the new sample
        """
        return x + (y - x)*self.random_state.random_sample()

    def sample_between_points_componentwise(self, x, y, mask=None):
        """
        Sample each dimension separately between the two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
            mask (np.array): array of 0,1s - specifies which dimensions
                                to sample
        Returns:
            np.array: the new sample being generated
        """
        if mask is None:
            return x + (y - x)*self.random_state.random_sample()
        else:
            return x + (y - x)*self.random_state.random_sample()*mask

    def sample_by_jittering(self, x, std):
        """
        Sample by jittering.
        Args:
            x (np.array): base point
            std (float): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample() - 0.5)*2.0*std

    def sample_by_jittering_componentwise(self, x, std):
        """
        Sample by jittering componentwise.
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample(len(x))-0.5)*2.0 * std

    def sample_by_gaussian_jittering(self, x, std):
        """
        Sample by Gaussian jittering
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return self.random_state.normal(x, std)

    def sample(self, X, y):
        """
        The samplig function reimplemented in child classes
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        return X, y

    def fit_resample(self, X, y):
        """
        Alias of the function "sample" for compatibility with imbalanced-learn
        pipelines
        """
        return self.sample(X, y)

    def sample_with_timing(self, X, y):
        begin = time.time()
        X_samp, y_samp = self.sample(X, y)
        _logger.info(self.__class__.__name__ + ": " +
                     ("runtime: %f" % (time.time() - begin)))
        return X_samp, y_samp

    def preprocessing_transform(self, X):
        """
        Transforms new data according to the possible transformation
        implemented by the function "sample".
        Args:
            X (np.matrix): features
        Returns:
            np.matrix: transformed features
        """
        return X

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass

    def set_params(self, **params):
        """
        Set parameters

        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))

    def __str__(self):
        return self.descriptor()


class UnderSampling(StatisticsMixin,
                    ParameterCheckingMixin,
                    ParameterCombinationsMixin):
    """
    Base class of undersampling approaches.
    """

    def __init__(self):
        """
        Constructorm
        """
        super().__init__()

    def sample(self, X, y):
        """
        Carry out undersampling
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        pass

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass

    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))


class NoSMOTE(OverSampling):
    """
    The goal of this class is to provide a functionality to send data through
    on any model selection/evaluation pipeline with no oversampling carried
    out. It can be used to get baseline estimates on preformance.
    """

    categories = []

    def __init__(self, random_state=None):
        """
        Constructor of the NoSMOTE object.

        Args:
            random_state (int/np.random.RandomState/None): dummy parameter for \
                        the compatibility of interfaces
        """
        super().__init__()

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({}, raw=False)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {}


class SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{smote,
                author={Chawla, N. V. and Bowyer, K. W. and Hall, L. O. and
                            Kegelmeyer, W. P.},
                title={{SMOTE}: synthetic minority over-sampling technique},
                journal={Journal of Artificial Intelligence Research},
                volume={16},
                year={2002},
                pages={321--357}
              }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0
            means that after sampling the number of minority samples will
                                 be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # _logger.info(self.__class__.__name__ + ": " +
        #              "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determining the number of samples to generate
        # by class OverSampling 標籤欲處理
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            # _logger.warning(self.__class__.__name__ +
            #                ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting the model
        n_neigh = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        if n_to_sample == 0:
            return X.copy(), y.copy()

        # generating samples 生成樣本
        base_indices = self.random_state.choice(list(range(len(X_min))),
                                                n_to_sample)
        neighbor_indices = self.random_state.choice(list(range(1, n_neigh)),
                                                    n_to_sample)

        X_base = X_min[base_indices]
        X_neighbor = X_min[ind[base_indices, neighbor_indices]]

        samples = X_base + np.multiply(self.random_state.rand(n_to_sample,
                                                              1),
                                       X_neighbor - X_base)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class MulticlassOversampling(StatisticsMixin):
    """
    Carries out multiclass oversampling

    Example::

        import smote_variants as sv
        import sklearn.datasets as datasets

        dataset= datasets.load_wine()

        oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())

        X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])
    """

    def __init__(self,
                 oversampler=SMOTE(random_state=2),
                 strategy="eq_1_vs_many_successive"):
        """
        Constructor of the multiclass oversampling object

        Args:
            oversampler (obj): an oversampling object
            strategy (str/obj): a multiclass oversampling strategy, currently
                                'eq_1_vs_many_successive' or
                                'equalize_1_vs_many'
        """
        self.oversampler = oversampler
        self.strategy = strategy

    def sample_equalize_1_vs_many(self, X, y):
        """
        Does the sample generation by oversampling each minority class to the
        cardinality of the majority class using all original samples in each
        run.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        message = "Running multiclass oversampling with strategy %s"
        message = message % str(self.strategy)
        _logger.info(self.__class__.__name__ + ": " + message)

        if 'proportion' not in self.oversampler.get_params():
            message = ("Multiclass oversampling strategy %s cannot be "
                       "used with oversampling techniques without proportion"
                       " parameter")
            message = message % str(self.strategy)
            raise ValueError(message)

        # extract class label statistics
        self.class_label_statistics(X, y)

        # sort labels by number of samples
        class_labels = self.class_stats.keys()
        class_labels = sorted(class_labels, key=lambda x: -self.class_stats[x])

        majority_class_label = class_labels[0]

        # determining the majority class data
        X_maj = X[y == majority_class_label]

        # dict to store the results
        results = {}
        results[majority_class_label] = X_maj.copy()

        # running oversampling for all minority classes against all oversampled
        # classes
        for i in range(1, len(class_labels)):
            message = "Sampling minority class with label: %d"
            message = message % class_labels[i]
            _logger.info(self.__class__.__name__ + ": " + message)

            # extract current minority class
            minority_class_label = class_labels[i]
            X_min = X[y == minority_class_label]
            X_maj = X[y != minority_class_label]

            # prepare data to pass to oversampling
            X_training = np.vstack([X_maj, X_min])
            y_training = np.hstack(
                [np.repeat(0, len(X_maj)), np.repeat(1, len(X_min))])

            # prepare parameters by properly setting the proportion value
            params = self.oversampler.get_params()

            num_to_generate = self.class_stats[majority_class_label] - \
                self.class_stats[class_labels[i]]
            num_to_gen_to_all = len(X_maj) - self.class_stats[class_labels[i]]

            params['proportion'] = num_to_generate/num_to_gen_to_all

            # instantiating new oversampling object with the proper proportion
            # parameter
            oversampler = self.oversampler.__class__(**params)

            # executing the sampling
            X_samp, y_samp = oversampler.sample(X_training, y_training)

            # registaring the newly oversampled minority class in the output
            # set
            results[class_labels[i]] = X_samp[len(
                X_training):][y_samp[len(X_training):] == 1]

        # constructing the output set
        X_final = results[class_labels[1]]
        y_final = np.repeat(class_labels[1], len(results[class_labels[1]]))

        for i in range(2, len(class_labels)):
            X_final = np.vstack([X_final, results[class_labels[i]]])
            y_new = np.repeat(class_labels[i], len(results[class_labels[i]]))
            y_final = np.hstack([y_final, y_new])

        return np.vstack([X, X_final]), np.hstack([y, y_final])

    def sample_equalize_1_vs_many_successive(self, X, y):
        """
        Does the sample generation by oversampling each minority class
        successively to the cardinality of the majority class,
        incorporating the results of previous oversamplings.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        message = "Running multiclass oversampling with strategy %s"
        message = message % str(self.strategy)
        _logger.info(self.__class__.__name__ + ": " + message)

        if 'proportion' not in self.oversampler.get_params():
            message = ("Multiclass oversampling strategy %s cannot be used"
                       " with oversampling techniques without proportion"
                       " parameter") % str(self.strategy)
            raise ValueError(message)

        # extract class label statistics
        self.class_label_statistics(X, y)

        # sort labels by number of samples
        class_labels = self.class_stats.keys()
        class_labels = sorted(class_labels, key=lambda x: -self.class_stats[x])

        majority_class_label = class_labels[0]

        # determining the majority class data
        X_maj = X[y == majority_class_label]

        # dict to store the results
        results = {}
        results[majority_class_label] = X_maj.copy()

        # running oversampling for all minority classes against all
        # oversampled classes
        for i in range(1, len(class_labels)):
            message = "Sampling minority class with label: %d"
            message = message % class_labels[i]
            _logger.info(self.__class__.__name__ + ": " + message)

            # extract current minority class
            minority_class_label = class_labels[i]
            X_min = X[y == minority_class_label]

            # prepare data to pass to oversampling
            X_training = np.vstack([X_maj, X_min])
            y_training = np.hstack(
                [np.repeat(0, len(X_maj)), np.repeat(1, len(X_min))])

            # prepare parameters by properly setting the proportion value
            params = self.oversampler.get_params()

            n_majority = self.class_stats[majority_class_label]
            n_class_i = self.class_stats[class_labels[i]]
            num_to_generate = n_majority - n_class_i

            num_to_gen_to_all = i * n_majority - n_class_i

            params['proportion'] = num_to_generate/num_to_gen_to_all

            # instantiating new oversampling object with the proper proportion
            # parameter
            oversampler = self.oversampler.__class__(**params)

            # executing the sampling
            X_samp, y_samp = oversampler.sample(X_training, y_training)

            # adding the newly oversampled minority class to the majority data
            X_maj = np.vstack([X_maj, X_samp[y_samp == 1]])

            # registaring the newly oversampled minority class in the output
            # set
            result_mask = y_samp[len(X_training):] == 1
            results[class_labels[i]] = X_samp[len(X_training):][result_mask]

        # constructing the output set
        X_final = results[class_labels[1]]
        y_final = np.repeat(class_labels[1], len(results[class_labels[1]]))

        for i in range(2, len(class_labels)):
            X_final = np.vstack([X_final, results[class_labels[i]]])
            y_new = np.repeat(class_labels[i], len(results[class_labels[i]]))
            y_final = np.hstack([y_final, y_new])

        return np.vstack([X, X_final]), np.hstack([y, y_final])

    def sample(self, X, y):
        """
        Does the sample generation according to the oversampling strategy.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        if self.strategy == "eq_1_vs_many_successive":
            return self.sample_equalize_1_vs_many_successive(X, y)
        elif self.strategy == "equalize_1_vs_many":
            return self.sample_equalize_1_vs_many(X, y)
        else:
            message = "Multiclass oversampling startegy %s not implemented."
            message = message % self.strategy
            raise ValueError(message)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the multiclass oversampling object
        """
        return {'oversampler': self.oversampler, 'strategy': self.strategy}


class OversamplingClassifier(BaseEstimator, ClassifierMixin):
    """
    This class wraps an oversampler and a classifier, making it compatible
    with sklearn based pipelines.
    這個類包裝了一個過採樣器和一個分類器，使其與基於 sklearn 的管道兼容。
    """

    def __init__(self, oversampler, classifier):
        """
        Constructor of the wrapper.

        Args:
            oversampler (obj): an oversampler object
            classifier (obj): an sklearn-compatible classifier
        """

        self.oversampler = oversampler
        self.classifier = classifier

    def fit(self, X, y=None):
        """
        Carries out oversampling and fits the classifier.

        Args:
            X (np.ndarray): feature vectors
            y (np.array): target values

        Returns:
            obj: the object itself
        """

        X_samp, y_samp = self.oversampler.sample(X, y)
        self.classifier.fit(X_samp, y_samp)

        return self

    def predict(self, X):
        """
        Carries out the predictions.

        Args:
            X (np.ndarray): feature vectors
        """

        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Carries out the predictions with probability estimations.

        Args:
            X (np.ndarray): feature vectors
        """

        return self.classifier.predict_proba(X)

    def get_params(self, deep=True):
        """
        Returns the dictionary of parameters.

        Args:
            deep (bool): wether to return parameters with deep discovery

        Returns:
            dict: the dictionary of parameters
        """

        return {'oversampler': self.oversampler, 'classifier': self.classifier}

    def set_params(self, **parameters):
        """
        Sets the parameters.

        Args:
            parameters (dict): the parameters to set.

        Returns:
            obj: the object itself
        """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self


class MLPClassifierWrapper:
    """
    Wrapper over MLPClassifier of sklearn to provide easier parameterization
    """

    def __init__(self,
                 activation='relu',
                 hidden_layer_fraction=0.1,
                 alpha=0.0001,
                 random_state=None):
        """
        Constructor of the MLPClassifier

        Args:
            activation (str): name of the activation function
            hidden_layer_fraction (float): fraction of the hidden neurons of
                                            the number of input dimensions
            alpha (float): alpha parameter of the MLP classifier
            random_state (int/np.random.RandomState/None): initializer of the
                                                            random state
        """
        self.activation = activation
        self.hidden_layer_fraction = hidden_layer_fraction
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the model to the data

        Args:
            X (np.ndarray): features
            y (np.array): target labels

        Returns:
            obj: the MLPClassifierWrapper object
        """
        hidden_layer_size = max([1, int(len(X[0])*self.hidden_layer_fraction)])
        self.model = MLPClassifier(activation=self.activation,
                                   hidden_layer_sizes=(hidden_layer_size,),
                                   alpha=self.alpha,
                                   random_state=self.random_state).fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the labels of the unseen data

        Args:
            X (np.ndarray): unseen features

        Returns:
            np.array: predicted labels
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predicts the class probabilities of the unseen data

        Args:
            X (np.ndarray): unseen features

        Returns:
            np.matrix: predicted class probabilities
        """
        return self.model.predict_proba(X)

    def get_params(self, deep=False):
        """
        Returns the parameters of the classifier.

        Returns:
            dict: the parameters of the object
        """
        return {'activation': self.activation,
                'hidden_layer_fraction': self.hidden_layer_fraction,
                'alpha': self.alpha,
                'random_state': self.random_state}

    def copy(self):
        """
        Creates a copy of the classifier.

        Returns:
            obj: a copy of the classifier
        """
        return MLPClassifierWrapper(**self.get_params())


class Folding():
    """
    Cache-able folding of dataset for cross-validation
    """

    def __init__(self, dataset, validator, cache_path=None, random_state=None):
        """
        Constructor of Folding object

        Args:
            dataset (dict): dataset dictionary with keys 'data', 'target'
                            and 'DESCR'
            validator (obj): cross-validator object
            cache_path (str): path to cache directory
            random_state (int/np.random.RandomState/None): initializer of
                                                            the random state
        """
        self.dataset = dataset
        self.db_name = self.dataset['name']
        self.validator = validator
        self.cache_path = cache_path
        self.filename = 'folding_' + self.db_name + '.pickle'
        self.db_size = len(dataset['data'])
        self.db_n_attr = len(dataset['data'][0])
        self.imbalanced_ratio = np.sum(
            self.dataset['target'] == 0)/np.sum(self.dataset['target'] == 1)
        self.random_state = random_state

    def do_folding(self):
        """
        Does the folding or reads it from file if already available

        Returns:
            list(tuple): list of tuples of X_train, y_train, X_test, y_test
                            objects
        """

        self.validator.random_state = self.random_state

        if not hasattr(self, 'folding'):
            cond_cache_none = self.cache_path is None
            if not cond_cache_none:
                filename = os.path.join(self.cache_path, self.filename)
                cond_file_not_exists = not os.path.isfile(filename)
            else:
                cond_file_not_exists = False

            if cond_cache_none or cond_file_not_exists:
                _logger.info(self.__class__.__name__ +
                             (" doing folding %s" % self.filename))

                self.folding = {}
                self.folding['folding'] = []
                self.folding['db_size'] = len(self.dataset['data'])
                self.folding['db_n_attr'] = len(self.dataset['data'][0])
                n_maj = np.sum(self.dataset['target'] == 0)
                n_min = np.sum(self.dataset['target'] == 1)
                self.folding['imbalanced_ratio'] = n_maj / n_min

                X = self.dataset['data']
                y = self.dataset['target']

                data = self.dataset['data']
                target = self.dataset['target']

                for train, test in self.validator.split(data, target, target):
                    folding = (X[train], y[train], X[test], y[test])
                    self.folding['folding'].append(folding)
                if self.cache_path is not None:
                    _logger.info(self.__class__.__name__ +
                                 (" dumping to file %s" % self.filename))
                    random_filename = np.random.randint(1000000)
                    random_filename = str(random_filename) + '.pickle'
                    random_filename = os.path.join(self.cache_path,
                                                   random_filename)
                    pickle.dump(self.folding, open(random_filename, "wb"))
                    os.rename(random_filename, os.path.join(
                        self.cache_path, self.filename))
            else:
                _logger.info(self.__class__.__name__ +
                             (" reading from file %s" % self.filename))
                self.folding = pickle.load(
                    open(os.path.join(self.cache_path, self.filename), "rb"))
        return self.folding

    def get_params(self, deep=False):
        return {'db_name': self.db_name}

    def descriptor(self):
        return str(self.get_params())


class Sampling():
    """
    Cache-able sampling of dataset folds
    """

    def __init__(self,
                 folding,
                 sampler,
                 sampler_parameters,
                 scaler,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            folding (obj): Folding object
            sampler (class): class of a sampler object
            sampler_parameters (dict): a parameter combination for the sampler
                                        object
            scaler (obj): scaler object
            random_state (int/np.random.RandomState/None): initializer of the
                                                            random state
        """
        self.folding = folding
        self.db_name = folding.db_name
        self.sampler = sampler
        self.sampler_parameters = sampler_parameters
        self.sampler_parameters['random_state'] = random_state
        self.scaler = scaler
        self.cache_path = folding.cache_path
        self.filename = self.standardized_filename('sampling')
        self.random_state = random_state

    def standardized_filename(self,
                              prefix,
                              db_name=None,
                              sampler=None,
                              sampler_parameters=None):
        """
        standardizes the filename

        Args:
            filename (str): filename

        Returns:
            str: standardized name
        """
        import hashlib

        db_name = (db_name or self.db_name)

        sampler = (sampler or self.sampler)
        sampler = sampler.__name__
        sampler_parameters = sampler_parameters or self.sampler_parameters
        _logger.info(str(sampler_parameters))
        from collections import OrderedDict
        sampler_parameters_ordered = OrderedDict()
        for k in sorted(list(sampler_parameters.keys())):
            sampler_parameters_ordered[k] = sampler_parameters[k]

        message = " sampler parameter string "
        message = message + str(sampler_parameters_ordered)
        _logger.info(self.__class__.__name__ + message)
        sampler_parameter_str = hashlib.md5(
            str(sampler_parameters_ordered).encode('utf-8')).hexdigest()

        filename = '_'.join(
            [prefix, db_name, sampler, sampler_parameter_str]) + '.pickle'
        filename = re.sub('["\\,:(){}]', '', filename)
        filename = filename.replace("'", '')
        filename = filename.replace(": ", "_")
        filename = filename.replace(" ", "_")
        filename = filename.replace("\n", "_")

        return filename

    def cache_sampling(self):
        try:
            import mkl
            mkl.set_num_threads(1)
            _logger.info(self.__class__.__name__ +
                         (" mkl thread number set to 1 successfully"))
        except Exception as e:
            _logger.info(self.__class__.__name__ +
                         (" setting mkl thread number didn't succeed"))
            _logger.info(str(e))

        if not os.path.isfile(os.path.join(self.cache_path, self.filename)):
            # if the sampled dataset does not exist
            sampler_categories = self.sampler.categories
            is_extensive = OverSampling.cat_extensive in sampler_categories
            has_proportion = 'proportion' in self.sampler_parameters
            higher_prop_sampling_avail = None

            if is_extensive and has_proportion:
                proportion = self.sampler_parameters['proportion']
                all_pc = self.sampler.parameter_combinations()
                all_proportions = np.unique([p['proportion'] for p in all_pc])
                all_proportions = all_proportions[all_proportions > proportion]

                for p in all_proportions:
                    tmp_par = self.sampler_parameters.copy()
                    tmp_par['proportion'] = p
                    tmp_filename = self.standardized_filename(
                        'sampling', self.db_name, self.sampler, tmp_par)

                    filename = os.path.join(self.cache_path, tmp_filename)
                    if os.path.isfile(filename):
                        higher_prop_sampling_avail = (p, tmp_filename)
                        break

            if (not is_extensive or not has_proportion or
                    (is_extensive and has_proportion and
                        higher_prop_sampling_avail is None)):
                _logger.info(self.__class__.__name__ + " doing sampling")
                begin = time.time()
                sampling = []
                folds = self.folding.do_folding()
                for X_train, y_train, X_test, y_test in folds['folding']:
                    s = self.sampler(**self.sampler_parameters)

                    if self.scaler is not None:
                        print(self.scaler.__class__.__name__)
                        X_train = self.scaler.fit_transform(X_train, y_train)
                    X_samp, y_samp = s.sample_with_timing(X_train, y_train)

                    if hasattr(s, 'transform'):
                        X_test_trans = s.preprocessing_transform(X_test)
                    else:
                        X_test_trans = X_test.copy()

                    if self.scaler is not None:
                        X_samp = self.scaler.inverse_transform(X_samp)

                    sampling.append((X_samp, y_samp, X_test_trans, y_test))
                runtime = time.time() - begin
            else:
                higher_prop, higher_prop_filename = higher_prop_sampling_avail
                message = " reading and resampling from file %s to %s"
                message = message % (higher_prop_filename, self.filename)
                _logger.info(self.__class__.__name__ + message)
                filename = os.path.join(self.cache_path, higher_prop_filename)
                tmp_results = pickle.load(open(filename, 'rb'))
                tmp_sampling = tmp_results['sampling']
                tmp_runtime = tmp_results['runtime']

                sampling = []
                folds = self.folding.do_folding()
                nums = [len(X_train) for X_train, _, _, _ in folds['folding']]
                i = 0
                for X_train, y_train, X_test, y_test in tmp_sampling:
                    new_num = (len(X_train) - nums[i])/higher_prop*proportion
                    new_num = int(new_num)
                    offset = nums[i] + new_num
                    X_offset = X_train[:offset]
                    y_offset = y_train[:offset]
                    sampling.append((X_offset, y_offset, X_test, y_test))
                    i = i + 1
                runtime = tmp_runtime/p*proportion

            results = {}
            results['sampling'] = sampling
            results['runtime'] = runtime
            results['db_size'] = folds['db_size']
            results['db_n_attr'] = folds['db_n_attr']
            results['imbalanced_ratio'] = folds['imbalanced_ratio']

            _logger.info(self.__class__.__name__ +
                         (" dumping to file %s" % self.filename))

            random_filename = np.random.randint(1000000)
            random_filename = str(random_filename) + '.pickle'
            random_filename = os.path.join(self.cache_path, random_filename)
            pickle.dump(results, open(random_filename, "wb"))
            os.rename(random_filename, os.path.join(
                self.cache_path, self.filename))

    def do_sampling(self):
        self.cache_sampling()
        results = pickle.load(
            open(os.path.join(self.cache_path, self.filename), 'rb'))
        return results

    def get_params(self, deep=False):
        return {'folding': self.folding.get_params(),
                'sampler_name': self.sampler.__name__,
                'sampler_parameters': self.sampler_parameters}

    def descriptor(self):
        return str(self.get_params())


class Evaluation():
    """
    Cache-able evaluation of classifier on sampling
    """

    def __init__(self,
                 sampling,
                 classifiers,
                 n_threads=None,
                 random_state=None):
        """
        Constructor of an Evaluation object

        Args:
            sampling (obj): Sampling object
            classifiers (list(obj)): classifier objects
            n_threads (int/None): number of threads
            random_state (int/np.random.RandomState/None): random state
                                                            initializer
        """
        self.sampling = sampling
        self.classifiers = classifiers
        self.n_threads = n_threads
        self.cache_path = sampling.cache_path
        self.filename = self.sampling.standardized_filename('eval')
        self.random_state = random_state

        self.labels = []
        for i in range(len(classifiers)):
            from collections import OrderedDict
            sampling_parameters = OrderedDict()
            sp = self.sampling.sampler_parameters
            for k in sorted(list(sp.keys())):
                sampling_parameters[k] = sp[k]
            cp = classifiers[i].get_params()
            classifier_parameters = OrderedDict()
            for k in sorted(list(cp.keys())):
                classifier_parameters[k] = cp[k]

            label = str((self.sampling.db_name, sampling_parameters,
                         classifiers[i].__class__.__name__,
                         classifier_parameters))
            self.labels.append(label)

        print(self.labels)

    def calculate_metrics(self, all_pred, all_test, all_folds):
        """
        Calculates metrics of binary classifiction

        Args:
            all_pred (np.matrix): predicted probabilities
            all_test (np.matrix): true labels

        Returns:
            dict: all metrics of binary classification
        """

        results = {}
        if all_pred is not None:
            all_pred_labels = np.apply_along_axis(
                lambda x: np.argmax(x), 1, all_pred)

            results['tp'] = np.sum(np.logical_and(
                np.equal(all_test, all_pred_labels), (all_test == 1)))
            results['tn'] = np.sum(np.logical_and(
                np.equal(all_test, all_pred_labels), (all_test == 0)))
            results['fp'] = np.sum(np.logical_and(np.logical_not(
                np.equal(all_test, all_pred_labels)), (all_test == 0)))
            results['fn'] = np.sum(np.logical_and(np.logical_not(
                np.equal(all_test, all_pred_labels)), (all_test == 1)))
            results['p'] = results['tp'] + results['fn']
            results['n'] = results['fp'] + results['tn']
            results['acc'] = (results['tp'] + results['tn']) / \
                (results['p'] + results['n'])
            results['sens'] = results['tp']/results['p']
            results['spec'] = results['tn']/results['n']
            results['ppv'] = results['tp']/(results['tp'] + results['fp'])
            results['npv'] = results['tn']/(results['tn'] + results['fn'])
            results['fpr'] = 1.0 - results['spec']
            results['fdr'] = 1.0 - results['ppv']
            results['fnr'] = 1.0 - results['sens']
            results['bacc'] = (results['tp']/results['p'] +
                               results['tn']/results['n'])/2.0
            results['gacc'] = np.sqrt(
                results['tp']/results['p']*results['tn']/results['n'])
            results['f1'] = 2*results['tp'] / \
                (2*results['tp'] + results['fp'] + results['fn'])
            mcc_num = results['tp']*results['tn'] - results['fp']*results['fn']
            mcc_denom_0 = (results['tp'] + results['fp'])
            mcc_denom_1 = (results['tp'] + results['fn'])
            mcc_denom_2 = (results['tn'] + results['fp'])
            mcc_denom_3 = (results['tn'] + results['fn'])
            mcc_denom = mcc_denom_0 * mcc_denom_1 * mcc_denom_2*mcc_denom_3
            results['mcc'] = mcc_num/np.sqrt(mcc_denom)
            results['l'] = (results['p'] + results['n']) * \
                np.log(results['p'] + results['n'])
            tp_fp = (results['tp'] + results['fp'])
            tp_fn = (results['tp'] + results['fn'])
            tn_fp = (results['fp'] + results['tn'])
            tn_fn = (results['fn'] + results['tn'])
            results['ltp'] = results['tp']*np.log(results['tp']/(tp_fp*tp_fn))
            results['lfp'] = results['fp']*np.log(results['fp']/(tp_fp*tn_fp))
            results['lfn'] = results['fn']*np.log(results['fn']/(tp_fn*tn_fn))
            results['ltn'] = results['tn']*np.log(results['tn']/(tn_fp*tn_fn))
            results['lp'] = results['p'] * \
                np.log(results['p']/(results['p'] + results['n']))
            results['ln'] = results['n'] * \
                np.log(results['n']/(results['p'] + results['n']))
            uc_num = (results['l'] + results['ltp'] + results['lfp'] +
                      results['lfn'] + results['ltn'])
            uc_denom = (results['l'] + results['lp'] + results['ln'])
            results['uc'] = uc_num/uc_denom
            results['informedness'] = results['sens'] + results['spec'] - 1.0
            results['markedness'] = results['ppv'] + results['npv'] - 1.0
            results['log_loss'] = log_loss(all_test, all_pred)
            results['auc'] = roc_auc_score(all_test, all_pred[:, 1])
            aucs = [roc_auc_score(all_test[all_folds == i],
                                  all_pred[all_folds == i, 1])
                    for i in range(np.max(all_folds)+1)]
            results['auc_mean'] = np.mean(aucs)
            results['auc_std'] = np.std(aucs)
            test_labels, preds = zip(
                *sorted(zip(all_test, all_pred[:, 1]), key=lambda x: -x[1]))
            test_labels = np.array(test_labels)
            th = int(0.2*len(test_labels))
            results['p_top20'] = np.sum(test_labels[:th] == 1)/th
            results['brier'] = np.mean((all_pred[:, 1] - all_test)**2)
        else:
            results['tp'] = 0
            results['tn'] = 0
            results['fp'] = 0
            results['fn'] = 0
            results['p'] = 0
            results['n'] = 0
            results['acc'] = 0
            results['sens'] = 0
            results['spec'] = 0
            results['ppv'] = 0
            results['npv'] = 0
            results['fpr'] = 1
            results['fdr'] = 1
            results['fnr'] = 1
            results['bacc'] = 0
            results['gacc'] = 0
            results['f1'] = 0
            results['mcc'] = np.nan
            results['l'] = np.nan
            results['ltp'] = np.nan
            results['lfp'] = np.nan
            results['lfn'] = np.nan
            results['ltn'] = np.nan
            results['lp'] = np.nan
            results['ln'] = np.nan
            results['uc'] = np.nan
            results['informedness'] = 0
            results['markedness'] = 0
            results['log_loss'] = np.nan
            results['auc'] = 0
            results['auc_mean'] = 0
            results['auc_std'] = 0
            results['p_top20'] = 0
            results['brier'] = 1

        return results

    def do_evaluation(self):
        """
        Does the evaluation or reads it from file

        Returns:
            dict: all metrics
        """

        if self.n_threads is not None:
            try:
                import mkl
                mkl.set_num_threads(self.n_threads)
                message = " mkl thread number set to %d successfully"
                message = message % self.n_threads
                _logger.info(self.__class__.__name__ + message)
            except Exception as e:
                message = " setting mkl thread number didn't succeed"
                _logger.info(self.__class__.__name__ + message)

        evaluations = {}
        if os.path.isfile(os.path.join(self.cache_path, self.filename)):
            evaluations = pickle.load(
                open(os.path.join(self.cache_path, self.filename), 'rb'))

        already_evaluated = np.array([li in evaluations for li in self.labels])

        if not np.all(already_evaluated):
            samp = self.sampling.do_sampling()
        else:
            return list(evaluations.values())

        # setting random states
        for i in range(len(self.classifiers)):
            clf_params = self.classifiers[i].get_params()
            if 'random_state' in clf_params:
                clf_params['random_state'] = self.random_state
                self.classifiers[i] = self.classifiers[i].__class__(
                    **clf_params)
            if isinstance(self.classifiers[i], CalibratedClassifierCV):
                clf_params = self.classifiers[i].base_estimator.get_params()
                clf_params['random_state'] = self.random_state
                class_inst = self.classifiers[i].base_estimator.__class__
                new_inst = class_inst(**clf_params)
                self.classifiers[i].base_estimator = new_inst

        for i in range(len(self.classifiers)):
            if not already_evaluated[i]:
                message = " do the evaluation %s %s %s"
                message = message % (self.sampling.db_name,
                                     self.sampling.sampler.__name__,
                                     self.classifiers[i].__class__.__name__)
                _logger.info(self.__class__.__name__ + message)
                all_preds, all_tests, all_folds = [], [], []
                minority_class_label = None
                majority_class_label = None
                fold_idx = -1
                for X_train, y_train, X_test, y_test in samp['sampling']:
                    fold_idx += 1

                    # X_train[X_train == np.inf]= 0
                    # X_train[X_train == -np.inf]= 0
                    # X_test[X_test == np.inf]= 0
                    # X_test[X_test == -np.inf]= 0

                    class_labels = np.unique(y_train)
                    min_class_size = np.min(
                        [np.sum(y_train == c) for c in class_labels])

                    ss = StandardScaler()
                    X_train_trans = ss.fit_transform(X_train)
                    nonzero_var_idx = np.where(ss.var_ > 1e-8)[0]
                    X_test_trans = ss.transform(X_test)

                    enough_minority_samples = min_class_size > 4
                    y_train_big_enough = len(y_train) > 4
                    two_classes = len(class_labels) > 1
                    at_least_one_feature = (len(nonzero_var_idx) > 0)

                    if not enough_minority_samples:
                        message = " not enough minority samples: %d"
                        message = message % min_class_size
                        _logger.warning(
                            self.__class__.__name__ + message)
                    elif not y_train_big_enough:
                        message = (" number of minority training samples is "
                                   "not enough: %d")
                        message = message % len(y_train)
                        _logger.warning(self.__class__.__name__ + message)
                    elif not two_classes:
                        message = " there is only 1 class in training data"
                        _logger.warning(self.__class__.__name__ + message)
                    elif not at_least_one_feature:
                        _logger.warning(self.__class__.__name__ +
                                        (" no information in features"))
                    else:
                        all_tests.append(y_test)
                        if (minority_class_label is None or
                                majority_class_label is None):
                            class_labels = np.unique(y_train)
                            n_0 = sum(class_labels[0] == y_test)
                            n_1 = sum(class_labels[1] == y_test)
                            if n_0 < n_1:
                                minority_class_label = int(class_labels[0])
                                majority_class_label = int(class_labels[1])
                            else:
                                minority_class_label = int(class_labels[1])
                                majority_class_label = int(class_labels[0])

                        X_fit = X_train_trans[:, nonzero_var_idx]
                        self.classifiers[i].fit(X_fit, y_train)
                        clf = self.classifiers[i]
                        X_pred = X_test_trans[:, nonzero_var_idx]
                        pred = clf.predict_proba(X_pred)
                        all_preds.append(pred)
                        all_folds.append(
                            np.repeat(fold_idx, len(all_preds[-1])))

                if len(all_tests) > 0:
                    all_preds = np.vstack(all_preds)
                    all_tests = np.hstack(all_tests)
                    all_folds = np.hstack(all_folds)

                    evaluations[self.labels[i]] = self.calculate_metrics(
                        all_preds, all_tests, all_folds)
                else:
                    evaluations[self.labels[i]] = self.calculate_metrics(
                        None, None, None)

                evaluations[self.labels[i]]['runtime'] = samp['runtime']
                sampler_name = self.sampling.sampler.__name__
                evaluations[self.labels[i]]['sampler'] = sampler_name
                clf_name = self.classifiers[i].__class__.__name__
                evaluations[self.labels[i]]['classifier'] = clf_name
                sampler_parameters = self.sampling.sampler_parameters.copy()

                evaluations[self.labels[i]]['sampler_parameters'] = str(
                    sampler_parameters)
                evaluations[self.labels[i]]['classifier_parameters'] = str(
                    self.classifiers[i].get_params())
                evaluations[self.labels[i]]['sampler_categories'] = str(
                    self.sampling.sampler.categories)
                evaluations[self.labels[i]
                            ]['db_name'] = self.sampling.folding.db_name
                evaluations[self.labels[i]]['db_size'] = samp['db_size']
                evaluations[self.labels[i]]['db_n_attr'] = samp['db_n_attr']
                evaluations[self.labels[i]
                            ]['imbalanced_ratio'] = samp['imbalanced_ratio']

        if not np.all(already_evaluated):
            _logger.info(self.__class__.__name__ +
                         (" dumping to file %s" % self.filename))
            random_filename = os.path.join(self.cache_path, str(
                np.random.randint(1000000)) + '.pickle')
            pickle.dump(evaluations, open(random_filename, "wb"))
            os.rename(random_filename, os.path.join(
                self.cache_path, self.filename))

        return list(evaluations.values())



def trans(X):
    """
    Transformation function used to aggregate the evaluation results.

    Args:
        X (pd.DataFrame): a grouping of a data frame containing evaluation
                            results
    """
    auc_std = X.iloc[np.argmax(X['auc_mean'].values)]['auc_std']
    cp_auc = X.sort_values('auc')['classifier_parameters'].iloc[-1]
    cp_acc = X.sort_values('acc')['classifier_parameters'].iloc[-1]
    cp_gacc = X.sort_values('gacc')['classifier_parameters'].iloc[-1]
    cp_f1 = X.sort_values('f1')['classifier_parameters'].iloc[-1]
    cp_p_top20 = X.sort_values('p_top20')['classifier_parameters'].iloc[-1]
    cp_brier = X.sort_values('brier')['classifier_parameters'].iloc[-1]
    sp_auc = X.sort_values('auc')['sampler_parameters'].iloc[-1]
    sp_acc = X.sort_values('acc')['sampler_parameters'].iloc[-1]
    sp_gacc = X.sort_values('gacc')['sampler_parameters'].iloc[-1]
    sp_f1 = X.sort_values('f1')['sampler_parameters'].iloc[-1]
    sp_p_top20 = X.sort_values('p_top20')['sampler_parameters'].iloc[-1]
    sp_brier = X.sort_values('p_top20')['sampler_parameters'].iloc[0]

    return pd.DataFrame({'auc': np.max(X['auc']),
                         'auc_mean': np.max(X['auc_mean']),
                         'auc_std': auc_std,
                         'brier': np.min(X['brier']),
                         'acc': np.max(X['acc']),
                         'f1': np.max(X['f1']),
                         'p_top20': np.max(X['p_top20']),
                         'gacc': np.max(X['gacc']),
                         'runtime': np.mean(X['runtime']),
                         'db_size': X['db_size'].iloc[0],
                         'db_n_attr': X['db_n_attr'].iloc[0],
                         'imbalanced_ratio': X['imbalanced_ratio'].iloc[0],
                         'sampler_categories': X['sampler_categories'].iloc[0],
                         'classifier_parameters_auc': cp_auc,
                         'classifier_parameters_acc': cp_acc,
                         'classifier_parameters_gacc': cp_gacc,
                         'classifier_parameters_f1': cp_f1,
                         'classifier_parameters_p_top20': cp_p_top20,
                         'classifier_parameters_brier': cp_brier,
                         'sampler_parameters_auc': sp_auc,
                         'sampler_parameters_acc': sp_acc,
                         'sampler_parameters_gacc': sp_gacc,
                         'sampler_parameters_f1': sp_f1,
                         'sampler_parameters_p_top20': sp_p_top20,
                         'sampler_parameters_brier': sp_brier,
                         }, index=[0])


def _clone_classifiers(classifiers):
    """
    Clones a set of classifiers

    Args:
        classifiers (list): a list of classifier objects
    """
    results = []
    for c in classifiers:
        if isinstance(c, MLPClassifierWrapper):
            results.append(c.copy())
        else:
            results.append(clone(c))

    return results


def _cache_samplings(folding,
                     samplers,
                     scaler,
                     max_n_sampler_par_comb=35,
                     n_jobs=1,
                     random_state=None):
    """

    """
    _logger.info("create sampling objects, random_state: %s" %
                 str(random_state or ""))
    sampling_objs = []

    random_state_init = random_state
    random_state = np.random.RandomState(random_state_init)

    _logger.info("samplers: %s" % str(samplers))
    for s in samplers:
        sampling_par_comb = s.parameter_combinations()
        _logger.info(sampling_par_comb)
        domain = np.array(list(range(len(sampling_par_comb))))
        n_random = min([len(sampling_par_comb), max_n_sampler_par_comb])
        random_indices = random_state.choice(domain, n_random, replace=False)
        _logger.info("random_indices: %s" % random_indices)
        sampling_par_comb = [sampling_par_comb[i] for i in random_indices]
        _logger.info(sampling_par_comb)

        for spc in sampling_par_comb:
            sampling_objs.append(Sampling(folding,
                                          s,
                                          spc,
                                          scaler,
                                          random_state_init))

    # sorting sampling objects to optimize execution
    def key(x):
        if (isinstance(x.sampler, ADG) or isinstance(x.sampler, AMSCO) or
                isinstance(x.sampler, DSRBF)):
            if 'proportion' in x.sampler_parameters:
                return 30 + x.sampler_parameters['proportion']
            else:
                return 30
        elif 'proportion' in x.sampler_parameters:
            return x.sampler_parameters['proportion']
        elif OverSampling.cat_memetic in x.sampler.categories:
            return 20
        else:
            return 10

    sampling_objs = list(reversed(sorted(sampling_objs, key=key)))

    # executing sampling in parallel
    _logger.info("executing %d sampling in parallel" % len(sampling_objs))
    Parallel(n_jobs=n_jobs, batch_size=1)(delayed(s.cache_sampling)()
                                          for s in sampling_objs)

    return sampling_objs


def _cache_evaluations(sampling_objs,
                       classifiers,
                       n_jobs=1,
                       random_state=None):
    # create evaluation objects
    _logger.info("create classifier jobs")
    evaluation_objs = []

    num_threads = None if n_jobs is None or n_jobs == 1 else 1

    for s in sampling_objs:
        evaluation_objs.append(Evaluation(s, _clone_classifiers(
            classifiers), num_threads, random_state))

    _logger.info("executing %d evaluation jobs in parallel" %
                 (len(evaluation_objs)))
    # execute evaluation in parallel
    evals = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(e.do_evaluation)() for e in evaluation_objs)

    return evals


def _read_db_results(cache_path_db):
    results = []
    evaluation_files = glob.glob(os.path.join(cache_path_db, 'eval*.pickle'))

    for f in evaluation_files:
        eval_results = pickle.load(open(f, 'rb'))
        results.append(list(eval_results.values()))

    return results


def read_oversampling_results(datasets, cache_path=None, all_results=False):
    """
    Reads the results of the evaluation

    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset
                            is a dict with 'data', 'target' and 'name' keys
        cache_path (str): path to a cache directory
        all_results (bool): True to return all results, False to return an
                                aggregation

    Returns:
        pd.DataFrame: all results or the aggregated results if all_results is
                        False
    """

    results = []
    for dataset_spec in datasets:

        # loading dataset if needed and determining dataset name
        if not isinstance(dataset_spec, dict):
            dataset = dataset_spec()
        else:
            dataset = dataset_spec

        if 'name' in dataset:
            dataset_name = dataset['name']
        else:
            dataset_name = dataset_spec.__name__

        dataset['name'] = dataset_name

        # determining dataset specific cache path
        cache_path_db = os.path.join(cache_path, dataset_name)

        # reading the results
        res = _read_db_results(cache_path_db)

        # concatenating the results
        _logger.info("concatenating results")
        db_res = [pd.DataFrame(r) for r in res]
        db_res = pd.concat(db_res).reset_index(drop=True)

        _logger.info("aggregating the results")
        if all_results is False:
            db_res = db_res.groupby(by=['db_name', 'classifier', 'sampler'])
            db_res.apply(trans).reset_index().drop('level_3', axis=1)

        results.append(db_res)

    return pd.concat(results).reset_index(drop=True)


def evaluate_oversamplers(datasets,
                          samplers,
                          classifiers,
                          cache_path,
                          validator=RepeatedStratifiedKFold(
                              n_splits=5, n_repeats=3),
                          scaler=None,
                          all_results=False,
                          remove_cache=False,
                          max_samp_par_comb=35,
                          n_jobs=1,
                          random_state=None):
    """
    Evaluates oversampling techniques using various classifiers on various
        datasets

    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset
                            is a dict with 'data', 'target' and 'name' keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        cache_path (str): path to a cache directory
        validator (obj): validator object
        scaler (obj): scaler object
        all_results (bool): True to return all results, False to return an
                                aggregation
        remove_cache (bool): True to remove sampling objects after
                                        evaluation
        max_samp_par_comb (int): maximum number of sampler parameter
                                    combinations to be tested
        n_jobs (int): number of parallel jobs
        random_state (int/np.random.RandomState/None): initializer of the
                                                        random state

    Returns:
        pd.DataFrame: all results or the aggregated results if all_results is
                        False

    Example::

        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        datasets= [imbd.load_glass2, imbd.load_ecoli4]
        oversamplers= [sv.SMOTE_ENN, sv.NEATER, sv.Lee]
        classifiers= [KNeighborsClassifier(n_neighbors= 3),
                      KNeighborsClassifier(n_neighbors= 5),
                      DecisionTreeClassifier()]

        cache_path= '/home/<user>/smote_validation/'

        results= evaluate_oversamplers(datasets,
                                       oversamplers,
                                       classifiers,
                                       cache_path)
    """

    if cache_path is None:
        raise ValueError('cache_path is not specified')

    results = []
    for dataset_spec in datasets:
        # loading dataset if needed and determining dataset name
        if not isinstance(dataset_spec, dict):
            dataset = dataset_spec()
        else:
            dataset = dataset_spec

        if 'name' in dataset:
            dataset_name = dataset['name']
        else:
            dataset_name = dataset_spec.__name__

        dataset['name'] = dataset_name

        dataset_original_target = dataset['target'].copy()
        class_labels = np.unique(dataset['target'])
        n_0 = sum(dataset['target'] == class_labels[0])
        n_1 = sum(dataset['target'] == class_labels[1])
        if n_0 < n_1:
            min_label = class_labels[0]
            maj_label = class_labels[1]
        else:
            min_label = class_labels[1]
            maj_label = class_labels[0]
        min_ind = np.where(dataset['target'] == min_label)[0]
        maj_ind = np.where(dataset['target'] == maj_label)[0]
        np.put(dataset['target'], min_ind, 1)
        np.put(dataset['target'], maj_ind, 0)

        cache_path_db = os.path.join(cache_path, dataset_name)
        if not os.path.isdir(cache_path_db):
            _logger.info("creating cache directory")
            os.makedirs(cache_path_db)

        # checking of samplings and evaluations are available
        samplings_available = False
        evaluations_available = False

        samplings = glob.glob(os.path.join(cache_path_db, 'sampling*.pickle'))
        if len(samplings) > 0:
            samplings_available = True

        evaluations = glob.glob(os.path.join(cache_path_db, 'eval*.pickle'))
        if len(evaluations) > 0:
            evaluations_available = True

        message = ("dataset: %s, samplings_available: %s, "
                   "evaluations_available: %s")
        message = message % (dataset_name, str(samplings_available),
                             str(evaluations_available))
        _logger.info(message)

        if (remove_cache and evaluations_available and
                not samplings_available):
            # remove_cache is enabled and evaluations are available,
            # they are being read
            message = ("reading result from cache, sampling and evaluation is"
                       " not executed")
            _logger.info(message)
            res = _read_db_results(cache_path_db)
        else:
            _logger.info("doing the folding")
            folding = Folding(dataset, validator, cache_path_db, random_state)
            folding.do_folding()

            _logger.info("do the samplings")
            sampling_objs = _cache_samplings(folding,
                                             samplers,
                                             scaler,
                                             max_samp_par_comb,
                                             n_jobs,
                                             random_state)

            _logger.info("do the evaluations")
            res = _cache_evaluations(
                sampling_objs, classifiers, n_jobs, random_state)

        dataset['target'] = dataset_original_target

        # removing samplings once everything is done
        if remove_cache:
            filenames = glob.glob(os.path.join(cache_path_db, 'sampling*'))
            _logger.info("removing unnecessary sampling files")
            if len(filenames) > 0:
                for f in filenames:
                    os.remove(f)

        _logger.info("concatenating the results")
        db_res = [pd.DataFrame(r) for r in res]
        db_res = pd.concat(db_res).reset_index(drop=True)

        random_filename = os.path.join(cache_path_db, str(
            np.random.randint(1000000)) + '.pickle')
        pickle.dump(db_res, open(random_filename, "wb"))
        os.rename(random_filename, os.path.join(
            cache_path_db, 'results.pickle'))

        _logger.info("aggregating the results")
        if all_results is False:
            db_res = db_res.groupby(by=['db_name', 'classifier', 'sampler'])
            db_res = db_res.apply(trans).reset_index().drop('level_3', axis=1)

        results.append(db_res)

    return pd.concat(results).reset_index(drop=True)


def model_selection(dataset,
                    samplers,
                    classifiers,
                    cache_path,
                    score='auc',
                    validator=RepeatedStratifiedKFold(n_splits=5, n_repeats=3),
                    remove_cache=False,
                    max_samp_par_comb=35,
                    n_jobs=1,
                    random_state=None):
    """
    Evaluates oversampling techniques on various classifiers and a dataset
    and returns the oversampling and classifier objects giving the best
    performance

    Args:
        dataset (dict): a dataset is a dict with 'data', 'target' and 'name'
                        keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        cache_path (str): path to a cache directory
        score (str): 'auc'/'acc'/'gacc'/'f1'/'brier'/'p_top20'
        validator (obj): validator object
        all_results (bool): True to return all results, False to return an
                            aggregation
        remove_cache (bool): True to remove sampling objects after
                                        evaluation
        max_samp_par_comb (int): maximum number of sampler parameter
                                    combinations to be tested
        n_jobs (int): number of parallel jobs
        random_state (int/np.random.RandomState/None): initializer of the
                                                        random state

    Returns:
        obj, obj: the best performing sampler object and the best performing
                    classifier object

    Example::

        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        datasets= imbd.load_glass2()
        oversamplers= [sv.SMOTE_ENN, sv.NEATER, sv.Lee]
        classifiers= [KNeighborsClassifier(n_neighbors= 3),
                      KNeighborsClassifier(n_neighbors= 5),
                      DecisionTreeClassifier()]

        cache_path= '/home/<user>/smote_validation/'

        sampler, classifier= model_selection(dataset,
                                             oversamplers,
                                             classifiers,
                                             cache_path,
                                             'auc')
    """

    if score not in ['auc', 'acc', 'gacc', 'f1', 'brier', 'p_top20']:
        raise ValueError("score %s not supported" % score)

    results = evaluate_oversamplers(datasets=[dataset],
                                    samplers=samplers,
                                    classifiers=classifiers,
                                    cache_path=cache_path,
                                    validator=validator,
                                    remove_cache=remove_cache,
                                    max_samp_par_comb=max_samp_par_comb,
                                    n_jobs=n_jobs,
                                    random_state=random_state)

    # extracting the best performing classifier and oversampler parameters
    # regarding AUC
    highest_score = results[score].idxmax()
    cl_par_name = 'classifier_parameters_' + score
    samp_par_name = 'sampler_parameters_' + score
    cl, cl_par, samp, samp_par = results.loc[highest_score][['classifier',
                                                             cl_par_name,
                                                             'sampler',
                                                             samp_par_name]]

    # instantiating the best performing oversampler and classifier objects
    samp_obj = eval(samp)(**eval(samp_par))
    cl_obj = eval(cl)(**eval(cl_par))

    return samp_obj, cl_obj


def cross_validate(dataset,
                   sampler,
                   classifier,
                   validator=RepeatedStratifiedKFold(n_splits=5, n_repeats=3),
                   scaler=StandardScaler(),
                   random_state=None):
    """
    Evaluates oversampling techniques on various classifiers and a dataset
    and returns the oversampling and classifier objects giving the best
    performance

    Args:
        dataset (dict): a dataset is a dict with 'data', 'target' and 'name'
                        keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        validator (obj): validator object
        scaler (obj): scaler object
        random_state (int/np.random.RandomState/None): initializer of the
                                                        random state

    Returns:
        pd.DataFrame: the cross-validation scores

    Example::

        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.neighbors import KNeighborsClassifier

        dataset= imbd.load_glass2()
        sampler= sv.SMOTE_ENN
        classifier= KNeighborsClassifier(n_neighbors= 3)

        sampler, classifier= model_selection(dataset,
                                             oversampler,
                                             classifier)
    """

    class_labels = np.unique(dataset['target'])
    binary_problem = (len(class_labels) == 2)

    dataset_orig_target = dataset['target'].copy()
    if binary_problem:
        _logger.info("The problem is binary")
        n_0 = sum(dataset['target'] == class_labels[0])
        n_1 = sum(dataset['target'] == class_labels[1])
        if n_0 < n_1:
            min_label = class_labels[0]
            maj_label = class_labels[1]
        else:
            min_label = class_labels[0]
            maj_label = class_labels[1]

        min_ind = np.where(dataset['target'] == min_label)[0]
        maj_ind = np.where(dataset['target'] == maj_label)[0]
        np.put(dataset['target'], min_ind, 1)
        np.put(dataset['target'], maj_ind, 0)
    else:
        _logger.info("The problem is not binary")
        label_indices = {}
        for c in class_labels:
            label_indices[c] = np.where(dataset['target'] == c)[0]
        mapping = {}
        for i, c in enumerate(class_labels):
            np.put(dataset['target'], label_indices[c], i)
            mapping[i] = c

    runtimes = []
    all_preds, all_tests = [], []

    for train, test in validator.split(dataset['data'], dataset['target']):
        _logger.info("Executing fold")
        X_train, y_train = dataset['data'][train], dataset['target'][train]
        X_test, y_test = dataset['data'][test], dataset['target'][test]

        begin = time.time()
        X_samp, y_samp = sampler.sample(X_train, y_train)
        runtimes.append(time.time() - begin)

        X_samp_trans = scaler.fit_transform(X_samp)
        nonzero_var_idx = np.where(scaler.var_ > 1e-8)[0]
        X_test_trans = scaler.transform(X_test)

        all_tests.append(y_test)

        classifier.fit(X_samp_trans[:, nonzero_var_idx], y_samp)

        all_preds.append(classifier.predict_proba(
            X_test_trans[:, nonzero_var_idx]))

    if len(all_tests) > 0:
        all_preds = np.vstack(all_preds)
        all_tests = np.hstack(all_tests)

    dataset['target'] = dataset_orig_target

    _logger.info("Computing the results")

    results = {}
    results['runtime'] = np.mean(runtimes)
    results['sampler'] = sampler.__class__.__name__
    results['classifier'] = classifier.__class__.__name__
    results['sampler_parameters'] = str(sampler.get_params())
    results['classifier_parameters'] = str(classifier.get_params())
    results['db_size'] = len(dataset['data'])
    results['db_n_attr'] = len(dataset['data'][0])
    results['db_n_classes'] = len(class_labels)

    if binary_problem:
        results['imbalance_ratio'] = sum(
            dataset['target'] == maj_label)/sum(dataset['target'] == min_label)
        all_pred_labels = np.apply_along_axis(
            lambda x: np.argmax(x), 1, all_preds)

        results['tp'] = np.sum(np.logical_and(
            np.equal(all_tests, all_pred_labels), (all_tests == 1)))
        results['tn'] = np.sum(np.logical_and(
            np.equal(all_tests, all_pred_labels), (all_tests == 0)))
        results['fp'] = np.sum(np.logical_and(np.logical_not(
            np.equal(all_tests, all_pred_labels)), (all_tests == 0)))
        results['fn'] = np.sum(np.logical_and(np.logical_not(
            np.equal(all_tests, all_pred_labels)), (all_tests == 1)))
        results['p'] = results['tp'] + results['fn']
        results['n'] = results['fp'] + results['tn']
        results['acc'] = (results['tp'] + results['tn']) / \
            (results['p'] + results['n'])
        results['sens'] = results['tp']/results['p']
        results['spec'] = results['tn']/results['n']
        results['ppv'] = results['tp']/(results['tp'] + results['fp'])
        results['npv'] = results['tn']/(results['tn'] + results['fn'])
        results['fpr'] = 1.0 - results['spec']
        results['fdr'] = 1.0 - results['ppv']
        results['fnr'] = 1.0 - results['sens']
        results['bacc'] = (results['tp']/results['p'] +
                           results['tn']/results['n'])/2.0
        results['gacc'] = np.sqrt(
            results['tp']/results['p']*results['tn']/results['n'])
        results['f1'] = 2*results['tp'] / \
            (2*results['tp'] + results['fp'] + results['fn'])
        mcc_num = (results['tp']*results['tn'] - results['fp']*results['fn'])
        tp_fp = (results['tp'] + results['fp'])
        tp_fn = (results['tp'] + results['fn'])
        tn_fp = (results['tn'] + results['fp'])
        tn_fn = (results['tn'] + results['fn'])
        mcc_denom = np.sqrt(tp_fp * tp_fn * tn_fp * tn_fn)
        results['mcc'] = mcc_num/mcc_denom
        results['l'] = (results['p'] + results['n']) * \
            np.log(results['p'] + results['n'])
        results['ltp'] = results['tp']*np.log(results['tp']/(
            (results['tp'] + results['fp'])*(results['tp'] + results['fn'])))
        results['lfp'] = results['fp']*np.log(results['fp']/(
            (results['fp'] + results['tp'])*(results['fp'] + results['tn'])))
        results['lfn'] = results['fn']*np.log(results['fn']/(
            (results['fn'] + results['tp'])*(results['fn'] + results['tn'])))
        results['ltn'] = results['tn']*np.log(results['tn']/(
            (results['tn'] + results['fp'])*(results['tn'] + results['fn'])))
        results['lp'] = results['p'] * \
            np.log(results['p']/(results['p'] + results['n']))
        results['ln'] = results['n'] * \
            np.log(results['n']/(results['p'] + results['n']))
        ucc_num = (results['l'] + results['ltp'] + results['lfp'] +
                   results['lfn'] + results['ltn'])
        results['uc'] = ucc_num/(results['l'] + results['lp'] + results['ln'])
        results['informedness'] = results['sens'] + results['spec'] - 1.0
        results['markedness'] = results['ppv'] + results['npv'] - 1.0
        results['log_loss'] = log_loss(all_tests, all_preds)
        results['auc'] = roc_auc_score(all_tests, all_preds[:, 1])
        test_labels, preds = zip(
            *sorted(zip(all_tests, all_preds[:, 1]), key=lambda x: -x[1]))
        test_labels = np.array(test_labels)
        th = int(0.2*len(test_labels))
        results['p_top20'] = np.sum(test_labels[:th] == 1)/th
        results['brier'] = np.mean((all_preds[:, 1] - all_tests)**2)
    else:
        all_pred_labels = np.apply_along_axis(
            lambda x: np.argmax(x), 1, all_preds)

        results['acc'] = accuracy_score(all_tests, all_pred_labels)
        results['confusion_matrix'] = confusion_matrix(
            all_tests, all_pred_labels)
        sum_confusion = np.sum(results['confusion_matrix'], axis=0)
        results['gacc'] = gmean(np.diagonal(
            results['confusion_matrix'])/sum_confusion)
        results['class_label_mapping'] = mapping

    return pd.DataFrame({'value': list(results.values())},
                        index=results.keys())

# 修改後的ENN
class EditedNearestNeighborsMK1(NoiseFilter):
    """
    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    }
    """

    def __init__(self, remove='both', n_jobs=1, n_neighbors=4):
        """
        Constructor of the noise removal object

        Args:
            remove (str): class to remove from 'both'/'min'/'maj'
            n_jobs (int): number of parallel jobs
        """
        super().__init__()

        self.check_isin(remove, 'remove', ['both', 'min', 'maj'])
        self.check_n_jobs(n_jobs, 'n_jobs')
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)

        self.remove = remove
        self.n_jobs = n_jobs
        self.n_neighbors=n_neighbors
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'n_neighbors': [2, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        if len(X) < self.n_neighbors:
            _logger.info(self.__class__.__name__ + ': ' +
                         "Not enough samples for noise removal")
            return X.copy(), y.copy()

        nn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        to_remove = []
        # print(y[indices[i][1:]])
        for i in range(len(X)):
            if not y[i] == mode(y[indices[i][1:]]):
                if (self.remove == 'both' or
                    (self.remove == 'min' and y[i] == self.min_label) or
                        (self.remove == 'maj' and y[i] == self.maj_label)):
                    to_remove.append(i)

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)

    def get_params(self):
        """
        Get noise removal parameters

        Returns:
            dict: dictionary of parameters
        """
        return {'n_neighbors': self.n_neighbors,'remove': self.remove}



class GWOSMOTE_F_ENN(OverSampling):
    

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_memetic]
    #預設參數(可調)
    def __init__(self,
                 pack_size = 20,
                 min_values = [2,0.4],#K=2,p=0.4
                 max_values = [10,1.5],
                 iterations = 30,
                 n_jobs=1,
                 random_state=None):
 
        super().__init__()
        self.check_greater_or_equal(pack_size, "pack_size", 1)
        self.check_greater_or_equal(min_values, "min_values", [2,0.4])
        self.check_greater_or_equal(max_values, "max_values", [10,1.5])
        self.check_greater_or_equal(iterations, "iterations", 0)

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.pack_size = pack_size
        self.min_values = min_values
        self.max_values = max_values
        self.iterations = iterations

        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'pack_size': [20],
                                  'min_values': [2,0.4],
                                  'max_values': [10,1.5],
                                  'iterations':[30]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)


    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()
        # 求解函數，SMOTE用
        def evaluate(variables_values = [5,1.0]):
            # print('K')
            
            K=int(variables_values[0])
            proportion=variables_values[1]
        
            smote = SMOTE(proportion=proportion,
                            n_neighbors=K,
                            n_jobs=self.n_jobs,
                            random_state=self.random_state)
            X_samp, y_samp = smote.fit_resample(X, y)

            kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
            preds = []
            tests = []
            FF=[]
            for train, test in kfold.split(X_samp,y_samp):
                dt = DecisionTreeClassifier(random_state=2)

                dt.fit(X_samp[train], y_samp[train])
                yp=dt.predict(X_samp[test])
                preds.append(yp)
                tests.append(y_samp[test])
                FF.append(f1_score(y_samp[test],yp))
            preds = np.hstack(preds)
            tests = np.hstack(tests)

            F1=statistics.mean(FF)

            return F1

        eva=evaluate
        # 灰狼演算法
        def GWO(target_function=evaluate):
            pack_size = self.pack_size 
            min_values = self.min_values
            max_values = self.max_values
            iterations = self.iterations
            def target_function():
                return

            # Function: Initialize Variables 函数：初始化变量
            def initial_position(pack_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
                position = np.zeros((pack_size, len(min_values)+1))
                # print(position)
                for i in range(0, pack_size):
                    for j in range(0, len(min_values)):
                        position[i,j] = random.uniform(min_values[j], max_values[j])
                    # print(i)
                    # print(position)

                    # print(position[i,0:position.shape[1]-1])
                    position[i,-1] = target_function(position[i,0:position.shape[1]-1])
                # print(position[i,-1])
                # print(position)

                return position

            # Function: Initialize Alpha 初始化阿爾法
            def alpha_position(dimension = 2, target_function = target_function):
                alpha = np.zeros((1, dimension + 1))
                for j in range(0, dimension-1):
                    alpha[0,j] = int(2)
                print("alpha")

                # alpha[1,2]=1.0
                alpha[0,1:alpha.shape[1]-1]=0.1
                # print(alpha)
                # print(alpha[0,0:alpha.shape[1]-1])
                alpha[0,-1] = target_function(alpha[0,0:alpha.shape[1]-1])
                return alpha

            # Function: Initialize Beta
            def beta_position(dimension = 2, target_function = target_function):
                beta = np.zeros((1, dimension + 1))
                print("beta")
                for j in range(0, dimension-1):
                    beta[0,j] = int(2)
                beta[0,1:beta.shape[1]-1]=0.1
                beta[0,-1] = target_function(beta[0,0:beta.shape[1]-1])
                return beta

            # Function: Initialize Delta

            def delta_position(dimension = 2, target_function = target_function):
                delta =  np.zeros((1, dimension + 1))
                for j in range(0, dimension-1):
                    delta[0,j] = int(2)
                delta[0,1:delta.shape[1]-1]=0.1
                delta[0,-1] = target_function(delta[0,0:delta.shape[1]-1])
                return delta


            # Function: Updtade Pack by Fitness
            def update_pack(position, alpha, beta, delta):
                updated_position = np.copy(position)
                for i in range(0, position.shape[0]):
                    if (updated_position[i,-1] > alpha[0,-1]):
                        alpha[0,:] = np.copy(updated_position[i,:])
                    if (updated_position[i,-1] < alpha[0,-1] and updated_position[i,-1] > beta[0,-1]):
                        beta[0,:] = np.copy(updated_position[i,:])
                    if (updated_position[i,-1] < alpha[0,-1] and updated_position[i,-1] < beta[0,-1]  and updated_position[i,-1] > delta[0,-1]):
                        delta[0,:] = np.copy(updated_position[i,:])
                return alpha, beta, delta

            # Function: Updtade Position
            def update_position(position, alpha, beta, delta, a_linear_component = 2, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
                updated_position = np.copy(position)
                for i in range(0, updated_position.shape[0]):
                    for j in range (0, len(min_values)):   
                        r1_alpha              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                        r2_alpha              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                        a_alpha               = 2*a_linear_component*r1_alpha - a_linear_component
                        c_alpha               = 2*r2_alpha      
                        distance_alpha        = abs(c_alpha*alpha[0,j] - position[i,j]) 
                        x1                    = alpha[0,j] - a_alpha*distance_alpha   
                        r1_beta               = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                        r2_beta               = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                        a_beta                = 2*a_linear_component*r1_beta - a_linear_component
                        c_beta                = 2*r2_beta            
                        distance_beta         = abs(c_beta*beta[0,j] - position[i,j]) 
                        x2                    = beta[0,j] - a_beta*distance_beta                          
                        r1_delta              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                        r2_delta              = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                        a_delta               = 2*a_linear_component*r1_delta - a_linear_component
                        c_delta               = 2*r2_delta            
                        distance_delta        = abs(c_delta*delta[0,j] - position[i,j]) 
                        x3                    = delta[0,j] - a_delta*distance_delta                                 
                        updated_position[i,j] = np.clip(((x1 + x2 + x3)/3),min_values[j],max_values[j])     
                    updated_position[i,-1] = target_function(updated_position[i,0:updated_position.shape[1]-1])
                return updated_position

            # GWO Function
            def grey_wolf_optimizer(pack_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function):    
                count    = 0
                alpha    = alpha_position(dimension = len(min_values), target_function = target_function)
                beta     = beta_position(dimension  = len(min_values), target_function = target_function)
                delta    = delta_position(dimension = len(min_values), target_function = target_function)
                position = initial_position(pack_size = pack_size, min_values = min_values, max_values = max_values, target_function = target_function)
                while (count <= iterations):      
                    # print("Iteration = ", count, " f(x) = ", alpha[-1])      
                    a_linear_component = 2 - count*(2/iterations)
                    alpha, beta, delta = update_pack(position, alpha, beta, delta)
                    position           = update_position(position, alpha, beta, delta, a_linear_component = a_linear_component, min_values = min_values, max_values = max_values, target_function = target_function)       
                    count              = count + 1       
                print(alpha[-1])    
                return alpha
            return grey_wolf_optimizer( pack_size = self.pack_size ,min_values = self.min_values,max_values = self.max_values, iterations = self.iterations,target_function = evaluate)

        
        # 求解函數，ENN用                
        def evaluateENN(variables_values = [5,1.0]):

            K=int(variables_values[0])

            K=int(K)
            enn=EditedNearestNeighborsMK1(n_jobs=2,n_neighbors=K,remove='both')

            X_samp, y_samp =enn.remove_noise(X,y)

            kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
            preds = []
            tests = []
            FF=[]
            for train, test in kfold.split(X_samp,y_samp):
                dt = DecisionTreeClassifier(random_state=2)
                dt.fit(X_samp[train], y_samp[train])
                yp=dt.predict(X_samp[test])
                preds.append(yp)
                tests.append(y_samp[test])
                FF.append(f1_score(y_samp[test],yp))
            preds = np.hstack(preds)
            tests = np.hstack(tests)

            F1=statistics.mean(FF)
            return F1

        # 優化
        gwo=GWO(target_function = evaluate)
        gwoENN=GWO(target_function = evaluateENN)
        # 汲取參數並投入SMOTE
        SnX,Sny=SMOTE(proportion=gwo[-1][1],
                     n_neighbors=int(gwo[-1][0]),
                     n_jobs=self.n_jobs,
                     random_state=self.random_state).sample(X, y)
        # 汲取參數並投入ENN
        enn=EditedNearestNeighborsMK1(n_jobs=2,n_neighbors=int(gwoENN[-1][0]))
        ENNX,ENNy=enn.remove_noise(X,y)
        # 開始切分
        maj_nX= SnX[Sny == self.maj_label]
        min_nX= SnX[Sny == self.min_label]

        maj_ENNX=ENNX[ENNy==self.maj_label]
        min_ENNX=ENNX[ENNy==self.min_label]
        minx=X[y==self.min_label]

        XMAJ=[maj_nX,maj_ENNX]
        XMIN=[min_nX,min_ENNX,minx]
        yMAJ=[Sny[Sny==self.maj_label],ENNy[ENNy==self.maj_label]]
        yMin=[Sny[Sny==self.min_label],ENNy[ENNy==self.min_label],y[y==self.min_label]]
        kappaAAA=[]#實際為F1
        
        AAA=[]# 備用
        # 組合後投入分類，並分別引入List
        for A in range(len(XMAJ)):
            
            for B in range(len(XMIN)):
                ENDX=np.vstack([XMAJ[A],XMIN[B]])
                ENDy=np.hstack([yMAJ[A],yMin[B]])
                kfold = StratifiedKFold(n_splits=min([len(ENDX), 10]),shuffle=True,random_state=1)
                preds = []
                tests = []
                FFend=[]
                for train, test in kfold.split(ENDX,ENDy):
                    dt = DecisionTreeClassifier(random_state=2)
                    dt.fit(ENDX[train], ENDy[train])
                    # dt.fit(X_samp[train], y_samp[train])
                    yp=dt.predict(ENDX[test])
                    preds.append(yp)
                    tests.append(ENDy[test])
                    FFend.append(f1_score(ENDy[test],yp))
                FFeee=statistics.mean(FFend)    
                
                kappaAAA.append(FFeee)
                AAA.append(A)
        MAXKK=max(kappaAAA)
        # 判斷大小，並輸出最佳組合
        if MAXKK==kappaAAA[0]:
            SSX=np.vstack([XMAJ[0],XMIN[0]])
            SSy=np.hstack([yMAJ[0],yMin[0]])
        elif MAXKK==kappaAAA[1]:
            SSX=np.vstack([XMAJ[0],XMIN[1]])
            SSy=np.hstack([yMAJ[0],yMin[1]])
        elif MAXKK==kappaAAA[2]:
            SSX=np.vstack([XMAJ[0],XMIN[2]])
            SSy=np.hstack([yMAJ[0],yMin[2]])
        elif MAXKK==kappaAAA[3]:
            SSX=np.vstack([XMAJ[1],XMIN[0]])
            SSy=np.hstack([yMAJ[1],yMin[0]])
        elif MAXKK==kappaAAA[4]:
            SSX=np.vstack([XMAJ[1],XMIN[1]])
            SSy=np.hstack([yMAJ[1],yMin[1]])

        else:
            SSX=np.vstack([XMAJ[1],XMIN[2]])
            SSy=np.hstack([yMAJ[1],yMin[2]])    
        return SSX,SSy


    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'pack_size': self.pack_size,
                'min_values': self.min_values,
                'max_values': self.max_values,
                'iterations': self.iterations,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}











