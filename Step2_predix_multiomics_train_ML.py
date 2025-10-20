from sklearn.model_selection import train_test_split
import joblib
from joblib import Parallel, delayed

import datetime
import glob
import inspect
import inspect
import itertools
import logging
import os
# import seaborn as sns
import os.path
import pickle
import re
import statistics
import sys
import time
import urllib
from itertools import product, combinations
from logging import getLogger
from typing import Tuple, Dict, List

import ITMO_FS
import feature_engine
import lightgbm as lgb
import matplotlib.pyplot as plt
# import mico
# from mico import MutualInformationConicOptimization
# os.chdir('/home/gman/PycharmProjects/HandE_ML/mifs_master')
###################################################################################################import mifs
import numpy as np
import numpy.ma as ma
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pylab
# https://github.com/fbrundu/pymrmr
##################################################################################import pymrmr
import scipy as sp
import shap
import six
import sklearn.preprocessing
import xgboost as xgb
import xlrd
# import cv2
import xlsxwriter
import xlwt

from sklearn.model_selection import GridSearchCV

from BorutaShap import BorutaShap
from ITMO_FS.filters.multivariate import MultivariateFilter
from ITMO_FS.filters.univariate import f_ratio_measure, spearman_corr, gini_index, information_gain, reliefF_measure, \
    kendall_corr, su_measure
from ITMO_FS.filters.univariate import select_k_best, select_best_percentage, UnivariateFilter, measures
# ---------------------------------------------------------------------------------------------------------------------
#####################################################################################################################################from boruta import BorutaPy
from catboost import CatBoostClassifier
from feature_engine.selection import DropDuplicateFeatures, DropConstantFeatures, DropCorrelatedFeatures, \
    SmartCorrelatedSelection, DropHighPSIFeatures
from imbalanced_ensemble.ensemble import AdaCostClassifier, AdaUBoostClassifier, AsymBoostClassifier, \
    CompatibleAdaBoostClassifier, CompatibleBaggingClassifier, SelfPacedEnsembleClassifier
from imblearn.combine import SMOTETomek
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier, BalancedBaggingClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE)
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
# import seaborn as sns
from matplotlib.colors import ListedColormap
from numpy import genfromtxt
from numpy import nan
from plotly.subplots import make_subplots
from powershap import PowerShap
####################################################################from pymrmr import mRMR
from scipy import stats
from scipy.stats import rankdata
from scipy.stats import zscore
from sklearn import model_selection
from sklearn import set_config
from sklearn import set_config
from sklearn import tree  # , cross_validation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest, chi2, RFECV, SelectFpr, SelectFdr, \
    SelectFwe, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoCV, Lasso, Ridge, ElasticNet
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, f1_score, classification_report, cohen_kappa_score, make_scorer, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mutual_info_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, balanced_accuracy_score, average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
#########from sklearn.pipeline import Pipeline, FeatureUnion , make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
# from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from skrebate import ReliefF, SURF, MultiSURF, SURFstar, MultiSURFstar, TuRF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
#from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_validate


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm
#pip install arfs
import arfs

import arfs.feature_selection as arfsfs
import arfs.feature_selection.allrelevant as arfsgroot
from arfs.feature_selection import (
    MinRedundancyMaxRelevance,
    GrootCV,
    MissingValueThreshold,
    UniqueValuesThreshold,
    CollinearityThreshold,
    make_fs_summary,
)
from arfs.benchmark import highlight_tick, compare_varimp, sklearn_pimp_bench
from lightgbm import LGBMClassifier
from arfs.utils import LightForestClassifier
from sklearn.preprocessing import LabelEncoder

from fairlearn.metrics import demographic_parity_ratio, count, false_positive_rate, selection_rate, equalized_odds_ratio, MetricFrame

###############################################################################################from boruta import BorutaPy

import inspect

def unpanda(df):
    if not isinstance(df, np.ndarray):
        df = df.to_numpy()
    return df


def read_pickle(filename):
    with open(filename, 'rb') as handle:
        dct = pickle.load(handle)

    return dct


class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X = X.to_numpy()
        return X

from statsmodels.stats.outliers_influence import variance_inflation_factor

####################
class PandasSimpleImputer(SimpleImputer):
    """A wrapper around `SimpleImputer` to return data frames with columns.
    """

    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)





##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

def return_cross_validate_performance(all_tests ):
    overall_results_out_mean = []
    overall_results_out_std = []

    overall_results_out_mean.append([round(np.nanmean(all_tests['test_acc']), 4),
                                     round(np.nanmean(all_tests['test_sens']), 4),
                                     round(np.nanmean(all_tests['test_spec']), 4),
                                     round(np.nanmean(all_tests['test_roc']), 4),
                                     round(np.nanmean(all_tests['test_Fscore']), 4),
                                     round(np.nanmean(all_tests['test_kappa']), 4),
                                     round(np.nanmean(all_tests['test_PPV']), 4),
                                     round(np.nanmean(all_tests['test_NPV']), 4),
                                     round(np.nanmean(all_tests['test_MCC']), 4),
                                     round(np.nanmean(all_tests['test_AverPR_score']), 4),
                                     round(np.nanmean(all_tests['test_balanced']), 4)
                                     ]
                                    )  # 1 as the positive class

    overall_results_out_std.append([round(np.nanstd(all_tests['test_acc']), 4),
                                     round(np.nanstd(all_tests['test_sens']), 4),
                                     round(np.nanstd(all_tests['test_spec']), 4),
                                     round(np.nanstd(all_tests['test_roc']), 4),
                                     round(np.nanstd(all_tests['test_Fscore']), 4),
                                     round(np.nanstd(all_tests['test_kappa']), 4),
                                     round(np.nanstd(all_tests['test_PPV']), 4),
                                     round(np.nanstd(all_tests['test_NPV']), 4),
                                     round(np.nanstd(all_tests['test_MCC']), 4),
                                     round(np.nanstd(all_tests['test_AverPR_score']), 4),
                                     round(np.nanstd(all_tests['test_balanced']), 4)
                                    ]
                                   )  # 1 as the positive class


    df_results_out_mean = pd.DataFrame(overall_results_out_mean,
                                       columns=['Accuracy_Mean', 'Sensitivity_Mean', 'Specificity_Mean', 'ROC_Mean', 'F1_Mean', 'Kappa_Mean', 'PPV_Mean',
                                                'NPV_Mean', 'MCC_Mean', 'PrecRecall_Mean', 'Balanced_Acc_Mean'])


    df_results_out_std = pd.DataFrame(overall_results_out_std,
                                      columns=['Accuracy_Std', 'Sensitivity_Std', 'Specificity_Std', 'ROC_Std', 'F1_Std', 'Kappa_Std', 'PPV_Std', 'NPV_Std',
                                               'MCC_Std', 'PrecRecall_Std','Balanced_Acc_Std'])



    return df_results_out_mean.join(df_results_out_std)

##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
def read_pickle(filename):
    with open(filename, 'rb') as handle:
        dct = pickle.load(handle)

    return dct
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=10, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = SimpleImputer(strategy='median')#PandasSimpleImputer()###########Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                #################################################print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

def evaluate_model(model, X_test, y_test):
    """
    Displays the Accuracy, F1, AUROC, and graphical confusion matrix for a mode.
    """
    pred_test = model.predict(X_test)
    pred_prob_test = [p[1] for p in model.predict_proba(X_test)]

    return (accuracy_score(y_test, pred_test),
            balanced_accuracy_score(y_test, pred_test),
            f1_score(y_test, pred_test),
            recall_score(y_test, pred_test, pos_label= 0) ,
            recall_score(y_test, pred_test, pos_label= 1) ,
            precision_score(y_test, pred_test, pos_label= 1) ,
            precision_score(y_test, pred_test, pos_label=0),
            average_precision_score(y_test, pred_prob_test),
            roc_auc_score(y_test, pred_prob_test))

##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
class select_from_boruta(BaseEstimator, TransformerMixin):
    def __init__(self,model_input):
        self.zzz_columns = None
        self.model_input = model_input
    def fit(self, X, y ):
        zzz = BorutaShap(model=self.model_input, importance_measure='shap', classification=True)
        zzz.fit(X, y, n_trials=100, random_state=0)
        self.zzz_columns = zzz.Subset().columns
        return self
    def get_important_feats(self):
        #print('important')
        return self.zzz_columns
    def transform(self, X , y = None):
        return X[self.zzz_columns]
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
class PandasFromArray(BaseEstimator, TransformerMixin):
    def __init__(self,sel_columns):

        sel_columns = [s.strip('x') for s in sel_columns]
        sel_columns = list(map(int, sel_columns))
        self.sel_columns = sel_columns

    def fit(self, X, y=None):
        self.sel_columns = X.columns[self.sel_columns]
        return self

    def transform(self, X):
        return X[self.sel_columns]

##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
def set_pipeline(feat_selector,model_internal, selection_category , use_scaling, perc, n_estimators, max_iter):

    coll_thres = 0.8
    miss_thres = 0.2

    # pipeline for RUSBoostClassifier--------------------------------------------------------------------------
    if selection_category == 'rusboost':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),
                 ("arfs",
                  RUSBoostClassifier(estimator = feat_selector,n_estimators=n_estimators, sampling_strategy='auto', random_state=0))
                 # ('model', model_internal)
                 ])
        else:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ("arfs",
                  RUSBoostClassifier(estimator = feat_selector,n_estimators=n_estimators, sampling_strategy='auto', random_state=0))
                 # ('model', model_internal)
                 ])

    # pipeline for CatBoostClassifier--------------------------------------------------------------------------
    elif selection_category == 'adacost':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),
                 ("arfs",
                  AdaCostClassifier(estimator=feat_selector, n_estimators=n_estimators,algorithm='SAMME', random_state=0))
                 # ('model', model_internal)
                 ])
        else:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ("arfs",
                  AdaCostClassifier(estimator=feat_selector, n_estimators=n_estimators,algorithm='SAMME', random_state=0))
                 # ('model', model_internal)
                 ])

    # pipeline for CatBoostClassifier--------------------------------------------------------------------------
    elif selection_category == 'catboost':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),
                 ("arfs",
                  CatBoostClassifier(random_state=0, n_estimators=n_estimators,verbose=0))
                 # ('model', model_internal)
                 ])
        else:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ("arfs",
                  CatBoostClassifier(random_state=0, n_estimators=n_estimators,verbose=0))
                 # ('model', model_internal)
                 ])

    # pipeline for SelfPacedEnsembleClassifier--------------------------------------------------------------------------
    elif selection_category == 'selfpace':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),
                 ("arfs",
                  SelfPacedEnsembleClassifier(estimator=feat_selector, n_estimators=n_estimators, random_state=0))
                 # ('model', model_internal)
                 ])
        else:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ("arfs",
                  SelfPacedEnsembleClassifier(estimator=feat_selector, n_estimators=n_estimators, random_state=0))
                 # ('model', model_internal)
                 ])

    # pipeline for AsymBoostClassifier--------------------------------------------------------------------------
    elif selection_category == 'asymboost':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),
                 ("arfs",
                  AsymBoostClassifier(estimator=feat_selector,n_estimators=n_estimators,algorithm='SAMME', random_state=0))
                 # ('model', model_internal)
                 ])
        else:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ("arfs",
                  AsymBoostClassifier(estimator=feat_selector,n_estimators=n_estimators,algorithm='SAMME', random_state=0))
                 # ('model', model_internal)
                 ])

    #pipeline for arfs------------------------------------------------------------------------------------------------------
    elif selection_category == 'arfs':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),
                 ("arfs", arfsgroot.Leshy(feat_selector, perc=perc, n_estimators=n_estimators, verbose=1, alpha=0.05,max_iter=max_iter, random_state=0, importance="shap")),
                 ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ("arfs", arfsgroot.Leshy(feat_selector, perc=perc, n_estimators=n_estimators, verbose=1, alpha=0.05,max_iter=max_iter, random_state=0, importance="shap")),
                 ('model', model_internal)])


    #pipeline for CompatibleAdaBoostClassifier--------------------------------------------------------------------------
    elif selection_category == 'compada':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),
                 ("arfs", CompatibleAdaBoostClassifier(estimator=feat_selector, random_state=0, n_estimators= n_estimators, algorithm = 'SAMME'))
                 #('model', model_internal)
                ])
        else:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ("arfs", CompatibleAdaBoostClassifier(estimator=feat_selector, random_state=0, n_estimators= n_estimators, algorithm = 'SAMME'))
                 #('model', model_internal)
                 ])

    # pipeline for boruta------------------------------------------------------------------------------------------------------
    elif selection_category == 'boruta':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline([
                                         ("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                                         ('collinear', CollinearityThreshold(threshold=coll_thres)),
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ('trasform', StandardScaler().set_output(transform='pandas')),
                                         ('arfs', select_from_boruta(feat_selector)),
                                         ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline([
                                         ("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                                         ('collinear', CollinearityThreshold(threshold=coll_thres)),
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ('arfs', select_from_boruta(feat_selector)),
                                         ('model', model_internal)])

    # pipeline for simple FS------------------------------------------------------------------------------------------------------
    elif selection_category == 'simple':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                                         ('collinear', CollinearityThreshold(threshold=coll_thres)),
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ('trasform', StandardScaler().set_output(transform='pandas')),
                                         ("arfs", feat_selector),
                                         ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                                         ('collinear', CollinearityThreshold(threshold=coll_thres)),
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ("arfs", feat_selector),
                                         ('model', model_internal)])

    # pipeline for simple FS with pd to numpy transform------------------------------------------------------------------------------
    elif selection_category == 'simple_panda':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                                         ('collinear', CollinearityThreshold(threshold=coll_thres)),
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ('trasform', StandardScaler().set_output(transform='pandas')),
                                        # ("unpandarizer", FunctionTransformer(unpanda)),
                                        # ("unpandarizer", CustomTransformer()),
                                         ("arfs", feat_selector),
                                         ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                                         ('collinear', CollinearityThreshold(threshold=coll_thres)),
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         #("unpandarizer", FunctionTransformer(unpanda)),
                                         #("unpandarizer", CustomTransformer()),
                                         ("arfs", feat_selector),
                                         ('model', model_internal)])

    # pipeline for simple FS with pd to numpy transform-  +   y_train_to_numpy() argotera------------------------------------------
    elif selection_category == 'simple_panda_y_numpy':
        if use_scaling == 'norm':
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),
                 #("unpandarizer", FunctionTransformer(unpanda)),
                 ("unpandarizer", CustomTransformer()),
                 ("arfs", feat_selector),
                 ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                 ('collinear', CollinearityThreshold(threshold=coll_thres)),
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),  #epitides gia na travaei to get_names_out
                 #("unpandarizer", FunctionTransformer(unpanda)),
                 ("unpandarizer", CustomTransformer()),
                 ("arfs", feat_selector),
                 ('model', model_internal)])

    else:
        print('wrong selection category')


    return arfs_fs_pipeline


####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
#fair_metrics = ['menopause', 'tumor_subtype']

def return_fairness(fair_ytest, fair_ypred, fair_metrics_fold, fair_metrics):

    equal_odds_ratio = equalized_odds_ratio(fair_ytest,fair_ypred,sensitive_features=fair_metrics_fold[fair_metrics])  #The ideal value for this metric is 1, which indicates that the true
                                                                                                    #and false positive rates for different groups are equal

    demo_par_ratio = demographic_parity_ratio(fair_ytest,fair_ypred,sensitive_features=fair_metrics_fold[fair_metrics]) # The ideal value for this metric is 1, which indicates that the
                                                                                                       # selection rates for different groups are equal.
    return [equal_odds_ratio, demo_par_ratio]
####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

def return_feature_importance(pipe_set, train, y_train, skf, num_thres, selection_category, model_name, data_for_fair_ml):
    ####selection_category   --> arfs / boruta / simple / simplepd / simplepdnu
    all_features = []
    all_important_scores = []

    #check oti exei fugei to patient_id
    fair_metrics = data_for_fair_ml.columns.values

    if selection_category == 'simplepdnu':
        y_train_real = y_train.values
    else:
        y_train_real = y_train
    all_tests = cross_validate(pipe_set, train, y_train_real, cv=skf, scoring=custom_scoring, return_estimator=True)

    ##### assess fairness from the skf stratified kfold
    fair_results = []
    for i, (train_index, test_index) in enumerate(skf):#############3(skf.split(train, y_train.values)):
        #print(f"Fold {i}:")
        fair_ytest = y_train.iloc[test_index].values
        fair_metrics_fold = data_for_fair_ml.iloc[test_index]

        if np.isnan(all_tests['test_acc'][i]):
            fair_results.append(np.full(len(fair_metrics)*2, np.nan).tolist())
        else:
            fair_ypred = all_tests['estimator'][i].predict(train.iloc[test_index])
            fair_results.append(return_fairness(fair_ytest, fair_ypred, fair_metrics_fold, fair_metrics))

        #print(f"  Test:  index={test_index}")
        #all_tests['estimator'][0].predict(train.iloc[test_index])
        #print(accuracy_score(y_train.iloc[test_index], all_tests['estimator'][i].predict(train.iloc[test_index])))


    for iii in range(len(all_tests['estimator'])):
        if selection_category == 'catboost': #OK
            tmp = all_tests['estimator'][iii]['arfs'].feature_importances_
            all_features.append(all_tests['estimator'][iii]['arfs'].feature_names_)
            all_important_scores.append(tmp)
        #---------------------------------------------------------------------------------------------------------------
        #arfs selection returns significant feature names. no weights
        elif selection_category == 'arfs': #OK
            tmp = all_tests['estimator'][iii]['arfs'].get_feature_names_out()
            all_important_scores.append([1] * len(tmp))
            all_features.append(tmp)
        #---------------------------------------------------------------------------------------------------------------
        elif selection_category == 'boruta': #OK
            tmp = all_tests['estimator'][iii]['arfs'].get_important_feats().values
            all_features.append(tmp)
            all_important_scores.append([1] * len(tmp))
        # ---------------------------------------------------------------------------------------------------------------
        elif selection_category == 'compada': #OK
            tmp = all_tests['estimator'][iii]['arfs'].feature_importances_
            all_features.append(all_tests['estimator'][iii]['arfs'].feature_names_in_)
            all_important_scores.append(tmp)
        # ---------------------------------------------------------------------------------------------------------------
        elif selection_category == 'asymboost': #OK
            tmp = all_tests['estimator'][iii]['arfs'].feature_importances_
            all_features.append(all_tests['estimator'][iii]['arfs'].feature_names_in_)
            all_important_scores.append(tmp)
        # ---------------------------------------------------------------------------------------------------------------
        elif selection_category == 'adacost': #OK
            tmp = all_tests['estimator'][iii]['arfs'].feature_importances_
            all_features.append(all_tests['estimator'][iii]['arfs'].feature_names_in_)
            all_important_scores.append(tmp)
        # ---------------------------------------------------------------------------------------------------------------
        elif selection_category == 'rusboost': #OK
            tmp = all_tests['estimator'][iii]['arfs'].feature_importances_
            all_features.append(all_tests['estimator'][iii]['arfs'].feature_names_in_)
            all_important_scores.append(tmp)
        # ---------------------------------------------------------------------------------------------------------------
        elif selection_category == 'selfpace':  # OK
            tmp = all_tests['estimator'][iii]['arfs'].feature_importances_
            all_features.append(all_tests['estimator'][iii]['arfs'].feature_names_in_)
            all_important_scores.append(tmp)
        # ---------------------------------------------------------------------------------------------------------------
        elif selection_category == 'simple': #OK
            tmp = all_tests['estimator'][iii]['arfs'].get_feature_names_out()
            all_features.append(tmp)
            all_important_scores.append([1] * len(tmp))
        # ---------------------------------------------------------------------------------------------------------------
        elif selection_category == 'simplepd':
            tmp = all_tests['estimator'][iii]['arfs'].get_feature_names_out()
            all_features.append(tmp)
            all_important_scores.append([1] * len(tmp))
        # ---------------------------------------------------------------------------------------------------------------
        elif selection_category == 'simplepdnu': #OK
            tmp = np.abs(all_tests['estimator'][iii]['arfs'].feature_importances_)
            all_important_scores.append(tmp)
            all_features.append(all_tests['estimator'][iii]['unique'].get_feature_names_out())
        else:
            print('wrong selection category')


    all_features_df = pd.DataFrame([np.concatenate(all_features)  ,  np.concatenate(all_important_scores)   ]).transpose()
    all_features_df.columns = ['feats_' , 'ranks_']

    feature_count = pd.DataFrame( all_features_df.groupby('feats_')['ranks_'].mean() )
    feature_count['counts_'] = all_features_df.groupby('feats_').count()


    return feature_count , all_tests, fair_results#feature_count.head(num_thres).index

####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

def set_internal_model(estimator_number):

    models = [
    #    RandomForestClassifier(n_estimators=estimator_number),
        DecisionTreeClassifier(random_state=0),
        #SelfPacedEnsembleClassifier(estimator=RandomForestClassifier(), n_estimators=estimator_number, random_state=0),
        #CatBoostClassifier(random_state=0, verbose=0),
        #SelfPacedEnsembleClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number,random_state=0),
        #AdaCostClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number, algorithm='SAMME', random_state=0),
    #    AdaUBoostClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number, algorithm='SAMME', random_state=0),
        #AsymBoostClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number, algorithm='SAMME', random_state=0),
        #CompatibleAdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number, algorithm='SAMME', random_state=0),
    #    CompatibleBaggingClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number, random_state=0),
     #   EasyEnsembleClassifier(n_estimators=estimator_number, sampling_strategy='auto', random_state=0, estimator=LogisticRegression(solver='liblinear', random_state=0,  penalty='l1')),
        #EasyEnsembleClassifier(n_estimators=estimator_number, sampling_strategy='auto', random_state=0, estimator=LogisticRegression(solver='liblinear', random_state=0,penalty='l2')),
        #EasyEnsembleClassifier(n_estimators=estimator_number, estimator=AdaBoostClassifier(n_estimators=100, algorithm='SAMME', random_state=0), sampling_strategy='auto',  random_state=0),
        #EasyEnsembleClassifier(n_estimators=estimator_number, estimator=GradientBoostingClassifier(n_estimators=100, random_state=0), sampling_strategy='auto', random_state=0),
        EasyEnsembleClassifier(n_estimators=estimator_number, estimator=DecisionTreeClassifier(random_state=0), sampling_strategy='auto', random_state=0),
        #BalancedBaggingClassifier(n_estimators=estimator_number, estimator=None, sampling_strategy='auto', random_state=0),
        BalancedRandomForestClassifier(n_estimators=estimator_number, sampling_strategy='auto', random_state=0),
    #    LogisticRegression(solver='liblinear', random_state=0, penalty='l1'),
        LogisticRegression(solver='saga', random_state=0, penalty='elasticnet', l1_ratio=0.5, verbose=0),
        ######LGBMClassifier(n_estimators=estimator_number, random_state=0),
        #RUSBoostClassifier(n_estimators=estimator_number, estimator=None, sampling_strategy='auto', random_state=0),
        XGBClassifier(n_estimators=estimator_number, estimator=None, importance_type='gain', random_state=0,eval_metric='aucpr'),
        HistGradientBoostingClassifier(random_state=0),
        ExtraTreeClassifier(random_state=0)
    ]

    models_name = [
     #   'RF',
        'DecTree',
        #'SelfPaceRF',
        #'CatBoost',
        #'SelfPaceDT',
        #'AdaCost',
     #   'AdaUBoostDT',
        #'AsymBoostDT',
        #'CompatibleAdaBoostDT',
    #    'CompatibleBaggingDT',
     #   'EasyEnsembleLogReg',
       # 'EasyEnsembleLogReg2',
       # 'EasyEnsembleAdaBoost',
       # 'EasyEnsembleGraBoost',
        'EasyEnsembleDecTree',
        #'BalancedBagging',
        'BalancedRandForest',
     #   'LogisticRegression',
        'LogisticRegressionElastic',
        #######'LGBM',
        #'RUSBoost',
        'XGB',
        'HistGrad',
        'ExtraTree'
    ]
    return models_name , models

###################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

def set_imblearn_selectors(n_estimators ):

    models = [#RandomForestClassifier(oob_score=False, n_estimators=n_estimators,random_state=0),
              DecisionTreeClassifier(random_state=0)           ]

    models_name = [#'RF',
        'DT']
    # ----------------------------------------------------------

    return models_name, models

##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

def set_arfs_boruta_selectors(n_estimators ):#, shape_of_data):

    models = [#RandomForestClassifier(oob_score=False, n_estimators=n_estimators,random_state=0),
              #DecisionTreeClassifier(random_state=0),
              CatBoostClassifier(random_state=0, n_estimators=n_estimators, verbose = False),
              #LGBMClassifier(random_state=0, n_estimators=n_estimators),
              #LightForestClassifier(n_feat=shape_of_data, n_estimators=n_estimators),
              XGBClassifier(random_state=0, verbosity=0, eval_metric="logloss", n_estimators=n_estimators)]

    models_name = [#'RF', #'DT',
     'CatBoost', #'LightFor',
     'XGB']
    # ----------------------------------------------------------

    return models_name, models
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

kappa_scorer = make_scorer(cohen_kappa_score)
auc_scorer = make_scorer(roc_auc_score)
F_measure_scorer = make_scorer(f1_score, pos_label=1)
specificity = make_scorer(recall_score, pos_label=0)
sensitivity = make_scorer(recall_score, pos_label=1)
PPV_score = make_scorer(precision_score, pos_label=1)
NPV_score = make_scorer(precision_score, pos_label=0)
MCC = make_scorer(matthews_corrcoef)
average_PR_score = scorer = make_scorer(average_precision_score)

custom_scoring = {'acc': 'accuracy',
                  'roc': auc_scorer,
                  'sens': sensitivity,
                  'spec': specificity,
                  'Fscore': F_measure_scorer,
                  'kappa': kappa_scorer,
                  'balanced': 'balanced_accuracy',
                  'PPV': PPV_score,
                  'NPV': NPV_score,
                  'MCC': MCC,
                  'AverPR_score': average_PR_score
                  }

####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
def GMAN_FEAT_SEL_PARALLEL( comb_step, n_estimators , max_iter , perc , num_thres, pipeline_scaling, skf, data_for_fair_ml):

    tmp_all_results = []
    all_features_selected = []
    selected_pipeline = []

    y_train = comb_step[0][1]['pCR']

    #leave patient_id for fair analysis below
    train = comb_step[0][1].drop('pCR', axis=1)

    df_columns = ['Accuracy_Mean', 'Sensitivity_Mean', 'Specificity_Mean', 'ROC_Mean', 'F1_Mean', 'Kappa_Mean', 'PPV_Mean',
               'NPV_Mean', 'MCC_Mean', 'PrecRecall_Mean', 'Balanced_Acc_Mean' , 'Accuracy_Std', 'Sensitivity_Std', 'Specificity_Std',
               'ROC_Std', 'F1_Std', 'Kappa_Std', 'PPV_Std',
               'NPV_Std', 'MCC_Std', 'PrecRecall_Std', 'Balanced_Acc_Std',
                'EqualOddsRatio_Mean', 'DemoParityRatio_Mean', 'EqualOddsRatio_Std', 'DemoParityRatio_Std']

    #all_classifier_results = pd.DataFrame(columns = df_columns)

    # # ------------------------------------------------------------------------------------------------------------------
    tmp_name = comb_step[2][0] + '__' + comb_step[2][1] + '__' + comb_step[1][0]
    print(tmp_name)

    try:
        if comb_step[2][0] == 'boruta':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'boruta', pipeline_scaling, perc, n_estimators,max_iter)
        elif comb_step[2][0] == 'arfs':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'arfs', pipeline_scaling, perc, n_estimators,max_iter)
        elif comb_step[2][0] == 'simple':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'simple', pipeline_scaling, perc, n_estimators,max_iter)
        elif comb_step[2][0] == 'simplepd':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'simple_panda', pipeline_scaling, perc,n_estimators, max_iter)
        elif comb_step[2][0] == 'simplepdnu':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'simple_panda_y_numpy', pipeline_scaling, perc,n_estimators, max_iter)
        elif comb_step[2][0] == 'catboost':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'catboost', pipeline_scaling,perc, n_estimators, max_iter)
        elif comb_step[2][0] == 'compada':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'compada', pipeline_scaling,perc, n_estimators, max_iter)
        elif comb_step[2][0] == 'asymboost':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'asymboost', pipeline_scaling, perc, n_estimators,max_iter)
        elif comb_step[2][0] == 'adacost':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'adacost', pipeline_scaling, perc, n_estimators,max_iter)
        elif comb_step[2][0] == 'rusboost':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'rusboost', pipeline_scaling, perc, n_estimators,max_iter)
        elif comb_step[2][0] == 'selfpace':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'selfpace', pipeline_scaling, perc, n_estimators,max_iter)
        else:
            print('PROBLEMMMMMMM')

        data_for_fair_ml = data_for_fair_ml[data_for_fair_ml['patient_id'].isin(train['patient_id'])]
        train = train.drop('patient_id', axis=1)
        data_for_fair_ml = data_for_fair_ml.drop('patient_id', axis=1)

        tmp3 = return_feature_importance(pipe_set, train, y_train, skf, num_thres, comb_step[2][0], comb_step[1][0], data_for_fair_ml)

        fair_results = pd.DataFrame(tmp3[2], columns=['equal_odds_ratio', 'demo_par_ratio'])

        tmp2 = return_cross_validate_performance(tmp3[1])
        tmp2.insert(0, 'Model', tmp_name)

        ###add mean std of fair_results
        tmp4 = fair_results.describe().loc[["mean", "std"]]
        tmp4 = pd.DataFrame([tmp4['equal_odds_ratio']['mean'], tmp4['demo_par_ratio']['mean'], tmp4['equal_odds_ratio']['std'], tmp4['demo_par_ratio']['std'] ]).transpose()
        tmp4.columns= ['EqualOddsRatio_Mean', 'DemoParityRatio_Mean', 'EqualOddsRatio_Std', 'DemoParityRatio_Std']

        tmp2 = pd.concat([tmp2, tmp4.round(4)], axis=1)

        all_features_selected.append(tmp3[0])
        print('DONE:----------' + tmp_name + '-------------')

        #pipe_set.fit(train, y_train)
        #ttt = pipe_set['missing'].transform(train)
        #ttt = pipe_set['collinear'].transform(ttt)
        #ttt = pipe_set['imputer'].transform(ttt)
        #ttt = pipe_set['unique'].transform(ttt)
        #ttt = pipe_set['arfs'].transform(ttt)

    except:
        pipe_set = nan
        tmp2 = pd.DataFrame(np.nan, index=[0], columns=df_columns )
        tmp3 = pd.DataFrame(np.nan, index=[0], columns=['feats_', 'ranks_', 'counts_'])

        tmp2.insert(0, 'Model', tmp_name)
        tmp2['Model'] = tmp_name
        all_features_selected.append(tmp3)
        print("An exception occurred" + ' ---------- ' + tmp_name + ' -------------')

    tmp_all_results.append(tmp2)
    selected_pipeline.append(pipe_set)

    tmp_all_results = pd.concat(tmp_all_results)

    return tmp_all_results , all_features_selected , selected_pipeline
####################################################################################################################

#Equalized odds : equalized odds can be thought of as a stricter definition of fairness.
#For example, it would be the percentage of unqualified people who received a job offer

#Demographic parity promotes “blind fairness,” where similar individuals are treated the same, irrespective of their demographics
#Equalized odds ensures that prediction errors are distributed equally across different groups
#Equalized odds is often harder to achieve than demographic parity but provides a stronger group fairness metric.

def return_overall_train_performance(pkl_from_train) :
    all_classifier_results = pd.DataFrame(
        columns=['Accuracy_Mean', 'Sensitivity_Mean', 'Specificity_Mean', 'ROC_Mean', 'F1_Mean', 'Kappa_Mean',
                 'PPV_Mean',
                 'NPV_Mean', 'MCC_Mean', 'PrecRecall_Mean', 'Balanced_Acc_Mean', 'Accuracy_Std', 'Sensitivity_Std',
                 'Specificity_Std',
                 'ROC_Std', 'F1_Std', 'Kappa_Std', 'PPV_Std',
                 'NPV_Std', 'MCC_Std', 'PrecRecall_Std', 'Balanced_Acc_Std', 'EqualOddsRatio_Mean', 'DemoParityRatio_Mean', 'EqualOddsRatio_Std', 'DemoParityRatio_Std'])


    train_results = []
    for i in range(len(pkl_from_train)):
        #print(  str(i) + '_____ '   +  str(len(pkl_from_train[i][0].columns))   )
        print( str(i) +   '::::::'     +  pkl_from_train[i][0].columns )
        train_results.append(pkl_from_train[i][0])

    ndf = pd.concat(train_results)
    PERFORMANCE_ALL = ndf.groupby("Model").mean()
    PERFORMANCE_ALL.columns = ['MEAN_of_' + str(col)  for col in PERFORMANCE_ALL.columns]
    #-------------------------------------------------------------

    selected_features = []
    tmp_pipeline = []
    for ii in range(len(train_results)): #outer iteration
        tmp = pkl_from_train[ii][1][0]
        tmp['Model'] = pkl_from_train[ii][0]['Model'][0]
        tmp_pipeline.append(tmp)



    eee2 = pd.concat(tmp_pipeline)
    eee2['feat_name'] = eee2.index

    jjj = eee2.groupby(['Model','feat_name'])['counts_'].sum()
    jjj2 = eee2.groupby(['Model','feat_name'])['ranks_'].mean()

    selected_features = pd.concat([jjj , jjj2], axis=1).reset_index()
    selected_features = selected_features[selected_features['feat_name'] != 0]

    return selected_features , PERFORMANCE_ALL


####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

def my_main_funct(selection_type , treatment, pipeline_scaling_, num_thres, select_modalities, path_train_test):
    num_thres = int(num_thres)
    num_thres_to_write = num_thres
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # STEP #1 Import Data

    if pipeline_scaling_ == 'norm':
        pipeline_scaling = 'norm'
        pipe_write = 'norm'
        print('-------------------  NORM ---------------------')
    else:
        pipeline_scaling = 'no_norm'
        pipe_write = 'no_norm'

    main_path_pkl = os.getcwd() + '/'#'/home/gman/PycharmProjects/SklearnML/'  #'/mimer/NOBACKUP/groups/foukakis_ai/manikis/Python/hande_ml/'
    path_to_write = path_train_test + '/model_results/'
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    #STEP #2 Import train/test splits + stratified CV

    skf = read_pickle(main_path_pkl + 'skf_folds.pkl')

    if treatment == '1':
        drug_name = 'T-DM1'
        vect_for_fair_input = 'Clin_ER'
    elif treatment == '0':
        drug_name = 'DHP'
        vect_for_fair_input = 'Clin_ER'
    elif treatment == '2':
        drug_name = 'BOTH'
        vect_for_fair_input = 'Clin_Arm'
    else:
        print('ERROR TREATMENT SELECTED')

    model_train_vect = read_pickle(main_path_pkl + 'model_train_vect__' + drug_name +'.pkl')
    external_test_vect = read_pickle(main_path_pkl + 'external_test_vect__' + drug_name +'.pkl')

    vect_for_fair = [vect_for_fair_input]
    vect_for_fair.append('patient_id')
    tmp = pd.concat( [model_train_vect[0] , external_test_vect[0]] )
    data_for_fair_ml = pd.DataFrame( tmp[vect_for_fair]   )
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # STEP #3 Input Arguments

    n_estimators = 100
    max_iter = 100
    perc = 90
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # SELECT FEATURES TO RUN THE MODELS
    if select_modalities == 'clinical':                                                 #4 variables + ARM
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith('Clin_')] + list(['pCR','patient_id'])
    elif select_modalities == 'DNA':
        selected_columns = [column for column in model_train_vect[0].columns if         #52 variables
                            column.startswith('DNA_')] + list(['pCR','patient_id'])
    elif select_modalities == 'RNA':
        selected_columns = [column for column in model_train_vect[0].columns if         #52 variables
                            column.startswith('RNA_')] + list(['pCR','patient_id'])
    elif select_modalities == 'Image':
        selected_columns = [column for column in model_train_vect[0].columns if         #3 variables
                            column.startswith('WSI_')] + list(['pCR','patient_id'])
    elif select_modalities == 'Proteomics':
        selected_columns = [column for column in model_train_vect[0].columns if         #22 variables
                            column.startswith('Prot_')] + list(['pCR','patient_id'])
    elif select_modalities == 'run1':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith(('Clin_', 'DNA_'))] + list(['pCR','patient_id'])
    elif select_modalities == 'run2':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith(('Clin_', 'DNA_', 'RNA_'))] + list(['pCR','patient_id'])
    elif select_modalities == 'run3':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith(('Clin_', 'DNA_', 'RNA_', 'Prot_'))] + list(['pCR','patient_id'])
    elif select_modalities == 'run4':
        selected_columns = model_train_vect[0].columns
    elif select_modalities == 'simple':
        selected_columns = ['DNA_ERBB2_CNA', 'RNA_mRNA-ERBB2' , 'Prot_ERBB2', 'pCR','patient_id']
    else:
        print('nothing to run')

    if num_thres>=(len(selected_columns)-2):
        num_thres_to_write = len(selected_columns) - 2
        if select_modalities == 'simple':
            num_thres = (len(selected_columns) - 2 -1)
        else:
            num_thres = (len(selected_columns) - 2)

    for i in range(len(model_train_vect)):
        model_train_vect[i] = model_train_vect[i][selected_columns]

    for i in range(len(external_test_vect)):
        external_test_vect[i] = external_test_vect[i][selected_columns]


    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # STEP #4 set all train/test splits, selectors, classifiers

    # train test splits
    list_of_train_tests = []
    for i in range(len(model_train_vect)):
        #### test never used pass an empty df
        list_of_train_tests.append( ( 'Run' + str(i) ,model_train_vect[i],  pd.DataFrame([])  ))   #####################external_test_vect[i]))

########################################################################################################################
    #----- classifiers
    model_names_int, models_int = set_internal_model(n_estimators)  # estimators as argument

    list_of_classifiers = []
    for i in range(len(model_names_int)):
        list_of_classifiers.append((model_names_int[i] ,models_int[i]))
    list_of_classifiers_to_store = list_of_classifiers
    model_names_int, models_int = set_arfs_boruta_selectors(n_estimators )#, list_of_train_tests[0][1].shape[1])
    ########################################################################################################################
    # ------ arfs selectors
    list_of_arfs_selectors = []
    for i in range(len(model_names_int)):
        list_of_arfs_selectors.append(('arfs',model_names_int[i], models_int[i]))
    ########################################################################################################################
    # ------ boruta selectors
    list_of_boruta_selectors = []
    for i in range(len(model_names_int)):
        list_of_boruta_selectors.append(('boruta', model_names_int[i], models_int[i]))
    ########################################################################################################################

    model_names_int, models_int = set_imblearn_selectors(n_estimators)  # , list_of_train_tests[0][1].shape[1])

    # ------ rusboost selectors
    list_of_rusboost_selectors = []
    for i in range(len(model_names_int)):
        list_of_rusboost_selectors.append(('rusboost', model_names_int[i], models_int[i]))
    ########################################################################################################################
    # ------ selfpace selectors
    list_of_selfpace_selectors = []
    for i in range(len(model_names_int)):
        list_of_selfpace_selectors.append(('selfpace', model_names_int[i], models_int[i]))
    ########################################################################################################################
    # ------ compada selectors
    list_of_compada_selectors = []
    for i in range(len(model_names_int)):
        list_of_compada_selectors.append(('compada', model_names_int[i], models_int[i]))
    ########################################################################################################################
    # ------ adacost selectors
    list_of_adacost_selectors = []
    for i in range(len(model_names_int)):
        list_of_adacost_selectors.append(('adacost', model_names_int[i], models_int[i]))
    ########################################################################################################################
    # ------ asymboost selectors
    list_of_asymboost_selectors = []
    for i in range(len(model_names_int)):
        list_of_asymboost_selectors.append(('asymboost', model_names_int[i], models_int[i]))
    ########################################################################################################################
    # ------ catboost selectors
    list_of_catboost_selectors = []
    for i in range(len(model_names_int)):
        list_of_catboost_selectors.append(('catboost', model_names_int[i], models_int[i]))
    ########################################################################################################################


    # ------ simple selectors
    simple_models = [SelectFromModel(threshold=None, estimator=LinearSVC(penalty='l1', random_state=0)),
                    # SelectFromModel(threshold=None, estimator=Lasso(alpha=0.5, random_state=0)),
                     SelectFromModel(threshold=None, estimator=LogisticRegression(solver='liblinear', random_state=0, penalty='l1')),
                     #SelectFromModel(threshold=None, estimator=AdaBoostRegressor(estimator=None, random_state=0,
                     #                                                            n_estimators=n_estimators,
                     #                                                            learning_rate=1.0, loss='linear')),
                     SelectKBest(score_func=f_classif, k=num_thres)
        ]

    simple_models_names = ['SelectFrom_LinearSVC',
                           #'SelectFrom_Lasso',
                           'SelectFrom_LogRegr',
                      #     'SelectFrom_AdaBoostRegr',
                           'SelectKBest_F'
        ]

    list_of_simple_selectors = []
    for i in range(len(simple_models_names)):
        list_of_simple_selectors.append(('simple', simple_models_names[i], simple_models[i]))
    ########################################################################################################################
    # ------ simple with panda selectors

    simple_models_with_panda = [DropHighPSIFeatures(split_col=None, threshold=0.1),
                                SmartCorrelatedSelection(variables=None, method='spearman', threshold=0.2,
                                                         selection_method="variance")]
    simple_models_with_panda_names = ['DropHighPSI' , 'SmartCorr']

    list_of_simple_panda_selectors = []
    for i in range(len(simple_models_with_panda_names)):
        list_of_simple_panda_selectors.append(('simplepd', simple_models_with_panda_names[i], simple_models_with_panda[i]))
    ########################################################################################################################
    # ------ simple with panda y_numpy selectors
    simple_models_with_panda_y_numpy = [ReliefF(n_features_to_select=num_thres, n_neighbors=10),
                                        #SURF(n_features_to_select=num_thres),
                                        #SURFstar(n_features_to_select=num_thres),
                                        MultiSURF(n_features_to_select=num_thres)#,
                                        #MultiSURFstar(n_features_to_select=num_thres)
                                    ]

    simple_models_with_panda_y_numpy_names = ['Relief' , #'Surf' ,
                                              #'SurfStar' ,
                                              'MultiSURF'# , 'MultiSURFstar'
                                               ]

    list_of_simple_panda_numpy_selectors = []
    for i in range(len(simple_models_with_panda_y_numpy_names)):
        list_of_simple_panda_numpy_selectors.append(
            ('simplepdnu', simple_models_with_panda_y_numpy_names[i], simple_models_with_panda_y_numpy[i]))

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # STEP #5 select list of selectors, classifiers
    list_final_selectors = []

    #round #1 of combinations wwhich they have a model for prediction
    if selection_type == 'boruta' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_boruta_selectors
    if selection_type == 'arfs' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_arfs_selectors
    if selection_type == 'simple' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_simple_selectors + list_of_simple_panda_selectors + list_of_simple_panda_numpy_selectors

    combinations = list(itertools.product(list_of_train_tests, list_of_classifiers, list_final_selectors))

    list_final_selectors = []

    # round #2 of combinations wwhich they DONT have a model for prediction
    if selection_type == 'catboost' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_catboost_selectors
    if selection_type == 'asymboost' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_asymboost_selectors
    if selection_type == 'adacost' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_adacost_selectors
    if selection_type == 'compada' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_compada_selectors
    if selection_type == 'selfpace' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_selfpace_selectors
    if selection_type == 'rusboost' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_rusboost_selectors

    list_of_classifiers = []
    list_of_classifiers.append(('NONE', DummyClassifier()))

    combinations = combinations +  list(itertools.product(list_of_train_tests , list_of_classifiers , list_final_selectors))

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # STEP #6 run parallel
    new_list1 = []
    new_list1.extend([n_estimators for i in range(len(combinations))])

    new_list2 = []
    new_list2.extend([max_iter for i in range(len(combinations))])

    new_list3 = []
    new_list3.extend([perc for i in range(len(combinations))])

    new_list4 = []
    new_list4.extend([num_thres for i in range(len(combinations))])

    new_list5 = []
    new_list5.extend([pipeline_scaling for i in range(len(combinations))])

    new_list6 = []
    new_list6.extend([skf for i in range(len(combinations))])

    new_list7 = []
    new_list7.extend([data_for_fair_ml for i in range(len(combinations))])


    #data_for_fair_ml[data_for_fair_ml['patient_id'].isin(model_train_vect[i].index)]
    #GMAN_FEAT_SEL_PARALLEL(comb_step, n_estimators, max_iter, perc, num_thres, pipeline_scaling, vect_for_fair)

    aaa_all = Parallel(n_jobs=-1)(
        delayed(GMAN_FEAT_SEL_PARALLEL)(i, j, k, l, m, n, o, p) for i, j, k, l, m, n, o, p in
        zip(combinations, new_list1, new_list2, new_list3,new_list4, new_list5, new_list6, new_list7))

    if pipeline_scaling == 'norm':
        pipe_write = 'norm'
    else:
        pipe_write = 'no_norm'

    if treatment == '1' or treatment == '0':
        analysis_case = selection_type + '_' + pipe_write + '_TREAT_' + str(treatment)
        print(analysis_case)
    else:
        analysis_case = selection_type + '_' + pipe_write
        print('*********')

    print('**************  FINISHED ******************')

    output = open(path_to_write +  analysis_case + '_selfeats_' + str(num_thres_to_write) + '_' + select_modalities + '.pkl', 'wb')
    pickle.dump(aaa_all, output)
    output.close()

    output = open(path_to_write + 'list_of_classifiers_to_store.pkl', 'wb')
    pickle.dump(list_of_classifiers_to_store, output)
    output.close()

#####    joblib.dump(clf, 'radiomics_clinical_model.pkl')


if __name__ == '__main__':
    my_main_funct(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
