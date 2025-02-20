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
        print('important')
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

    #pipeline for arfs------------------------------------------------------------------------------------------------------
    if selection_category == 'arfs':
        if use_scaling:
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

    # pipeline for boruta------------------------------------------------------------------------------------------------------
    elif selection_category == 'boruta':
        if use_scaling:
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
        if use_scaling:
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
        if use_scaling:
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                                         ('collinear', CollinearityThreshold(threshold=coll_thres)),
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ('trasform', StandardScaler().set_output(transform='pandas')),
                                        # ("unpandarizer", FunctionTransformer(unpanda)),
                                         ("unpandarizer", CustomTransformer()),
                                         ("arfs", feat_selector),
                                         ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
                                         ('collinear', CollinearityThreshold(threshold=coll_thres)),
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         #("unpandarizer", FunctionTransformer(unpanda)),
                                         ("unpandarizer", CustomTransformer()),
                                         ("arfs", feat_selector),
                                         ('model', model_internal)])

    # pipeline for simple FS with pd to numpy transform-  +   y_train_to_numpy() argotera------------------------------------------
    elif selection_category == 'simple_panda_y_numpy':
        if use_scaling:
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

#selection_category = 'arfs'   'boruta'  'simple'
#return_feature_importance(pipe_set, train, y_train, skf, num_thres, comb_step[2][0], comb_step[1][0])
#selection_category = comb_step[2][0]
#model_name = comb_step[1][0]

def return_feature_importance(pipe_set, train, y_train, skf, num_thres, selection_category, model_name ):

    ####selection_category   --> arfs / boruta / simple / simplepd / simplepdnu
    all_results = pd.DataFrame(columns = ['feats' , 'rank'])
    all_features = []
    all_important_scores = []

    if selection_category == 'simplepdnu':
        all_tests = cross_validate(pipe_set, train, y_train.values, cv=skf, scoring=custom_scoring,return_estimator=True)
    else:
        all_tests = cross_validate(pipe_set, train, y_train, cv=skf, scoring=custom_scoring, return_estimator=True)

    for iii in range(len(all_tests['estimator'])):
        if selection_category == 'arfs':
            tmp = all_tests['estimator'][iii]['arfs'].get_feature_names_out()
            all_features.append(tmp)
        elif selection_category == 'boruta':
            tmp = all_tests['estimator'][iii]['arfs'].get_important_feats()
            all_features.append(tmp)
        elif selection_category == 'simple':
            tmp = all_tests['estimator'][iii]['arfs'].get_feature_names_out()
            all_features.append(tmp)
        elif selection_category == 'simplepd':
            tmp = all_tests['estimator'][iii]['arfs'].get_feature_names_out()
            tmp2 = PandasFromArray(tmp).fit(train).transform(train)
            all_features.append(tmp2.columns)
        elif selection_category == 'simplepdnu':
            tmp = all_tests['estimator'][iii]['arfs'].top_features_
            all_features.append(train.columns[tmp[range(num_thres)]])

        else:
            print('wrong selection category')

#------------------------------------------------------------------------------------------------------------------------------------------------
        if model_name in {'HistGrad', 'CompatibleBaggingClassifier', 'CompatibleBaggingDT','BalancedBagging',
                          'EasyEnsembleLogReg','EasyEnsembleLogReg2','EasyEnsembleAdaBoost','EasyEnsembleGraBoost','EasyEnsembleDecTree',}:
            all_important_scores.append([1] * len(tmp))
        elif model_name in {'LogisticRegression', 'LogisticRegressionElastic'}:
            all_important_scores.append(np.transpose(all_tests['estimator'][iii]['model'].coef_))
        else:
            all_important_scores.append(all_tests['estimator'][iii]['model'].feature_importances_)
# ------------------------------------------------------------------------------------------------------------------------------------------------

    all_features_df = pd.DataFrame([np.concatenate(all_features)  ,  np.concatenate(all_important_scores)   ]).transpose()
    all_features_df.columns = ['feats_' , 'ranks_']

    feature_count = pd.DataFrame( all_features_df.groupby('feats_')['ranks_'].mean() )
    feature_count['counts_'] = all_features_df.groupby('feats_').count()


    return feature_count , all_tests#feature_count.head(num_thres).index

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
        SelfPacedEnsembleClassifier(estimator=RandomForestClassifier(), n_estimators=estimator_number, random_state=0),
        CatBoostClassifier(random_state=0, verbose=0),
        SelfPacedEnsembleClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number,random_state=0),
        AdaCostClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number, algorithm='SAMME', random_state=0),
    #    AdaUBoostClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number, algorithm='SAMME', random_state=0),


        AsymBoostClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number, algorithm='SAMME', random_state=0),
        CompatibleAdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number, algorithm='SAMME', random_state=0),
    #    CompatibleBaggingClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=estimator_number, random_state=0),
     #   EasyEnsembleClassifier(n_estimators=estimator_number, sampling_strategy='auto', random_state=0, estimator=LogisticRegression(solver='liblinear', random_state=0,  penalty='l1')),
        EasyEnsembleClassifier(n_estimators=estimator_number, sampling_strategy='auto', random_state=0, estimator=LogisticRegression(solver='liblinear', random_state=0,penalty='l2')),
        EasyEnsembleClassifier(n_estimators=estimator_number, estimator=AdaBoostClassifier(n_estimators=100, algorithm='SAMME', random_state=0), sampling_strategy='auto',  random_state=0),
        EasyEnsembleClassifier(n_estimators=estimator_number, estimator=GradientBoostingClassifier(n_estimators=100, random_state=0), sampling_strategy='auto', random_state=0),
        EasyEnsembleClassifier(n_estimators=estimator_number, estimator=DecisionTreeClassifier(random_state=0), sampling_strategy='auto', random_state=0),
        BalancedBaggingClassifier(n_estimators=estimator_number, estimator=None, sampling_strategy='auto', random_state=0),
        BalancedRandomForestClassifier(n_estimators=estimator_number, sampling_strategy='auto', random_state=0),
    #    LogisticRegression(solver='liblinear', random_state=0, penalty='l1'),
        LogisticRegression(solver='saga', random_state=0, penalty='elasticnet', l1_ratio=0.5),
        ######LGBMClassifier(n_estimators=estimator_number, random_state=0),
        RUSBoostClassifier(n_estimators=estimator_number, estimator=None, sampling_strategy='auto', random_state=0),
        XGBClassifier(n_estimators=estimator_number, estimator=None, importance_type='gain', random_state=0,eval_metric='aucpr'),
        HistGradientBoostingClassifier(random_state=0),
        ExtraTreeClassifier(random_state=0)
    ]

    models_name = [
     #   'RF',
        'SelfPaceRF',
        'CatBoost',
        'SelfPaceDT',
        'AdaCost',
     #   'AdaUBoostDT',
        'AsymBoostDT',
        'CompatibleAdaBoostDT',
    #    'CompatibleBaggingDT',
     #   'EasyEnsembleLogReg',
        'EasyEnsembleLogReg2',
        'EasyEnsembleAdaBoost',
        'EasyEnsembleGraBoost',
        'EasyEnsembleDecTree',
        'BalancedBagging',
        'BalancedRandForest',
     #   'LogisticRegression',
        'LogisticRegressionElastic',
        #######'LGBM',
        'RUSBoost',
        'XGB',
        'HistGrad',
        'ExtraTree'
    ]
    return models_name , models

###################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

def set_arfs_boruta_selectors(n_estimators , shape_of_data):

    models = [#RandomForestClassifier(oob_score=False, n_estimators=n_estimators),
              CatBoostClassifier(random_state=0, n_estimators=n_estimators),
              ######LGBMClassifier(random_state=0, n_estimators=n_estimators),
              LightForestClassifier(n_feat=shape_of_data, n_estimators=n_estimators),
              XGBClassifier(random_state=0, verbosity=0, eval_metric="logloss", n_estimators=n_estimators)]

    models_name = [#'RF',
     'CatBoost', 'LightFor', 'XGB']
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


def GMAN_FEAT_SEL_PARALLEL( comb_step, n_estimators , max_iter , perc , num_thres, pipeline_scaling, skf):

    tmp_all_results = []
    all_features_selected = []
    selected_pipeline = []

    #train is in combinations[]
    y_train = comb_step[0][1]['pCR']
    train = comb_step[0][1].drop(['pCR'], axis=1)

    all_classifier_results = pd.DataFrame(columns = ['Accuracy_Mean', 'Sensitivity_Mean', 'Specificity_Mean', 'ROC_Mean', 'F1_Mean', 'Kappa_Mean', 'PPV_Mean',
               'NPV_Mean', 'MCC_Mean', 'PrecRecall_Mean', 'Balanced_Acc_Mean' , 'Accuracy_Std', 'Sensitivity_Std', 'Specificity_Std',
               'ROC_Std', 'F1_Std', 'Kappa_Std', 'PPV_Std',
               'NPV_Std', 'MCC_Std', 'PrecRecall_Std', 'Balanced_Acc_Std'])

    # # ------------------------------------------------------------------------------------------------------------------
    tmp_name = comb_step[2][0] + '__' + comb_step[2][1] + '__' + comb_step[1][0]
    print(tmp_name)
    
    try:
        if comb_step[2][0] == 'boruta':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'boruta', pipeline_scaling, perc,n_estimators, max_iter)
        if comb_step[2][0] == 'arfs':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'arfs', pipeline_scaling, perc, n_estimators,max_iter)
        if comb_step[2][0] == 'simple':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'simple', pipeline_scaling, perc, n_estimators, max_iter)
        if comb_step[2][0] == 'simplepd':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'simple_panda', pipeline_scaling, perc, n_estimators, max_iter)
        if comb_step[2][0] == 'simplepdnu':
            pipe_set = set_pipeline(comb_step[2][2], comb_step[1][1], 'simple_panda_y_numpy', pipeline_scaling, perc, n_estimators, max_iter)


        tmp3 = return_feature_importance(pipe_set, train, y_train, skf, num_thres, comb_step[2][0], comb_step[1][0])

        tmp2 = return_cross_validate_performance(tmp3[1])
        tmp2.insert(0, 'Model', tmp_name)
        all_features_selected.append(tmp3[0])
        print('DONE:----------' + tmp_name + '-------------')

    except:
        pipe_set = nan
        tmp2 = pd.DataFrame(np.nan, index=[0], columns=all_classifier_results.columns)
        tmp3 = pd.DataFrame(np.nan, index=[0], columns=['feats_', 'ranks_', 'counts_'])
        tmp2['Model'] = tmp_name
        all_features_selected.append(tmp3)
        print("An exception occurred")

    tmp_all_results.append(tmp2)
    selected_pipeline.append(pipe_set)

    tmp_all_results = pd.concat(tmp_all_results)

    return tmp_all_results , all_features_selected , selected_pipeline
####################################################################################################################


def return_overall_train_performance(pkl_from_train) :
    all_classifier_results = pd.DataFrame(
        columns=['Accuracy_Mean', 'Sensitivity_Mean', 'Specificity_Mean', 'ROC_Mean', 'F1_Mean', 'Kappa_Mean',
                 'PPV_Mean',
                 'NPV_Mean', 'MCC_Mean', 'PrecRecall_Mean', 'Balanced_Acc_Mean', 'Accuracy_Std', 'Sensitivity_Std',
                 'Specificity_Std',
                 'ROC_Std', 'F1_Std', 'Kappa_Std', 'PPV_Std',
                 'NPV_Std', 'MCC_Std', 'PrecRecall_Std', 'Balanced_Acc_Std'])


    train_results = []
    for i in range(len(pkl_from_train)):
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

####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
#selection_type = 'simple'  #simple, boura, arfs, all
#treatment = 1   # 0  1   or nan
#pipeline_scaling = False
#mimer/NOBACKUP/groups/foukakis_ai/manikis/Python/hande_ml/fusion/curated_metrics/clin_multiomics_curated_metrics_PREDIX_HER2.txt
#num_thres = 10
#select_modalities :
#run1 : Clin + DNA
#run2 : clin +DNA +RNA
#run3 : clin+DNA+RNA+prot
#run4 : clin+DNA+RNA+prot + image

def my_main_funct(selection_type , treatment, pipeline_scaling_, num_thres, select_modalities, path_train_test , path_to_write):
    num_thres = int(num_thres)
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # STEP #1 Import Data

    if pipeline_scaling_ == 'norm':
        pipeline_scaling = True
        pipe_write = 'norm'
        print('-------------------  NORM ---------------------')
    else:
        pipeline_scaling = False
        pipe_write = 'no_norm'

    main_path_pkl = path_train_test#'/home/gman/PycharmProjects/SklearnML/'  #'/mimer/NOBACKUP/groups/foukakis_ai/manikis/Python/hande_ml/'
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    #STEP #2 Import train/test splits + stratified CV

    skf = read_pickle(main_path_pkl + 'skf_folds.pkl')
    model_train_vect = read_pickle(main_path_pkl + 'model_train_vect.pkl')
    external_test_vect = read_pickle(main_path_pkl + 'external_test_vect.pkl')

    ####################################################################################################################
    if treatment == '1' or treatment == '0':
        print('TREATMENT SELECTED')
        for i in range(len(external_test_vect)):
            external_test_vect[i] = external_test_vect[i][(external_test_vect[i]['Clin_Arm'] == int(treatment))]
            model_train_vect[i] = model_train_vect[i][(model_train_vect[i]['Clin_Arm'] == int(treatment))]

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
    if select_modalities == 'clinical':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith('Clin_')] + list(['pCR'])
    elif select_modalities == 'DNA':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith('DNA_')] + list(['pCR'])
    elif select_modalities == 'RNA':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith('RNA_')] + list(['pCR'])
    elif select_modalities == 'Image':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith('WSI_')] + list(['pCR'])
    elif select_modalities == 'Proteomics':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith('Prot_')] + list(['pCR'])
    elif select_modalities == 'run1':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith(('Clin_', 'DNA_'))] + list(['pCR'])
    elif select_modalities == 'run2':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith(('Clin_', 'DNA_', 'RNA_'))] + list(['pCR'])
    elif select_modalities == 'run3':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith(('Clin_', 'DNA_', 'RNA_', 'Prot_'))] + list(['pCR'])
    elif select_modalities == 'run4':
        selected_columns = model_train_vect[0].columns
    else:
        print('nothing to run')

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
        list_of_train_tests.append( ( 'Run' + str(i) ,model_train_vect[i],external_test_vect[i]))

########################################################################################################################
    #----- classifiers
    model_names_int, models_int = set_internal_model(n_estimators)  # estimators as argument

    list_of_classifiers = []
    for i in range(len(model_names_int)):
        list_of_classifiers.append((model_names_int[i] ,models_int[i]))
    ########################################################################################################################
    #------ arfs selectors
    model_names_int, models_int = set_arfs_boruta_selectors(n_estimators , list_of_train_tests[0][1].shape[1])

    list_of_arfs_selectors = []
    for i in range(len(model_names_int)):
        list_of_arfs_selectors.append(('arfs',model_names_int[i], models_int[i]))
    ########################################################################################################################
    # ------ boruta selectors
    list_of_boruta_selectors = []
    for i in range(len(model_names_int)):
        list_of_boruta_selectors.append(('boruta', model_names_int[i], models_int[i]))
    ########################################################################################################################
    # ------ simple selectors

    simple_models = [SelectFromModel(threshold=None, estimator=LinearSVC(penalty='l1', random_state=0)),
                    # SelectFromModel(threshold=None, estimator=Lasso(alpha=0.5, random_state=0)),
                     SelectFromModel(threshold=None, estimator=LogisticRegression(solver='liblinear', random_state=0, penalty='l1')),
                     SelectFromModel(threshold=None, estimator=AdaBoostRegressor(estimator=None, random_state=0,
                                                                                 n_estimators=n_estimators,
                                                                                 learning_rate=1.0, loss='linear')),
                     SelectKBest(score_func=f_classif, k=num_thres)
        ]

    simple_models_names = ['SelectFrom_LinearSVC',
                           #'SelectFrom_Lasso',
                           'SelectFrom_LogRegr',
                           'SelectFrom_AdaBoostRegr',
                           'SelectKBest_F'
        ]

    list_of_simple_selectors = []
    for i in range(len(simple_models_names)):
        list_of_simple_selectors.append(('simple', simple_models_names[i], simple_models[i]))
    ########################################################################################################################
    # ------ simple with panda selectors

    simple_models_with_panda = [DropHighPSIFeatures(split_col=None, threshold=0.25),
                                SmartCorrelatedSelection(variables=None, method='spearman', threshold=0.8,
                                                         selection_method="variance")]
    simple_models_with_panda_names = ['DropHighPSI' , 'SmartCorr']

    list_of_simple_panda_selectors = []
    for i in range(len(simple_models_with_panda_names)):
        list_of_simple_panda_selectors.append(('simplepd', simple_models_with_panda_names[i], simple_models_with_panda[i]))
    ########################################################################################################################
    # ------ simple with panda y_numpy selectors
    simple_models_with_panda_y_numpy = [ReliefF(n_features_to_select=num_thres, n_neighbors=10),
                                        SURF(n_features_to_select=num_thres),
                                        #SURFstar(n_features_to_select=num_thres),
                                        MultiSURF(n_features_to_select=num_thres)#,
                                        #MultiSURFstar(n_features_to_select=num_thres)
                                    ]

    simple_models_with_panda_y_numpy_names = ['Relief' , 'Surf' ,
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

    if selection_type == 'boruta' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_boruta_selectors
    if selection_type == 'arfs' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_arfs_selectors
    if selection_type == 'simple' or selection_type == 'all':
        list_final_selectors = list_final_selectors + list_of_simple_selectors + list_of_simple_panda_selectors + list_of_simple_panda_numpy_selectors

    combinations = list(itertools.product(list_of_train_tests , list_of_classifiers , list_final_selectors))

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

    aaa_all = Parallel(n_jobs=-1)(
        delayed(GMAN_FEAT_SEL_PARALLEL)(i, j, k, l, m, n, o) for i, j, k, l, m, n, o in
        zip(combinations, new_list1, new_list2, new_list3,new_list4, new_list5, new_list6))


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

    output = open(path_to_write +  analysis_case + '_selfeats_' + str(num_thres) + '_' + select_modalities + '.pkl', 'wb')
    pickle.dump(aaa_all, output)
    output.close()


if __name__ == '__main__':
    my_main_funct(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])

