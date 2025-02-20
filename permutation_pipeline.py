from sklearn.model_selection import train_test_split

import joblib
from joblib import Parallel, delayed
from sklearn.metrics import RocCurveDisplay
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
from itertools import product
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

from sklearn.model_selection import permutation_test_score

from sklearn import metrics

###############################################################################################from boruta import BorutaPy

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X = X.to_numpy()
        return X


def unpanda(df):
    if not isinstance(df, np.ndarray):
        df = df.to_numpy()
    return df


def read_pickle(filename):
    with open(filename, 'rb') as handle:
        dct = pickle.load(handle)

    return dct


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

    overall_results_out_std.append([round(all_tests['test_acc'].std(), 4),
                                    round(all_tests['test_sens'].std(), 4),
                                    round(all_tests['test_spec'].std(), 4),
                                    round(all_tests['test_roc'].std(), 4),
                                    round(all_tests['test_Fscore'].std(), 4),
                                    round(all_tests['test_kappa'].std(), 4),
                                    round(all_tests['test_PPV'].std(), 4),
                                    round(all_tests['test_NPV'].std(), 4),
                                    round(all_tests['test_MCC'].std(), 4),
                                    round(all_tests['test_AverPR_score'].std(),4),
                                    round(all_tests['test_balanced'].std(), 4)
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

#exoume ta eksis options gia ta parakatw:
#1) arfs: apo to antistoixo paketo. travaei ta 4 modela apo models
#2) bruta: travaei to diko tou pipeline. travaei ta 4 modela apo models
#3) simple: allazei to function pou epistrefei ta important feature names. travaei apo to simple_models
#4) simple_models_with_panda: o feature selector thelei numpy kai oxi df opote metasxhmatizw. Meta ta gurnaw se df sta importances
#5) simple_models_with_panda_y_numpy: idio me 4 alla thelei to y_train na einai to_numpy()!!!


def set_pipeline(feat_selector,model_internal, selection_category , use_scaling, perc, n_estimators, max_iter):

    #pipeline for arfs------------------------------------------------------------------------------------------------------
    if selection_category == 'arfs':
        if use_scaling:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=0.1)),  # delete columns having >10% missing
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),
                 ("arfs", arfsgroot.Leshy(feat_selector, perc=perc, n_estimators=n_estimators, verbose=1, alpha=0.05,max_iter=max_iter, random_state=0, importance="shap")),
                 ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=0.1)),  # delete columns having >10% missing
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ("arfs", arfsgroot.Leshy(feat_selector, perc=perc, n_estimators=n_estimators, verbose=1, alpha=0.05,max_iter=max_iter, random_state=0, importance="shap")),
                 ('model', model_internal)])

    # pipeline for boruta------------------------------------------------------------------------------------------------------
    elif selection_category == 'boruta':
        if use_scaling:
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=0.1)),  # delete columns having >10% missing
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ('trasform', StandardScaler().set_output(transform='pandas')),
                                         ('arfs', select_from_boruta(feat_selector)),
                                         ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=0.1)),  # delete columns having >10% missing
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ('arfs', select_from_boruta(feat_selector)),
                                         ('model', model_internal)])

    # pipeline for simple FS------------------------------------------------------------------------------------------------------
    elif selection_category == 'simple':
        if use_scaling:
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=0.1)),  # delete columns having >10% missing
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ('trasform', StandardScaler().set_output(transform='pandas')),
                                         ("arfs", feat_selector),
                                         ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=0.1)),  # delete columns having >10% missing
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ("arfs", feat_selector),
                                         ('model', model_internal)])

    # pipeline for simple FS with pd to numpy transform------------------------------------------------------------------------------
    elif selection_category == 'simple_panda':
        if use_scaling:
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=0.1)),  # delete columns having >10% missing
                                         ("imputer", PandasSimpleImputer()),
                                         ("unique", UniqueValuesThreshold(threshold=1)),
                                         ('trasform', StandardScaler().set_output(transform='pandas')),
                                        # ("unpandarizer", FunctionTransformer(unpanda)),
                                         ("unpandarizer", CustomTransformer()),
                                         ("arfs", feat_selector),
                                         ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline([("missing", MissingValueThreshold(threshold=0.1)),  # delete columns having >10% missing
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
                [("missing", MissingValueThreshold(threshold=0.1)),  # delete columns having >10% missing
                 ("imputer", PandasSimpleImputer()),
                 ("unique", UniqueValuesThreshold(threshold=1)),
                 ('trasform', StandardScaler().set_output(transform='pandas')),
                 #("unpandarizer", FunctionTransformer(unpanda)),
                 ("unpandarizer", CustomTransformer()),
                 ("arfs", feat_selector),
                 ('model', model_internal)])
        else:
            arfs_fs_pipeline = Pipeline(
                [("missing", MissingValueThreshold(threshold=0.1)),  # delete columns having >10% missing
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

#evaluate training
#pkl_from_train = aaa

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
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
####################################################################################################################


##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
######################   arguments to run a single step ################################################################
# out = new_list3_test.index('arfs__CatBoost__SelfPaceRF')
# train_from_vect = new_list1_test[out]
# test_from_vect = new_list2_test[out]
# model_name = new_list3_test[out]
# selected_features = new_list4_test[out]
# input_pipe_set = new_list5_test[out]
# num_of_feats = new_list6_test[out]
# cv_permute = new_list7_test[out]
# file_to_write  = new_list8_test[out]
# pipeline_scaling = new_list9_test[out]


def permutation_phase_PARALLEL(train_from_vect , test_from_vect , model_name  ,  selected_features, input_pipe_set, num_of_feats , cv_permute, file_to_write, pipeline_scaling ):

    perf_columns = ['ROC_AUC_Permute' , 'Pvalue']

    ###################################################################################################################
    ###########  SOS we merge train and test just for overall cohort performance indication used further for external validation
    train_from_vect = pd.concat([train_from_vect, test_from_vect], axis=0)
    ###################################################################################################################
    roc_data_all = []
    tmp_all_results = []
    feature_all_selected = []

    y_train = train_from_vect['pCR']
    train = train_from_vect.drop(['pCR'], axis=1)

    y_test = test_from_vect['pCR']
    test = test_from_vect.drop(['pCR'], axis=1)

    #####  STEP1: pick BEST FEATURES
    feats_to_run = selected_features

# SOS  to be sorted
    tmp_name = model_name
    feats_to_run_ = feats_to_run['feat_name'].iloc[0:num_of_feats]

    feature_all_selected.append(feats_to_run_.values)

    #--------------------------------------------------------------------------------------------
    #check if training has run............
    if feats_to_run['counts_'].count() > 0:
        try:
            train_selected = train[feats_to_run_]
            test_selected = test[feats_to_run_]

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
            pipe_set = input_pipe_set

            ###################################################
            ############################  remove the feature selection
            idx = [idx
                for idx, name in enumerate(pipe_set.named_steps.keys())
                if name == 'arfs'
            ][0]
            pipe_set.steps.pop(idx)

            ############################  remove the COLLINEARITY
            idx = [idx
                   for idx, name in enumerate(pipe_set.named_steps.keys())
                   if name == 'collinear'
                   ][0]
            pipe_set.steps.pop(idx)
            ##############################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
            #######################################################################################################
            #######################################################################################################
            print('START:----------' + tmp_name + '-------------')
            #print(pipe_set)
            score, permutation_scores, pvalue = permutation_test_score(pipe_set, train_selected, y_train, scoring= "roc_auc", random_state=0, cv = cv_permute)
            tmp = pd.DataFrame([score, pvalue]).transpose()
            print('DONE:----------' + tmp_name + '-------------')
            #######################################################################################################
            #######################################################################################################
            roc_data = plot_roc_curves(cv_permute, train_selected, y_train, pipe_set, file_to_write, model_name, pipeline_scaling  )
            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################
#             ix_training, ix_test = [], []
#             # Loop through each fold and append the training & test indices to the empty lists above
#             for fold in cv_permute.split(train_selected, y_train):
#                 ix_training.append(fold[0]), ix_test.append(fold[1])
#
#             SHAP_values_per_fold = []  # -#-#
#             ## Loop through each outer fold and extract SHAP values
#             for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)):  # -#-#
#                 # Verbose
#                 print('\n------ Fold Number:', i)
#                 X_train_, X_test_ = train_selected.iloc[train_outer_ix, :], train_selected.iloc[test_outer_ix, :]
#                 y_train_, y_test_ = y_train.iloc[train_outer_ix], y_train.iloc[test_outer_ix]
#
# #----------------------------------------------
#                 ######na prougithei transform
#                 model = pipe_set.named_steps['model']
#                 model.fit(X_train_, y_train_)
#                 # shap.initjs()
#                 ex = shap.KernelExplainer(model.predict, X_test_.values)
#                 shap_values = ex.shap_values(X_test_.values)
#                 shap.summary_plot(shap_values, X_test_)
#    #--------------------------------
#
#
#                 fit = pipe_set.fit(X_train_, y_train_)
#                 yhat = fit.predict(X_test)
#                 result = mean_squared_error(y_test, yhat)
#                 print('RMSE:', round(np.sqrt(result), 4))
#
#                 # Use SHAP to explain predictions
#                 explainer = shap.TreeExplainer(model)
#                 shap_values = explainer.shap_values(X_test)
#                 for SHAPs in shap_values:
#                     SHAP_values_per_fold.append(SHAPs)  # -#-#
            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################

            tmp.columns = perf_columns
            tmp.insert(0, 'Model', tmp_name)
            tmp_all_results.append(tmp)
            roc_data_all.append(roc_data)

        except:
            tmp = pd.DataFrame(np.nan, index=[0], columns=perf_columns)
            tmp['Model'] = tmp_name
            tmp_all_results.append(tmp)
            roc_data_all.append(tuple())
            print("An exception occurred")
    else:
        ##############   return nan performance
        tmp = pd.DataFrame(np.nan, index=[0], columns=perf_columns)
        tmp['Model'] = tmp_name
        tmp_all_results.append(tmp)

    tmp_all_results = pd.concat(tmp_all_results)
    return  tmp_all_results , feature_all_selected, roc_data_all
# https://towardsdatascience.com/feature-selection-with-boruta-in-python-676e3877e596

#https://medium.com/geekculture/boruta-feature-selection-explained-in-python-7ae8bf4aa1e7

#https://forum.numer.ai/t/feature-selection-with-borutashap/4145/5

####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
#file_to_write = main_path_pkl  + 'Testing_' + case_to_run
def plot_roc_curves(cv_permute, train_selected, y_train, pipe_set ,   file_to_write, model_name , pipeline_scaling):
    tprs = []
    aucs = []
    SHAP_values_per_fold = []
    SHAP_values_per_fold2 = []
    mean_fpr = np.linspace(0, 1, 100)
    input_data_all = []
    count_fig = 0
    test_score = []
    for i, (train, test) in enumerate(cv_permute.split(train_selected, y_train)):
        #print(i)
        if model_name.startswith('simplepdnu__'):
            pipe_set.fit(train_selected.iloc[train], y_train.iloc[train].values)
        else:
            pipe_set.fit(train_selected.iloc[train], y_train.iloc[train])

        test_score.append(pd.DataFrame(evaluate_model(pipe_set, train_selected.iloc[test], y_train.iloc[test])).transpose())
        #---------------------------------------------------------------------------------------------------------------
        viz = RocCurveDisplay.from_estimator(pipe_set,train_selected.iloc[test], y_train.iloc[test],
                             alpha=0.3, lw=8)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        #---------------------------------------------------------------------------------------------------------------

        ###### do the pre-processing first
        input_data = train_selected.iloc[test]
        input_data = pipe_set['missing'].transform(input_data)
        input_data = pipe_set['imputer'].transform(input_data)
        input_data = pipe_set['unique'].transform(input_data)

        if pipeline_scaling:
            input_data = pipe_set['trasform'].transform(input_data)

        input_data_all.append(input_data)

        ex = shap.KernelExplainer(pipe_set['model'].predict, input_data.values)
        shap_values = ex.shap_values(input_data)

        # fig, ax = plt.subplots(figsize=(25, 100))
        # shap.summary_plot(np.array(shap_values), input_data, show=False, class_names=pipe_set['model'].classes_)
        # plt.gcf().set_size_inches(25, 25)
        # plt.tight_layout()
        # plt.savefig(file_to_write + '_' + model_name +  str(count_fig) +'__SHAP_2.pdf', dpi=700)
        # # fig.savefig(file_to_write + '_' + model_name + '__SHAP.png', dpi=300)
        # plt.close()
        # count_fig = count_fig + 1

        for SHAPs in shap_values:
            SHAP_values_per_fold.append(SHAPs)

    #ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    #-------------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(25, 25))
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=8, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.legend(loc="lower right")
    #plt.show()
    fig.savefig(file_to_write + '_' + model_name + '__ROC_' + str(round(mean_auc,3)) + '_' + str(round(std_auc,3)) + '.png', dpi = 300)
    plt.close(fig)
    # -------------------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(25, 100))
    ix_training, ix_test = [], []
    for fold in cv_permute.split(train_selected, y_train):
        ix_training.append(fold[0]), ix_test.append(fold[1])

    new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]

    new_df = []
    for int_ind in new_index:
        new_df.append(train_selected.iloc[int_ind].values)
    new_df = pd.DataFrame(new_df , columns=train_selected.columns)

    new_df2 =  pd.concat(input_data_all)

    shap.summary_plot(np.array(SHAP_values_per_fold), new_df2, show=False, plot_type="bar", class_names= pipe_set['model'].classes_)
    plt.gcf().set_size_inches(25, 25)
    plt.tight_layout()
    plt.savefig(file_to_write + '_' + model_name + '__SHAP.pdf', dpi=700)
    plt.close()
    #------------------

    fig, ax = plt.subplots(figsize=(25, 100))
    shap.summary_plot(np.array(SHAP_values_per_fold), new_df2, show=False, class_names=pipe_set['model'].classes_)
    plt.gcf().set_size_inches(25, 25)
    plt.tight_layout()
    plt.savefig(file_to_write + '_' + model_name + '__SHAP_2.pdf', dpi=700)
    #fig.savefig(file_to_write + '_' + model_name + '__SHAP.png', dpi=300)
    plt.close()

    eval_permut = pd.concat(test_score)

    return mean_fpr , mean_tpr, model_name, eval_permut


####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

# selection_type = 'all'    # all: simple + arfs + boruta
# select_modalities = 'run4'
# pipeline_scaling_ = 'no_norm'
# num_of_feats = 10
# treatment = '0'

def my_main_funct(selection_type , treatment, pipeline_scaling_, num_of_feats, select_modalities):
    num_of_feats = int(num_of_feats)
    print('-----1------')
    print(treatment)
    print(pipeline_scaling_)


    #main_path_pkl = .............
    #path_with_pickles = ........
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    #STEP #2 Import train/test splits + stratified CV

    if select_modalities == 'proteom':
        print('********************* PROTEOM *****************************')
        skf = read_pickle(main_path_pkl + 'skf_folds_proteomics.pkl')
        model_train_vect = read_pickle(main_path_pkl + 'model_train_vect_proteomics.pkl')
        external_test_vect = read_pickle(main_path_pkl + 'external_test_vect_proteomics.pkl')
        cv_permute = read_pickle(main_path_pkl + 'cv_permute_proteomics.pkl')
    else:
        skf = read_pickle(main_path_pkl + 'skf_folds.pkl')
        model_train_vect = read_pickle(main_path_pkl + 'model_train_vect.pkl')
        external_test_vect = read_pickle(main_path_pkl + 'external_test_vect.pkl')
        cv_permute = read_pickle(main_path_pkl + 'cv_permute.pkl')



    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # STEP #3 select case to run
    if pipeline_scaling_ == 'norm':
        pipeline_scaling = True
        pipe_write = 'norm'
        print('-------------------  NORM ---------------------')
    else:
        pipeline_scaling = False
        pipe_write = 'no_norm'

    if treatment == '1' or treatment == '0':
        print('******treat*****')

        analysis_case = selection_type + '_' + pipe_write + '_TREAT_' + str(treatment)
    else:
        analysis_case = selection_type + '_' + pipe_write


    case_to_run = 'results_' + analysis_case + '_selfeats_' + str(num_of_feats) + '_' + select_modalities
    print(case_to_run)

    train_info = read_pickle( path_with_pickles + case_to_run +  '.pkl')
    train_info_for_test = return_overall_train_performance(train_info)

    model_names_df = train_info_for_test[1].index.to_list()

    model_feat_ranking = []
    model_pipelines = []
    tested_names = []
    for i in model_names_df:
        tmp = train_info_for_test[0][train_info_for_test[0]["Model"].str.contains(i)]
        model_feat_ranking.append(         tmp.sort_values(by=['counts_'], ascending=False)                )
        for iii in range(len(train_info)):
            if train_info[iii][0]['Model'][0]==i and i not in tested_names :
                model_pipelines.append(train_info[iii][2][0])
                tested_names.append(i)


    ##### each model in parallel
    new_list1_test=[]
    new_list1_test.extend(model_train_vect[0] for i in range(len(model_names_df)))

    new_list2_test=[]
    new_list2_test.extend(external_test_vect[0] for i in range(len(model_names_df)))

    new_list3_test=[]
    new_list3_test.extend(model_names_df[i] for i in range(len(model_names_df)))

    new_list4_test=[]
    new_list4_test.extend(model_feat_ranking[i] for i in range(len(model_names_df)))

    new_list5_test=[]
    new_list5_test.extend(model_pipelines[i] for i in range(len(model_names_df)))

    new_list6_test=[]
    new_list6_test.extend(num_of_feats for i in range(len(model_names_df)))

    new_list7_test=[]
    new_list7_test.extend(cv_permute for i in range(len(model_names_df)))

    if not os.path.exists(main_path_pkl + 'ROC_curves'):
        os.makedirs(main_path_pkl + 'ROC_curves')
    if not os.path.exists(main_path_pkl + 'ROC_curves/' + case_to_run):
        os.makedirs(main_path_pkl + 'ROC_curves/' + case_to_run)

    file_to_write_ = main_path_pkl + 'ROC_curves/' + case_to_run + '/Testing_' + case_to_run
    new_list8_test = []
    new_list8_test.extend(file_to_write_ for i in range(len(model_names_df)))

    new_list9_test = []
    new_list9_test.extend(pipeline_scaling for i in range(len(model_names_df)))


    aaa2 = Parallel(n_jobs=-1)(delayed(permutation_phase_PARALLEL)(i,j,k,l,m,n,o,p,q) for i,j,k,l,m,n,o,p,q in zip(new_list1_test, new_list2_test , new_list3_test , new_list4_test, new_list5_test, new_list6_test, new_list7_test, new_list8_test, new_list9_test))
    return aaa2, case_to_run



if __name__ == '__main__':
    aaaa2 = my_main_funct(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    #main_path_pkl = ......
    output = open(main_path_pkl + 'Testing_' + aaaa2[1] + '.pkl', 'wb')
    pickle.dump(aaaa2[0], output)
    output.close()
