from pandas import read_pickle
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
from sklearn.preprocessing import MinMaxScaler


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
from fairlearn.metrics import demographic_parity_ratio, count, false_positive_rate, selection_rate, equalized_odds_ratio, MetricFrame

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

def return_overall_test_performance(test_samples) :
    all_classifier_results = pd.DataFrame(
        columns=['Accuracy_Mean', 'Sensitivity_Mean', 'Specificity_Mean', 'ROC_Mean', 'F1_Mean', 'Kappa_Mean',
                  'PPV_Mean',
                  'NPV_Mean', 'MCC_Mean', 'PrecRecall_Mean', 'Balanced_Acc_Mean', 'Accuracy_Std', 'Sensitivity_Std',
                  'Specificity_Std',
                  'ROC_Std', 'F1_Std', 'Kappa_Std', 'PPV_Std',
                  'NPV_Std', 'MCC_Std', 'PrecRecall_Std', 'Balanced_Acc_Std',
                  'EqualOddsRatio_Mean', 'DemoParityRatio_Mean', 'EqualOddsRatio_Std', 'DemoParityRatio_Std'])#, 'Pvalue'])


    train_results = []
    for i in range(len(test_samples)):
        train_results.append(test_samples[i][0])

    ndf = pd.concat(train_results)


    return ndf


##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
def return_fairness(fair_ytest, fair_ypred, fair_metrics_fold, fair_metrics):

    #my_metrics = {'tpr': recall_score,'fpr': false_positive_rate,'sel': selection_rate,'count': count}
    #acc_frame = MetricFrame(
    #    metrics=my_metrics,
    #    y_true=fair_ytest,
    #    y_pred=fair_ypred,
    #    sensitive_features=fair_metrics_fold[fair_metrics])

    equal_odds_ratio = equalized_odds_ratio(fair_ytest,fair_ypred,sensitive_features=fair_metrics_fold[fair_metrics])  #The ideal value for this metric is 1, which indicates that the true
                                                                                                    #and false positive rates for different groups are equal

    demo_par_ratio = demographic_parity_ratio(fair_ytest,fair_ypred,sensitive_features=fair_metrics_fold[fair_metrics]) # The ideal value for this metric is 1, which indicates that the
                                                                                                       # selection rates for different groups are equal.
    return [equal_odds_ratio, demo_par_ratio]
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
def permutation_train_after_selection_PARALLEL(train_from_vect, skf, model_name  ,  selected_features, input_pipe_set, num_of_feats , vect_for_fair_input, data_for_fair_ml, treatment):

    methods_to_check = ('rusboost__', 'adacost__', 'catboost__', 'selfpace__', 'asymboost__', 'compada__')

    perf_columns = ['Accuracy_Mean', 'Sensitivity_Mean', 'Specificity_Mean', 'ROC_Mean', 'F1_Mean', 'Kappa_Mean',
                  'PPV_Mean',
                  'NPV_Mean', 'MCC_Mean', 'PrecRecall_Mean', 'Balanced_Acc_Mean', 'Accuracy_Std', 'Sensitivity_Std',
                  'Specificity_Std',
                  'ROC_Std', 'F1_Std', 'Kappa_Std', 'PPV_Std',
                  'NPV_Std', 'MCC_Std', 'PrecRecall_Std', 'Balanced_Acc_Std',
                  'EqualOddsRatio_Mean', 'DemoParityRatio_Mean', 'EqualOddsRatio_Std', 'DemoParityRatio_Std']
    ###################################################################################################################
    ###########  SOS we merge train and test just for overall cohort performance indication used further for external validation
    ######train_from_vect = pd.concat([train_from_vect, test_from_vect], axis=0)
    ###################################################################################################################
    roc_data_all = []
    tmp_all_results = []

    #####  STEP1: pick BEST FEATURES
    feats_to_run = selected_features

    #####feats_to_run['ranks_'] = feats_to_run['ranks_'].str.get(0)

    #feats_to_run = feats_to_run.sort_values(by=['counts_', 'ranks_'], ascending=[False , False])
    feats_to_run = feats_to_run.sort_values(by=['hybrid_ranks_'], ascending=[False])

    tmp_name = model_name

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------

    feats_to_run_ = feats_to_run['feat_name'].iloc[0:num_of_feats]


    #--------------------------------------------------------------------------------------------
    #check if training has been run............
    if feats_to_run['counts_'].count() > 0:
        try:
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
            pipe_set = input_pipe_set

            if not model_name.startswith(methods_to_check):
                ############################  remove the feature selection
                idx = [idx for idx, name in enumerate(pipe_set.named_steps.keys()) if name == 'arfs' ][0]
                pipe_set.steps.pop(idx)

            ############################  remove the COLLINEARITY
            idx = [idx for idx, name in enumerate(pipe_set.named_steps.keys()) if name == 'collinear' ][0]
            pipe_set.steps.pop(idx)

            ############################  remove the UniqueValuesThreshold
            idx = [idx for idx, name in enumerate(pipe_set.named_steps.keys()) if name == 'unique' ][0]
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
            iii = 0

            data_for_permut = train_from_vect[iii][feats_to_run_]
            y_permut = train_from_vect[iii]['pCR']



            all_tests = cross_validate(pipe_set, data_for_permut, y_permut, cv=skf, scoring=custom_scoring, return_estimator=True, return_indices=True)
            tmp = return_cross_validate_performance(all_tests )

            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################
            ##### assess fairness from the outer loop train/
            data_for_fair_ml = data_for_fair_ml.drop('patient_id', axis=1)
            fair_results = []
            count_indices = 0
            #for test_index in train_test_indices[1]:
            for i, (train_index__, test_index__) in enumerate(skf):
                fair_ytest = y_permut.iloc[test_index__].values
                fair_metrics_fold = data_for_fair_ml.iloc[test_index__]

                if np.isnan(all_tests['test_acc'][count_indices]):
                    fair_results.append(np.full( 1 * 2, np.nan).tolist())  #SOS to 1 einai karfwto epeidh exoume mia parametro. alliws fix it
                else:
                    fair_ypred = all_tests['estimator'][count_indices].predict(data_for_permut.iloc[test_index__])
                    fair_results.append(return_fairness(fair_ytest, fair_ypred, fair_metrics_fold, vect_for_fair_input))

                count_indices = count_indices +1

            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################

            fair_results_total = pd.DataFrame(fair_results, columns=['equal_odds_ratio', 'demo_par_ratio'])
            fair_results_total = fair_results_total.describe().loc[["mean", "std"]]
            fair_results_total = pd.DataFrame([fair_results_total['equal_odds_ratio']['mean'], fair_results_total['demo_par_ratio']['mean'], fair_results_total['equal_odds_ratio']['std'], fair_results_total['demo_par_ratio']['std'] ]).transpose()
            fair_results_total.columns= ['EqualOddsRatio_Mean', 'DemoParityRatio_Mean', 'EqualOddsRatio_Std', 'DemoParityRatio_Std']

            tmp = pd.concat([tmp, fair_results_total.round(4)], axis=1)
            print('DONE:----------' + tmp_name + '-------------')

            tmp.insert(0, 'Model', tmp_name)
            tmp_all_results.append(tmp)

        except:
            tmp = pd.DataFrame(np.nan, index=[0], columns=perf_columns)
            tmp.insert(0, 'Model', tmp_name)
            tmp_all_results.append(tmp)
            print("An exception occurred")
    else:
        ##############   return nan performance
        tmp = pd.DataFrame(np.nan, index=[0], columns=perf_columns)
        tmp.insert(0, 'Model', tmp_name)
        tmp_all_results.append(tmp)

    tmp_all_results = pd.concat(tmp_all_results)
    return  tmp_all_results , feats_to_run_.values
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
def permutation_phase_PARALLEL(train_from_vect , test_from_vect , model_name  ,  selected_features, input_pipe_set , vect_for_fair_input, file_to_write, pipeline_scaling, data_for_fair_ml, train_test_indices, percent_cutoff, treatment ):

    methods_to_check = ('rusboost__', 'adacost__', 'catboost__', 'selfpace__', 'asymboost__', 'compada__')

    perf_columns = ['Accuracy_Mean', 'Sensitivity_Mean', 'Specificity_Mean', 'ROC_Mean', 'F1_Mean', 'Kappa_Mean',
                  'PPV_Mean',
                  'NPV_Mean', 'MCC_Mean', 'PrecRecall_Mean', 'Balanced_Acc_Mean', 'Accuracy_Std', 'Sensitivity_Std',
                  'Specificity_Std',
                  'ROC_Std', 'F1_Std', 'Kappa_Std', 'PPV_Std',
                  'NPV_Std', 'MCC_Std', 'PrecRecall_Std', 'Balanced_Acc_Std',
                  'EqualOddsRatio_Mean', 'DemoParityRatio_Mean', 'EqualOddsRatio_Std', 'DemoParityRatio_Std']#, 'Pvalue']

    roc_data_all = []
    tmp_all_results = []

    #####  STEP1: pick BEST FEATURES
    feats_to_run = selected_features

    #####feats_to_run['ranks_'] = feats_to_run['ranks_'].str.get(0)

    #feats_to_run = feats_to_run.sort_values(by=['counts_', 'ranks_'], ascending=[False , False])
    feats_to_run = feats_to_run.sort_values(by=['hybrid_ranks_'], ascending=[False])

    tmp_name = model_name

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    if percent_cutoff<0:
        num_of_feats = train_from_vect[0].shape[1]
    else:
        num_of_feats = np.percentile(feats_to_run['hybrid_ranks_'], percent_cutoff)
        num_of_feats = sum(feats_to_run['hybrid_ranks_'] > num_of_feats)

        if num_of_feats > np.round( (train_from_vect[0].shape[0] + test_from_vect[0].shape[0])/10 ) :
            num_of_feats = np.round( (train_from_vect[0].shape[0] + test_from_vect[0].shape[0])/10 )
            num_of_feats = num_of_feats.astype(int)


    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------

    feats_to_run_ = feats_to_run['feat_name'].iloc[0:num_of_feats]


    #--------------------------------------------------------------------------------------------
    #check if training has been run............
    if feats_to_run['counts_'].count() > 0:
        try:
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
            pipe_set = input_pipe_set

            if not model_name.startswith(methods_to_check):
                ############################  remove the feature selection
                idx = [idx for idx, name in enumerate(pipe_set.named_steps.keys()) if name == 'arfs' ][0]
                pipe_set.steps.pop(idx)

            ############################  remove the COLLINEARITY
            idx = [idx for idx, name in enumerate(pipe_set.named_steps.keys()) if name == 'collinear' ][0]
            pipe_set.steps.pop(idx)

            ############################  remove the UniqueValuesThreshold
            idx = [idx for idx, name in enumerate(pipe_set.named_steps.keys()) if name == 'unique' ][0]
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
            iii = 0

            y_permut_train = train_from_vect[iii]['pCR']
            y_permut_test = test_from_vect[iii]['pCR']

            data_for_permut = pd.concat([train_from_vect[iii][feats_to_run_], test_from_vect[iii][feats_to_run_]], axis=0)
            y_permut = pd.concat([y_permut_train, y_permut_test], axis=0)

            data_for_permut = data_for_permut.sort_index()
            y_permut = y_permut.sort_index()


            #score, permutation_scores, pvalue = permutation_test_score(pipe_set, data_for_permut, y_permut, scoring= "roc_auc", random_state=0, cv = zip(train_test_indices[0],train_test_indices[1]), verbose=0)
            pvalue = 0
            all_tests = cross_validate(pipe_set, data_for_permut, y_permut, cv=zip(train_test_indices[0],train_test_indices[1]), scoring=custom_scoring, return_estimator=True, return_indices=True)
            tmp = return_cross_validate_performance(all_tests )


            roc_data = plot_roc_curves(train_test_indices, data_for_permut, y_permut, pipe_set, file_to_write, model_name,pipeline_scaling, methods_to_check)

            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################
            ##### assess fairness from the outer loop train/
            data_for_fair_ml = data_for_fair_ml.drop('patient_id', axis=1)
            data_for_fair_ml = data_for_fair_ml.sort_index()
            fair_results = []
            count_indices = 0
            for test_index in train_test_indices[1]:
                #print(test_index)
                fair_ytest = y_permut.iloc[test_index].values
                fair_metrics_fold = data_for_fair_ml.iloc[test_index]

                if np.isnan(all_tests['test_acc'][count_indices]):
                    fair_results.append(np.full( 1 * 2, np.nan).tolist())  #SOS to 1 einai karfwto epeidh exoume mia parametro. alliws fix it
                else:
                    fair_ypred = all_tests['estimator'][count_indices].predict(data_for_permut.iloc[test_index])
                    fair_results.append(return_fairness(fair_ytest, fair_ypred, fair_metrics_fold, vect_for_fair_input))

                count_indices = count_indices +1

            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################

            fair_results_total = pd.DataFrame(fair_results, columns=['equal_odds_ratio', 'demo_par_ratio'])
            fair_results_total = fair_results_total.describe().loc[["mean", "std"]]
            fair_results_total = pd.DataFrame([fair_results_total['equal_odds_ratio']['mean'], fair_results_total['demo_par_ratio']['mean'], fair_results_total['equal_odds_ratio']['std'], fair_results_total['demo_par_ratio']['std'] ]).transpose()
            fair_results_total.columns= ['EqualOddsRatio_Mean', 'DemoParityRatio_Mean', 'EqualOddsRatio_Std', 'DemoParityRatio_Std']

            tmp = pd.concat([tmp, fair_results_total.round(4)], axis=1)
            #tmp['Pvalue'] = pvalue
            print('DONE:----------' + tmp_name + '-------------')

            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################
            #######################################################################################################################

            tmp.insert(0, 'Model', tmp_name)
            tmp_all_results.append(tmp)
            roc_data_all.append(roc_data)

        except:
            tmp = pd.DataFrame(np.nan, index=[0], columns=perf_columns)
            tmp.insert(0, 'Model', tmp_name)
            tmp_all_results.append(tmp)
            roc_data_all.append(tuple())
            print("An exception occurred")
    else:
        ##############   return nan performance
        tmp = pd.DataFrame(np.nan, index=[0], columns=perf_columns)
        tmp.insert(0, 'Model', tmp_name)
        tmp_all_results.append(tmp)

    tmp_all_results = pd.concat(tmp_all_results)
    return  tmp_all_results , feats_to_run_.values, roc_data_all, num_of_feats

####################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################
##########################################################################################################################################################################

def plot_roc_curves( train_test_indices, data_for_permut, y_permut, pipe_set, file_to_write, model_name, pipeline_scaling, methods_to_check  ):
    tprs = []
    aucs = []
    SHAP_values_per_fold = []
    mean_fpr = np.linspace(0, 1, 100)
    input_data_all = []
    count_fig = 0
    test_score = []
    #for i, (train, test) in enumerate(cv_permute.split(train_selected, y_train)):
    for train, test in zip(train_test_indices[0], train_test_indices[1]):
        #print(test)

        if model_name.startswith('simplepdnu__'):
            pipe_set.fit(data_for_permut.iloc[train], y_permut.iloc[train].values)
        else:
            pipe_set.fit(data_for_permut.iloc[train], y_permut.iloc[train])

        test_score.append(pd.DataFrame(evaluate_model(pipe_set, data_for_permut.iloc[test], y_permut.iloc[test])).transpose())

        input_data = data_for_permut.iloc[test]
        input_data_all.append(input_data)

        ex = shap.KernelExplainer(pipe_set.predict, data_for_permut.iloc[train], keep_index=True)
        shap_values = ex.shap_values(input_data)

        for SHAPs in shap_values:
            SHAP_values_per_fold.append(SHAPs)

        #---------------------------------------------------------------------------------------------------------------
        viz = RocCurveDisplay.from_estimator(pipe_set,data_for_permut.iloc[test], y_permut.iloc[test])

        #probability of the class with the greater label,
        #fpr, tpr, thresholds = metrics.roc_curve(y_permut.iloc[test].values, pipe_set.predict_proba(input_data)[:,1])
        #metrics.auc(fpr, tpr)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        #---------------------------------------------------------------------------------------------------------------

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)#metrics.auc(mean_fpr, mean_tpr)
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
    for fold in range(len(train_test_indices[0])):#cv_permute.split(data_for_permut, y_permut):
        #print(fold)
        ix_training.append(train_test_indices[0][fold]), ix_test.append(train_test_indices[1][fold])

    new_df2 =  pd.concat(input_data_all)

    shap.summary_plot(np.array(SHAP_values_per_fold), new_df2, show=False, plot_type="bar", class_names= pipe_set.classes_)

    plt.gcf().set_size_inches(25, 25)
    plt.tight_layout()
    plt.savefig(file_to_write + '_' + model_name + '__SHAP.pdf', dpi=700)
    plt.close()
    #------------------

    fig, ax = plt.subplots(figsize=(25, 100))

    shap.summary_plot(np.array(SHAP_values_per_fold), new_df2, show=False, class_names=pipe_set.classes_)

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
def my_main_funct(selection_type , treatment, pipeline_scaling_, num_of_feats, select_modalities, main_path_pkl, percent_cutoff):
    num_of_feats = int(num_of_feats)
    print(treatment)
    print(pipeline_scaling_)

    hybrid_rank_perc_counts = 0.5

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
    test_indices = read_pickle(main_path_pkl + 'test_indices__' + drug_name +'.pkl')
    train_indices = read_pickle(main_path_pkl + 'train_indices__' + drug_name +'.pkl')

    all_indices = [train_indices, test_indices]

    vect_for_fair = [vect_for_fair_input]
    vect_for_fair.append('patient_id')
    tmp = pd.concat([model_train_vect[0], external_test_vect[0]])

    data_for_fair_ml = pd.DataFrame(tmp[vect_for_fair])
    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    # SELECT FEATURES TO RUN THE MODELS
    if select_modalities == 'clinical':  # 4 variables + ARM
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith('Clin_')] + list(['pCR', 'patient_id'])
    elif select_modalities == 'DNA':
        selected_columns = [column for column in model_train_vect[0].columns if  # 52 variables
                            column.startswith('DNA_')] + list(['pCR', 'patient_id'])
    elif select_modalities == 'RNA':
        selected_columns = [column for column in model_train_vect[0].columns if  # 52 variables
                            column.startswith('RNA_')] + list(['pCR', 'patient_id'])
    elif select_modalities == 'Image':
        selected_columns = [column for column in model_train_vect[0].columns if  # 3 variables
                            column.startswith('WSI_')] + list(['pCR', 'patient_id'])
    elif select_modalities == 'Proteomics':
        selected_columns = [column for column in model_train_vect[0].columns if  # 22 variables
                            column.startswith('Prot_')] + list(['pCR', 'patient_id'])
    elif select_modalities == 'run1':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith(('Clin_', 'DNA_'))] + list(['pCR', 'patient_id'])
    elif select_modalities == 'run2':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith(('Clin_', 'DNA_', 'RNA_'))] + list(['pCR', 'patient_id'])
    elif select_modalities == 'run3':
        selected_columns = [column for column in model_train_vect[0].columns if
                            column.startswith(('Clin_', 'DNA_', 'RNA_', 'Prot_'))] + list(['pCR', 'patient_id'])
    elif select_modalities == 'run4':
        selected_columns = model_train_vect[0].columns
    elif select_modalities == 'simple':
        selected_columns = ['DNA_ERBB2_CNA', 'RNA_mRNA-ERBB2', 'Prot_ERBB2', 'pCR', 'patient_id']
    else:
        print('nothing to run')


    for i in range(len(model_train_vect)):
        model_train_vect[i] = model_train_vect[i][selected_columns]

    for i in range(len(external_test_vect)):
        external_test_vect[i] = external_test_vect[i][selected_columns]

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    #this is to normallize ranking from 0 to all train runs
    scaler_minmax = MinMaxScaler(feature_range=(0, len(external_test_vect)*len(skf)))
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

    #do the check when selected features are more than raw columns

    case_to_run = analysis_case + '_selfeats_' + str(num_of_feats)+ '_' + select_modalities
    print(case_to_run)

    train_info = pd.read_pickle( main_path_pkl + 'model_results/' + case_to_run +  '.pkl')
    train_info_for_test = return_overall_train_performance(train_info)

    list_of_classifiers_to_store = pd.read_pickle( main_path_pkl + 'model_results/list_of_classifiers_to_store.pkl')

    model_names_df = train_info_for_test[1].index.to_list()

    model_feat_ranking = []
    model_pipelines = []
    tested_names = []
    for i in model_names_df:
        tmp = train_info_for_test[0][train_info_for_test[0]["Model"].str.contains(i)]

        if not tmp.empty:
            #normalize ranks
            tmp['ranks_'] = scaler_minmax.fit_transform(tmp['ranks_'].values[:, None])
            tmp['hybrid_ranks_'] =  hybrid_rank_perc_counts*tmp['counts_'].values +  (1-hybrid_rank_perc_counts)*tmp['ranks_'].values

            model_feat_ranking.append(   tmp.sort_values(by=['hybrid_ranks_'], ascending=[False])     )      #tmp.sort_values(by=['ranks_','counts_'], ascending=[False,False])                )
        else:
            tmp['hybrid_ranks_'] = np.nan
            model_feat_ranking.append(tmp)

        for iii in range(len(train_info)):
            if train_info[iii][0]['Model'][0]==i and i not in tested_names :
                model_pipelines.append(train_info[iii][2][0])
                tested_names.append(i)


    ################# add model_names_df, model_feat_ranking, model_pipelines for FUSION!
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    split_strings_name = []
    for i in range(len(list_of_classifiers_to_store)):
        split_strings_name.append('fusion__NONE__' + list_of_classifiers_to_store[i][0])
    split_strings_pipe = []
    for i in range(len(list_of_classifiers_to_store)):
        split_strings_pipe.append(list_of_classifiers_to_store[i][1])

    #1
    model_names_df = model_names_df + split_strings_name

    #2
    model_feat_ranking_fused = pd.concat(model_feat_ranking)
    grouped = model_feat_ranking_fused.groupby(['feat_name'])['hybrid_ranks_'].mean()
    grouped = grouped.sort_values(ascending=[False])

    fused_ranking = []
    for i in range(len(split_strings_name)):
        fused_ranking.append( pd.DataFrame( {'Model': split_strings_name[i], 'feat_name': grouped.index,
        'counts_': [1]*len(grouped.index), 'ranks_': [1]*len(grouped.index), 'hybrid_ranks_': grouped.values } ) )
    model_feat_ranking = model_feat_ranking + fused_ranking

    #3
    coll_thres = 0.8
    miss_thres = 0.2
    fused_pipelines = []
    for i in range(len(split_strings_name)):
        if pipeline_scaling:
            fused_pipelines.append(Pipeline(
            [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
            ('collinear', CollinearityThreshold(threshold=coll_thres)),
            ("imputer", PandasSimpleImputer()),
            ("unique", UniqueValuesThreshold(threshold=1)),
            ('trasform', StandardScaler().set_output(transform='pandas')),
            ("arfs", DummyClassifier()),
            ('model', split_strings_pipe[i])]))
        else:
            fused_pipelines.append(Pipeline(
            [("missing", MissingValueThreshold(threshold=miss_thres)),  # delete columns having >10% missing
            ('collinear', CollinearityThreshold(threshold=coll_thres)),
            ("imputer", PandasSimpleImputer()),
            ("unique", UniqueValuesThreshold(threshold=1)),
            ('trasform', StandardScaler().set_output(transform='pandas')),
            ("arfs", DummyClassifier()),
            ('model', split_strings_pipe[i])]))

    model_pipelines = model_pipelines + fused_pipelines

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------


    ##### each model in parallel
    new_list1_test=[]
    new_list1_test.extend(model_train_vect for i in range(len(model_names_df)))

    new_list2_test=[]
    new_list2_test.extend(external_test_vect for i in range(len(model_names_df)))

    new_list3_test=[]
    new_list3_test.extend(model_names_df[i] for i in range(len(model_names_df)))

    new_list4_test=[]
    new_list4_test.extend(model_feat_ranking[i] for i in range(len(model_names_df)))

    new_list5_test=[]
    new_list5_test.extend(model_pipelines[i] for i in range(len(model_names_df)))

    new_list7_test=[]
    new_list7_test.extend(vect_for_fair_input for i in range(len(model_names_df)))

    if not os.path.exists(main_path_pkl + 'ROC_curves'):
        os.makedirs(main_path_pkl + 'ROC_curves')
    if not os.path.exists(main_path_pkl + 'ROC_curves/' + case_to_run):
        os.makedirs(main_path_pkl + 'ROC_curves/' + case_to_run)

    file_to_write_ = main_path_pkl + 'ROC_curves/' + case_to_run + '/Testing_'
    new_list8_test = []
    new_list8_test.extend(file_to_write_ for i in range(len(model_names_df)))

    new_list9_test = []
    new_list9_test.extend(pipeline_scaling for i in range(len(model_names_df)))

    new_list10_test = []
    new_list10_test.extend(data_for_fair_ml for i in range(len(model_names_df)))

    new_list11_test = []
    new_list11_test.extend(all_indices for i in range(len(model_names_df)))

    new_list12_test = []
    new_list12_test.extend(skf for i in range(len(model_names_df)))

    new_list13_test = []
    new_list13_test.extend(percent_cutoff for i in range(len(model_names_df)))

    new_list14_test = []
    new_list14_test.extend(treatment for i in range(len(model_names_df)))

    aaa2 = Parallel(n_jobs=-1)(delayed(permutation_phase_PARALLEL)(i,j,k,l,m,n,o,p,q,r,s, t) for i,j,k,l,m,n,o,p,q,r,s,t in zip(new_list1_test, new_list2_test , new_list3_test , new_list4_test, new_list5_test, new_list7_test, new_list8_test, new_list9_test, new_list10_test,new_list11_test,new_list13_test, new_list14_test))

    num_of_feats = aaa2[0][3]
    new_list6_test = []
    new_list6_test.extend(num_of_feats for i in range(len(model_names_df)))

    bbb2 = Parallel(n_jobs=-1)(
        delayed(permutation_train_after_selection_PARALLEL)(i, j, k, l, m, n, o, p, q) for i, j, k, l, m, n, o, p, q in
        zip(new_list1_test, new_list12_test, new_list3_test, new_list4_test, new_list5_test, new_list6_test, new_list7_test,new_list10_test, new_list14_test))

    #xxx2 = return_overall_test_performance(bbb2)

    output = open(main_path_pkl + 'Testing_' + case_to_run + '.pkl', 'wb')
    pickle.dump(aaa2, output)
    output.close()

    output = open(main_path_pkl + 'Training_at_Testing_' + case_to_run + '.pkl', 'wb')
    pickle.dump(bbb2, output)
    output.close()

    output = open(main_path_pkl + 'Model_Feats_Ranking_' + case_to_run + '.pkl', 'wb')
    pickle.dump(model_feat_ranking, output)
    output.close()

    output = open(main_path_pkl + 'Model_Names_' + case_to_run + '.pkl', 'wb')
    pickle.dump(model_names_df, output)
    output.close()

    output = open(main_path_pkl + 'Model_Pipelines_' + case_to_run + '.pkl', 'wb')
    pickle.dump(model_pipelines, output)
    output.close()


    return aaa2, case_to_run, bbb2



#if __name__ == '__main__':
#    my_main_funct(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    # main_path_pkl =    '/home/gman/PycharmProjects/SklearnML/'#'/mimer/NOBACKUP/groups/foukakis_ai/manikis/Python/hande_ml/latest_kang_paper/'



