#from feature_engine.encoding import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pickle
from sklearn.model_selection import StratifiedKFold
import os

#treatment = 'BOTH'# 'T-DM1' 'DHP' 'BOTH'

def predix_multiomics_prepare_data(treatment):
    sim_imp = SimpleImputer(strategy='most_frequent')
    encoder = OneHotEncoder(handle_unknown='error', sparse_output=False)

    main_path = os.getcwd() + '/'
    ########################################################################################################################
    data = pd.read_csv(main_path + 'clin_multiomics_curated_metrics_PREDIX_HER2.txt',sep='\t')
    data.dropna(subset=['pCR'], inplace=True)
    ########################################################################################################################
    ########################################################################################################################
    #select either running per treatment or using the overall cohort
    if treatment == 'T-DM1' or treatment == 'DHP':
        print('TREATMENT SELECTED')
        data = data[(data['Clin_Arm'] == treatment)]
        data = data.reset_index(drop=True)

    ########################################################################################################################
    ########################################################################################################################
    #get non-numerical data
    categ_names = list(set(data.columns) - set(data._get_numeric_data().columns))
    tmp = data[categ_names]

    numeric_data = data[list(set(data.columns.tolist()) - set(categ_names))]

    #-----------------------------------------------------------------------------------------------------------------------
    tmp.replace({False: -1, True: 1}, inplace=True)
    #-----------------------------------------------------------------------------------------------------------------------
    mapping = {'positive': 1.0, 'negative': 0.0, 'Positive': 1.0, 'Negative': 0.0, 'T-DM1': 1, 'DHP': 0, 'N+': 1, 'N0': 0}
    tmp = tmp.replace({'Prot_ERBB2_PG': mapping, 'Clin_ER': mapping, 'Clin_Arm': mapping})
    #-----------------------------------------------------------------------------------------------------------------------

    if len(tmp['Clin_ANYNODES'].value_counts()) == 2:
        tmp = tmp.replace({'Clin_ANYNODES': mapping})
    #-----------------------------------------------------------------------------------------------------------------------

    Clin_TUMSIZE = sim_imp.fit_transform(tmp['Clin_TUMSIZE'].to_frame())


    Clin_TUMSIZE = pd.DataFrame({'Clin_TUMSIZE': Clin_TUMSIZE.ravel()})
    Clin_TUMSIZE = encoder.fit_transform(Clin_TUMSIZE)

    categorical_columns = [f'{col}_{cat}' for i, col in enumerate(tmp['Clin_TUMSIZE'].to_frame().columns) for cat in encoder.categories_[i]]
    Clin_TUMSIZE = pd.DataFrame(Clin_TUMSIZE, columns=categorical_columns)

    #delete column Clin_TUMSIZE from tmp
    tmp = tmp.drop('Clin_TUMSIZE', axis=1)
    #-----------------------------------------------------------------------------------------------------------------------

    RNA_sspbc_subtype = sim_imp.fit_transform(tmp['RNA_sspbc.subtype'].to_frame())

    fit_RNA_sspbc_subtype = sim_imp.fit(tmp['RNA_sspbc.subtype'].to_frame())
    output = open(main_path + 'fit_for_RNA_sspbc_subtype__' + treatment +'.pkl', 'wb')
    pickle.dump(fit_RNA_sspbc_subtype, output)
    output.close()

    RNA_sspbc_subtype = pd.DataFrame({'RNA_sspbc_subtype': RNA_sspbc_subtype.ravel()})


    encode_RNA_sspbc_subtype = encoder.fit(RNA_sspbc_subtype)
    output = open(main_path + 'encode_for_RNA_sspbc_subtype__' + treatment +'.pkl', 'wb')
    pickle.dump(encode_RNA_sspbc_subtype, output)
    output.close()

    RNA_sspbc_subtype = encoder.fit_transform(RNA_sspbc_subtype)

    categorical_columns = [f'{col}_{cat}' for i, col in enumerate(tmp['RNA_sspbc.subtype'].to_frame().columns) for cat in encoder.categories_[i]]
    RNA_sspbc_subtype = pd.DataFrame(RNA_sspbc_subtype, columns=categorical_columns)

    #delete column Clin_TUMSIZE from tmp
    tmp = tmp.drop('RNA_sspbc.subtype', axis=1)

    tmp['Clin_ER'] = tmp['Clin_ER'].replace({'positive':1, 'negative':-1})
    #https://ryanandmattdatascience.com/ordinal-encoder/
    ########################################################################################################################

    #### merge dfs
    frames = [tmp, Clin_TUMSIZE, RNA_sspbc_subtype]
    tmp = pd.concat(frames, axis=1)


    #--------------------------------------------------------------
    #get categorical variables and encode them
    #data[categ_names] = data[categ_names].apply(LabelEncoder().fit_transform)
    data = pd.concat([tmp,numeric_data], axis=1)
    data = data.dropna(axis=1, how='all')

    #### all True False to 1/0
    data.replace({False: 0, True: 1}, inplace=True)

    #####data = data.loc[:, data.isnull().sum() < 0.2*data.shape[0]]
    ########################################################################################################################
    ########################################################################################################################


    #--------------------------------------------------------------
    #remove patients having missing data
    data = data[data.isnull().sum(axis=1) < round(0.025*len(data.columns))]

    data.columns = data.columns.str.replace('<=', "less_equal_", regex=True).values
    data.columns = data.columns.str.replace('>', "higher_", regex=True).values
    data.columns = data.columns.str.replace('<', "lower_", regex=True).values
    data.columns = data.columns.str.replace('<=', "less_equal_", regex=True).values

    data = data.reset_index(drop=True)

    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    #
    # #  split outer test sets and training set (further splitted into train-test)
    grps = data.groupby(['pCR','Clin_Arm','Clin_ER'], group_keys=False)
    grps.count()
    #
    test_proportion = 0.2
    outer_train_test = 100
    at_least = 1

    # multiple stratification

    external_test_vect = []
    model_train_vect = []
    train_indices = []
    test_indices = []

    for i in range(outer_train_test):
        external_test = grps.apply(lambda x: x.sample(max(round(len(x) * test_proportion), at_least)))
        train = data[~data.index.isin(external_test.index)]
        external_test_vect.append(external_test)
        model_train_vect.append(train)

        train_indices.append(train.index.values)
        test_indices.append(external_test.index.values)
    ##########################################################################################################################################################################
    ########################################

    output = open(main_path + 'model_train_vect__' + treatment +'.pkl', 'wb')
    pickle.dump(model_train_vect, output)
    output.close()

    output = open(main_path + 'external_test_vect__' + treatment +'.pkl', 'wb')
    pickle.dump(external_test_vect, output)
    output.close()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = []
    for train_idx, test_idx in skf.split(model_train_vect[0], model_train_vect[0]['pCR']):
        fold_indices.append((train_idx, test_idx))

    output = open(main_path + 'skf_folds.pkl', 'wb')
    pickle.dump(fold_indices, output)
    output.close()

    output = open(main_path + 'train_indices__' + treatment +'.pkl', 'wb')
    pickle.dump(train_indices, output)
    output.close()

    output = open(main_path + 'test_indices__' + treatment +'.pkl', 'wb')
    pickle.dump(test_indices, output)
    output.close()

    return data

xxx = predix_multiomics_prepare_data('BOTH')
