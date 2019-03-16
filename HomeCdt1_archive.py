# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:16:49 2018

@author: MXU29
"""
import numpy as np
import pandas as pd
import gc
import os
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import lightgbm as gbm

lb = LabelEncoder()
    
def LabelEncoding_Cat(df):
    df = df.copy()
    Cat_Var= df.select_dtypes('object').columns.tolist()
    for col in Cat_Var:
        df[col] = lb.fit_transform(df[col].astype('str'))
    return df

def Dummy(df):
    df = df.copy()
    Cat_Var= df.select_dtypes('object').columns.tolist()
    for col in Cat_Var:
        df = pd.concat([df, pd.get_dummies(df[col].astype('str'), prefix = col)], axis=1).drop(col, axis=1)
    return df
    
def Fill_NA(df):
    df = df.copy()
    Num_Features = df.select_dtypes(['float64', 'int64']).columns.tolist()
    df[Num_Features] = df[Num_Features].fillna(-999)
    return df

bureau= (pd.read_csv('../input/bureau.csv').pipe(LabelEncoding_Cat))
cred_card_bal = (pd.read_csv('../input/credit_card_balance.csv').pipe(LabelEncoding_Cat))
pos_cash_bal = (pd.read_csv('../input/POS_CASH_balance.csv').pipe(LabelEncoding_Cat))
prev = (pd.read_csv('../input/previous_application.csv').pipe(LabelEncoding_Cat))
inst = (pd.read_csv('../input/installments_payments.csv').pipe(LabelEncoding_Cat))

statGroup = ['mean', 'count', 'median', 'max', 'std', 'min']
#statGroup = ['mean', 'count', 'median', 'max', 'std', 'min', 'sum']

print('Preprocessing the bunch of csv files ...\n')
Label1 = [s+'_'+l for s in bureau.columns.tolist() if s!= 'SK_ID_CURR' for l in statGroup]
avg_bureau = bureau.groupby('SK_ID_CURR').agg(statGroup).reset_index()
avg_bureau.columns = ['SK_ID_CURR']+Label1

Label2 = [s+'_'+l for s in cred_card_bal.columns.tolist() if s!= 'SK_ID_CURR' for l in statGroup]
avg_cred_card_bal = cred_card_bal.groupby('SK_ID_CURR').agg(statGroup).reset_index()
avg_cred_card_bal.columns = ['SK_ID_CURR']+Label2

Label3 = [s+'_'+l for s in pos_cash_bal.columns.tolist() if s not in ['SK_ID_PREV', 'SK_ID_CURR'] for l in statGroup]
avg_pos_cash_bal = pos_cash_bal.groupby(['SK_ID_PREV', 'SK_ID_CURR']).agg(statGroup).groupby(level='SK_ID_CURR').agg('mean').reset_index()          
avg_pos_cash_bal.columns = ['SK_ID_CURR']+Label3

Label4 = [s+'_'+l for s in prev.columns.tolist() if s!= 'SK_ID_CURR' for l in statGroup]
avg_prev = prev.groupby('SK_ID_CURR').agg(statGroup).reset_index()
avg_prev.columns = ['SK_ID_CURR']+Label4

Label5 = [s+'_'+l for s in inst.columns.tolist() if s!= 'SK_ID_CURR' for l in statGroup]
avg_inst = inst.groupby('SK_ID_CURR').agg(statGroup).reset_index()
avg_inst.columns = ['SK_ID_CURR']+Label5

del(Label1,Label2,Label3,Label4, Label5)
train = pd.read_csv('../input/application_train.csv')          
test = pd.read_csv('../input/application_test.csv')

trainLen = train.shape[0]          
y = train.TARGET.copy()          

allData = (train.drop(labels = ['TARGET'], axis = 1).append(test)
                .pipe(LabelEncoding_Cat).pipe(Fill_NA)
                .merge(avg_bureau,on = 'SK_ID_CURR', how = 'left')
                .merge(avg_cred_card_bal, on = 'SK_ID_CURR', how = 'left')
                .merge(avg_pos_cash_bal, on = 'SK_ID_CURR', how = 'left')
                .merge(avg_prev, on ='SK_ID_CURR', how = 'left')
                .merge(avg_inst,on = 'SK_ID_CURR', how = 'left'))


del(train, test, bureau, cred_card_bal, pos_cash_bal, avg_prev, avg_bureau, avg_cred_card_bal, avg_pos_cash_bal, avg_inst)
gc.collect()

print('Preparing input data for lightbgm ... \n')
#
#featuresPoly = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 
#                'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'NAME_INCOME_TYPE',
#                'DAYS_LAST_PHONE_CHANGE', 'CODE_GENDER', 'NAME_EDUCATION_TYPE']
featuresPoly = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']
poly_features = allData[featuresPoly]

poly_transformer = PolynomialFeatures(degree = 2)
poly_transformer.fit(poly_features)
poly_features = poly_transformer.transform(poly_features)
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(featuresPoly))
poly_features['SK_ID_CURR'] = allData['SK_ID_CURR']
allData = allData.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

del(poly_features)
gc.collect()

allData.drop(labels = ['SK_ID_CURR'], axis = 1, inplace = True)
train = allData.iloc[:trainLen,:].copy()
test = allData.iloc[trainLen:, :].copy()


del(allData)

ModelParam = {'objective':'binary',
                'boosting_type': 'gbdt',
                'metric': 'auc',
                'nthread': 8,
                'shrinkage_rate': 0.025,
                'min_child_weight': 18,
                'bagging_fraction': 0.75,
                'feature_fraction': 0.75,
                'lambda_l1': 1.5,
                'lambda_l2': 1,
                'num_leaves': 36}

print('Training model ... \n')

folds = KFold(n_splits = 5, shuffle = True, random_state = 123456)

val_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])          

for n_fold, (train_idx, val_idx) in enumerate(folds.split(train)):
    dtrain = gbm.Dataset(train.iloc[train_idx], y.iloc[train_idx])
    dval = gbm.Dataset(train.iloc[val_idx], y.iloc[val_idx])
    cur_gbm = gbm.train(params=ModelParam, train_set=dtrain, num_boost_round=2000, verbose_eval=1000,valid_sets=dval,
                        valid_names=['train','valid'])
    val_preds[val_idx] = cur_gbm.predict(train.iloc[val_idx]) 
    sub_preds+=cur_gbm.predict(test) / folds.n_splits
    print('Fold %2d AUC : %.6f' %(n_fold + 1, roc_auc_score(y.iloc[val_idx], val_preds[val_idx])))
    del dtrain, dval
    gc.collect()
    
print('Full AUC score %.6f' % roc_auc_score(y, val_preds))
print('Outputing results ... \n')

Submission = pd.read_csv('../input/sample_submission.csv')
Submission['TARGET'] = sub_preds.copy()
Submission.to_csv('lightgbm_sub.csv', index = False)

print('DONE!')

trainFull = train.copy()
trainFull['TARGET'] = y
summary=trainFull.groupby(['TARGET']).agg(statGroup)
