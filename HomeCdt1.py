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
from sklearn.model_selection import KFold, StratifiedKFold
from lightgbm import LGBMClassifier
import time
import seaborn as sns
import matplotlib.pyplot as plt
#import lightgbm as lgbm

path = 'G:/My Drive/Shared Drive/data/Kaggle/Home_Credit/input/'
os.chdir(path)

lb = LabelEncoder()

def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
def LabelEncoding_Cat(df):
    df = df.copy()
    Cat_Var= df.select_dtypes('object').columns.tolist()
    Num_Var = [s for s in df.columns.tolist() if s not in Cat_Var]
    Num_Var.remove('SK_ID_CURR')
    for col in Cat_Var:
        df[col] = lb.fit_transform(df[col].astype('str'))
    return df, Cat_Var, Num_Var

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
    

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

    
# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            #is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=32,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            #scale_pos_weight=11
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    
    display_importances(feature_importance_df)
    return test_df, sub_preds, feature_importance_df
    


t0 = time.time()
train = pd.read_csv('../input/application_train.csv')          
test = pd.read_csv('../input/application_test.csv')


        
trainLen = train.shape[0]          
y = train.TARGET.copy()          

allData_org, allDataCat, allDataNum = (train.append(test)
                .pipe(LabelEncoding_Cat))

allData = (allData_org.drop(labels = ['TARGET'], axis = 1).pipe(Fill_NA))
allData['TARGET'] = allData_org['TARGET']

del(train, test, allData_org)
gc.collect()

docs = [_f for _f in allData.columns if 'FLAG_DOC' in _f]
live = [_f for _f in allData.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

allData['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
 
        
inc_by_org = allData[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

allData['NEW_INC_BY_ORG'] = allData['ORGANIZATION_TYPE'].map(inc_by_org)
allData['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = allData['DAYS_LAST_PHONE_CHANGE'] / allData['DAYS_EMPLOYED']
allData['NEW_CREDIT_TO_INCOME_RATIO'] = allData['AMT_CREDIT'] / allData['AMT_INCOME_TOTAL']

allData['NEW_CREDIT_TO_GOODS_RATIO'] = allData['AMT_CREDIT'] / allData['AMT_GOODS_PRICE']
allData['NEW_DOC_IND_KURT'] = allData[docs].kurtosis(axis=1)
allData['NEW_LIVE_IND_SUM'] = allData[live].sum(axis=1)
allData['NEW_INC_PER_CHLD'] = allData['AMT_INCOME_TOTAL'] / (1 + allData['CNT_CHILDREN'])

allData['NEW_SOURCES_PROD'] = allData['EXT_SOURCE_1'] * allData['EXT_SOURCE_2'] * allData['EXT_SOURCE_3']
allData['NEW_EXT_SOURCES_MEAN'] = allData[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
allData['NEW_SCORES_STD'] = allData[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)


allData['NEW_SCORES_STD'] = allData['NEW_SCORES_STD'].fillna(allData['NEW_SCORES_STD'].mean())
allData['NEW_CAR_TO_BIRTH_RATIO'] = allData['OWN_CAR_AGE'] / allData['DAYS_BIRTH']
allData['NEW_CAR_TO_EMPLOY_RATIO'] = allData['OWN_CAR_AGE'] / allData['DAYS_EMPLOYED']
allData['NEW_PHONE_TO_BIRTH_RATIO'] = allData['DAYS_LAST_PHONE_CHANGE'] / allData['DAYS_BIRTH']

# Some simple new features (percentages)
allData['DAYS_EMPLOYED_PERC'] = allData['DAYS_EMPLOYED'] / allData['DAYS_BIRTH']
allData['INCOME_CREDIT_PERC'] = allData['AMT_INCOME_TOTAL'] / allData['AMT_CREDIT']
allData['INCOME_PER_PERSON'] = allData['AMT_INCOME_TOTAL'] / allData['CNT_FAM_MEMBERS']
allData['ANNUITY_INCOME_PERC'] = allData['AMT_ANNUITY'] /  (1+allData['AMT_INCOME_TOTAL'])
allData['DAYS_EMPLOYED_PERC'] = allData['DAYS_EMPLOYED_PERC'].fillna(0.157798)
#allData['CHILDREN_RATIO'] = allData['CNT_CHILDREN'] / allData['CNT_FAM_MEMBERS']
allData['INCOME_PER_FAMILY_MEMBER'] = allData.AMT_INCOME_TOTAL / allData.CNT_FAM_MEMBERS
allData['RATIO_INCOME_GOODS'] = allData.AMT_INCOME_TOTAL -  allData.AMT_GOODS_PRICE
allData['PAYMENT_RATE'] =  allData['AMT_CREDIT'] / allData['AMT_ANNUITY']

statGroupNum = ['mean', 'count', 'median', 'max', 'std', 'min']
statGroupCate = ['count', lambda x: x.mode().iloc[0], 'mean', 'nunique']
#statGroup = ['mean', 'count', 'median', 'max', 'std', 'min', 'sum']

print('Preprocessing bureau：')
print("current time: {:.0f}s".format(time.time() - t0))
bureau, bureauCat, bureauNum= (pd.read_csv('../input/bureau.csv').pipe(LabelEncoding_Cat))

bureauNum.remove('SK_ID_BUREAU')
Label1 = [s+'_'+l for s in bureauNum for l in statGroupNum]
avg_bureau_num = bureau.groupby('SK_ID_CURR')[bureauNum].agg(statGroupNum).reset_index()
avg_bureau_num.columns = ['SK_ID_CURR']+Label1
Label12 = [s+'_'+l for s in bureauCat for l in ['count','mostFreq', 'mean', 'nunique']]
avg_bureau_cat = bureau.groupby('SK_ID_CURR')[bureauCat].agg(statGroupCate).reset_index()
avg_bureau_cat.columns = ['SK_ID_CURR']+Label12

allData = (allData.merge(avg_bureau_num,on = 'SK_ID_CURR', how = 'left')
.merge(avg_bureau_cat,on = 'SK_ID_CURR', how = 'left'))

del(bureau, bureauCat, bureauNum, avg_bureau_num, avg_bureau_cat)
gc.collect()

print('Preprocessing credit_card_balance:')
print("current time: {:.0f}s".format(time.time() - t0))
cred_card_bal, cred_card_balCat, cred_card_balNum = (pd.read_csv('../input/credit_card_balance.csv').pipe(LabelEncoding_Cat))

cred_card_balNum.remove('SK_ID_PREV')
Label2 = [s+'_'+l for s in cred_card_balNum for l in statGroupNum]
avg_cred_card_balNum = cred_card_bal.groupby('SK_ID_CURR')[cred_card_balNum].agg(statGroupNum).reset_index()
avg_cred_card_balNum.columns = ['SK_ID_CURR']+Label2
Label22 = [s+'_'+l for s in cred_card_balCat for l in ['count','mostFreq', 'mean', 'nunique']]
avg_cred_card_balCat = cred_card_bal.groupby('SK_ID_CURR')[cred_card_balCat].agg(statGroupCate).reset_index()
avg_cred_card_balCat.columns = ['SK_ID_CURR']+Label22

allData = (allData.merge(avg_cred_card_balNum, on = 'SK_ID_CURR', how = 'left')
.merge(avg_cred_card_balCat, on = 'SK_ID_CURR', how = 'left'))

del(cred_card_bal, cred_card_balCat, cred_card_balNum, avg_cred_card_balNum, avg_cred_card_balCat)
gc.collect()

print('Preprocessing POS_CASH_balance:')
print("current time: {:.0f}s".format(time.time() - t0))
pos_cash_bal, pos_cash_balCat, pos_cash_balNum = (pd.read_csv('../input/POS_CASH_balance.csv').pipe(LabelEncoding_Cat))

Label3 = [s+'_'+l for s in pos_cash_balNum for l in statGroupNum]
avg_pos_cash_balNum = pos_cash_bal.groupby(['SK_ID_PREV', 'SK_ID_CURR'])[pos_cash_balNum].agg(statGroupNum).groupby(level='SK_ID_CURR').agg('mean').reset_index()          
avg_pos_cash_balNum.columns = ['SK_ID_CURR']+Label3
Label32 = [s+'_'+l for s in pos_cash_balCat for l in ['count','mostFreq', 'mean', 'nunique']]
avg_pos_cash_balCat = pos_cash_bal.groupby(['SK_ID_PREV', 'SK_ID_CURR'])[pos_cash_balCat].agg(statGroupCate).groupby(level='SK_ID_CURR').agg('mean').reset_index()       
avg_pos_cash_balCat.columns = ['SK_ID_CURR']+Label32

allData = (allData.merge(avg_pos_cash_balNum, on = 'SK_ID_CURR', how = 'left')
.merge(avg_pos_cash_balCat, on = 'SK_ID_CURR', how = 'left'))

del(pos_cash_bal, pos_cash_balCat, pos_cash_balNum, avg_pos_cash_balNum, avg_pos_cash_balCat)
gc.collect()
prev, prevCat, prevNum = (pd.read_csv('../input/previous_application.csv').pipe(LabelEncoding_Cat))
 
print('Preprocessing previous_application:')
print("current time: {:.0f}s".format(time.time() - t0))
     
 # Days 365.243 values -> nan
prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
# Add feature: value ask / value received percentage 
prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
prevNum+=['APP_CREDIT_PERC']
Label4 = [s+'_'+l for s in prevNum for l in statGroupNum]
avg_prevNum = prev.groupby('SK_ID_CURR')[prevNum].agg(statGroupNum).reset_index()
avg_prevNum.columns = ['SK_ID_CURR']+Label4
Label42 = [s+'_'+l for s in prevCat for l in ['count','mostFreq', 'mean', 'nunique']]
avg_prevCat = prev.groupby('SK_ID_CURR')[prevCat].agg(statGroupCate).reset_index()
avg_prevCat.columns = ['SK_ID_CURR']+Label42

allData = (allData.merge(avg_prevNum, on ='SK_ID_CURR', how = 'left')
.merge(avg_prevCat, on ='SK_ID_CURR', how = 'left'))
    
del(prev, prevCat, prevNum, avg_prevNum, avg_prevCat)
gc.collect()

inst, instCat, instNum = (pd.read_csv('../input/installments_payments.csv').pipe(LabelEncoding_Cat))

print('Preprocessing installments_payments:')
print("current time: {:.0f}s".format(time.time() - t0))

inst['PAYMENT_PERC'] = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT']
inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
# Days past due and days before due (no negative values)
inst['DPD'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
inst['DBD'] = inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']
inst['DPD'] = inst['DPD'].apply(lambda x: x if x > 0 else 0)
inst['DBD'] = inst['DBD'].apply(lambda x: x if x > 0 else 0)

instNum+=['PAYMENT_DIFF', 'PAYMENT_PERC', 'DPD', 'DBD']
Label5 = [s+'_'+l for s in instNum for l in statGroupNum]
avg_instNum = inst.groupby('SK_ID_CURR')[instNum].agg(statGroupNum).reset_index()
avg_instNum.columns = ['SK_ID_CURR']+Label5
Label52 = [s+'_'+l for s in instCat for l in ['count','mostFreq', 'mean', 'nunique']]
#avg_instCat = inst.groupby('SK_ID_CURR')[instCat].agg(statGroupCate).reset_index()
#avg_instCat.columns = ['SK_ID_CURR']+Label52

allData = (allData.merge(avg_instNum,on = 'SK_ID_CURR', how = 'left'))

del(inst, instCat, instNum, avg_instNum)
gc.collect()

del(Label1,Label2,Label3,Label4, Label5)
del(Label12,Label22,Label32,Label42, Label52)







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


#
featuresPoly = ['INCOME_CREDIT_PERC', 'INCOME_PER_PERSON', 'ANNUITY_INCOME_PERC', 'PAYMENT_RATE']
poly_features = allData[featuresPoly]

poly_transformer = PolynomialFeatures(degree = 2)
poly_transformer.fit(poly_features)
poly_features = poly_transformer.transform(poly_features)
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(featuresPoly))
poly_features['SK_ID_CURR'] = allData['SK_ID_CURR']
allData = allData.merge(poly_features, on = 'SK_ID_CURR', how = 'left')


#del(poly_features)
gc.collect()

allData.drop(labels = ['SK_ID_CURR'], axis = 1, inplace = True)
#allData= allData_archive.copy()
#
#for col in allData_archive:
#    if (allData_archive[col].quantile(0.0001) == allData_archive[col].quantile(0.99)):
#        allData.drop(labels = [col], axis = 1, inplace = True)    


#for col in allData_archive:
#    if (allData_archive[col].count()/len(allData_archive) <= 0.02):
#        allData.drop(labels = [col], axis = 1, inplace = True)    
#   

test_df_output, sub_preds_output, feat_importance = kfold_lightgbm(allData, num_folds= 5, stratified= False, debug= False)

# Write submission file and plot feature importance
test_df = pd.read_csv('../input/application_test.csv')
test_df_out = pd.DataFrame()
test_df_out['SK_ID_CURR'] = test_df['SK_ID_CURR']
test_df_out['TARGET'] = sub_preds_output

test_df_out.to_csv('submission_kernel02.csv', index= False)
#
#     
#train = allData.iloc[:trainLen,:].copy()
#test = allData.iloc[trainLen:, :].copy()
#
#  
##del(allData)
#
#ModelParam = {'objective':'binary',
#                'boosting_type': 'gbdt',
#                'metric': 'auc',
#                'nthread': 8,
#                'shrinkage_rate': 0.025,
#                'min_child_weight': 18,
#                'bagging_fraction': 0.75,
#                'feature_fraction': 0.75,
#                'lambda_l1': 1.5,
#                'lambda_l2': 1,
#                'num_leaves': 36}
#
#
#print('Training model ... \n')
#
#folds = KFold(n_splits = 5, shuffle = True, random_state = 123456)
#
#val_preds = np.zeros(train.shape[0])
#sub_preds = np.zeros(test.shape[0])          
#
#for n_fold, (train_idx, val_idx) in enumerate(folds.split(train)):
#    dtrain = gbm.Dataset(train.iloc[train_idx], y.iloc[train_idx])
#    dval = gbm.Dataset(train.iloc[val_idx], y.iloc[val_idx])
#    cur_gbm = gbm.train(params=ModelParam, train_set=dtrain, num_boost_round=2000, verbose_eval=1000,valid_sets=dval,
#                        valid_names=['train','valid'])
#    val_preds[val_idx] = cur_gbm.predict(train.iloc[val_idx]) 
#    sub_preds+=cur_gbm.predict(test) / folds.n_splits
#    print('Fold %2d AUC : %.6f' %(n_fold + 1, roc_auc_score(y.iloc[val_idx], val_preds[val_idx])))
#    del dtrain, dval
#    gc.collect()
#    
#print('Full AUC score %.6f' % roc_auc_score(y, val_preds))
#print('Outputing results ... \n')
#
#Submission = pd.read_csv('../input/sample_submission.csv')
#Submission['TARGET'] = sub_preds.copy()
#Submission.to_csv('lightgbm_sub.csv', index = False)
#
print('DONE!')
