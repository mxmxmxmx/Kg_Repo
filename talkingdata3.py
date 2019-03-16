# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:56:40 2018

@author: MXU29
"""

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os

DEBUG = True
KERNEL = False
LEN_VAL = 250000

def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):
    df = train_df
    agg_suffix='nextClick'
    agg_type='float32'
    
    GROUP_BY_NEXT_CLICKS = [
            {'groupby': ['ip', 'os', 'device']}
            ]
    spec = GROUP_BY_NEXT_CLICKS[0]
    print(">> \nExtracting %s time calculation features...\n" %(agg_suffix))
    new_feature = '{}_{}'.format('_'.join(spec['groupby']), agg_suffix)   
    all_features = spec['groupby'] + ['click_time']
    df[new_feature] = df[]
    
def preProcessing(data):
    data['hour'] = pd.to_datetime(data.click_time).dt.hour.astype('int8')
    data['day'] = pd.to_datetime(data.click_time).dt.day.astype('int8') 
    data['minute'] = pd.to_datetime(data.click_time).dt.minute.astype('int8')
    data = do_next_Click( data,agg_suffix='nextClick', agg_type='float32'  ); gc.collect()
    data = do_prev_Click( data,agg_suffix='prevClick', agg_type='float32'  ); gc.collect()  ## Removed temporarily due RAM sortage. 
    
#    data = do_countuniq( data, ['ip'], 'channel' ); gc.collect()
    data = do_countuniq( data, ['ip', 'device', 'os'], 'app'); gc.collect()
    data = do_countuniq( data, ['ip'], 'hour' ); gc.collect()
    data = do_countuniq( data, ['ip'], 'app'); gc.collect()
    data = do_countuniq( data, ['ip', 'app'], 'os'); gc.collect()
    data = do_countuniq( data, ['ip'], 'device'); gc.collect()
    data = do_countuniq( data, ['app'], 'channel'); gc.collect()
    data = do_cumcount( data, ['ip'], 'os'); gc.collect()
    data = do_cumcount( data, ['ip', 'device', 'os'], 'app'); gc.collect()
    data = do_count( data, ['ip', 'hour'] ); gc.collect()
    data = do_count( data, ['ip', 'app']); gc.collect()
    data = do_count( data, ['ip', 'app', 'os']); gc.collect()
    data = do_count( data, ['ip', 'device', 'app', 'os']); gc.collect()
    data = do_count( data, ['app', 'channel']); gc.collect()
#    data = do_var( data, ['ip', 'day', 'channel'], 'hour'); gc.collect()
#    data = do_var( data, ['ip', 'app', 'os'], 'hour'); gc.collect()
    data = do_var( data, ['ip', 'app', 'channel'], 'day'); gc.collect()
#    data = do_mean( data, ['ip', 'app', 'channel'], 'hour' ); gc.collect()

    del data['day']
    gc.collect()
    
    return (data)
    
def Clssification(fromRow, toRow):
    fromRow, toRow = 1, 1000000
    
    dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint8',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }
        
    print('starting to load training data...', fromRow,'to', toRow)
    
    if KERNEL:
        workingDir= 'G:/My Drive/Shared Drive/data/Kaggle/TalkingData/input/'
        os.chdir(workingDir)
        
    train_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], skiprows=range(1,fromRow), 
                           nrows=toRow-fromRow, dtype=dtypes, 
                           usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    
    print('loading test data...')
    if DEBUG:
        test_df = pd.read_csv("../input/test.csv", nrows=1000000, parse_dates=['click_time'], 
                              dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv("../input/test.csv", parse_dates=['click_time'], 
                              dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    len_train = len(train_df)
    val_df = train_df[(len_train - LEN_VAL):]
    gc.collect()
    
    for data in [train_df, val_df, test_df]:
        data = preProcessing(data)
