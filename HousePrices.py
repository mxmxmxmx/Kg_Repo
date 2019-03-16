# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:36:02 2018

@author: MXU29
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

os.chdir('G:\\My Drive\\Shared Drive\\data\Kaggle\\House_Prices\\input')

df_train = pd.read_csv('../input/train.csv')

df_train.columns

df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
data.plot.scatter(x=var, y = 'SalePrice', ylim = (0, 800000))

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)
f, ax = plt.subplots(figsize=(16, 8))
fig  = sns.boxplot(x=var, y = 'SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)

corrmat = df_train.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square=True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
