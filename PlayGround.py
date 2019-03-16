# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 22:28:11 2018

@author: Meng Xu
Used to try different small chunk of codes, just for fun! Play it :)
"""

import pandas as pd
from IPython.display import display
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt

data = {'Platoon': ['A','A','A','A','A','A','B','B','B','B','B','C','C','C','C','C'],
       'Casualties': [1,4,5,7,5,5,6,1,4,5,6,7,4,6,4,6]}
df = pd.DataFrame(data)

raw_data = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])

raw_data = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'], 
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])


raw_data = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
df_n = pd.DataFrame(raw_data, columns = ['subject_id','test_id'])

