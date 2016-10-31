#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
  Author:   --<>
  Purpose: 
  Created: 10/29/2016
"""
# Kaggle projects: Allstate Claims Severity  https://www.kaggle.com/c/allstate-claims-severity

import unittest

# General libraries.
import re, os, sys
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
#from sklearn.feature_extraction import 
from sklearn import preprocessing

from sklearn.utils import shuffle

from sklearn.metrics import mean_absolute_error

from sklearn import linear_model

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

os.chdir("c:/Users/James/SkyDrive/Documents/MIDS-Berkeley/W207 Machine Learning/AllstateChallenge/")

    
data_dir = '.\\data_in'
train_csv = 'train.csv'


df_data = pd.read_csv( os.path.join(data_dir, train_csv) )

df_data_encoded = df_data.copy()

# encoding into the categorical value
le = preprocessing.LabelEncoder()
for c in df_data_encoded.columns:
    if c.find('cat') >=0: # -1: substring not found, >=0, starting index
        df_data_encoded[c] = le.fit_transform(df_data_encoded[c])

np.random.seed(100)


col = list(df_data_encoded.columns)
col.remove('loss')
col.remove('id')

X = df_data_encoded[col]
y = np.log10(df_data_encoded.loss) # conver the loss to log scale
id = df_data_encoded.id

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split( X, y, id, test_size=0.33, random_state=1)

#id_train, x_train,  y_train = shuffle( df_train.id, df_train[col] , df_train.loss, random_state=0)

pd.options.mode.chained_assignment = None


lr = linear_model.LinearRegression()

# Train the model using the training sets
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

s = cross_val_score(lr, X_train, y_train, scoring='neg_mean_absolute_error')

print('mean_absolute_error on training data: {0}'.format(s))

mae = mean_absolute_error(y_test, y_pred)
print('mean_absolute_error on test data {0}'.format(mae))

print('end')
#if __name__ == '__main__':
    #unittest.main()
