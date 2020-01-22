#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:24:03 2020

@author: duck777
"""
#import library untuk python
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
#import dataset pada variable
dataset = pd.read_csv("Data.csv")
#membuat variable untuk data independen
X = dataset.iloc[:, :-1]
#membuat variable untuk data dependen
Y = dataset.iloc[:, 3]
#modelling untuk mengisi data kosong / missing data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(X.iloc[:, 1:3])
#memasukkan model ke dalam variable independen yang punya nilai NaN
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])