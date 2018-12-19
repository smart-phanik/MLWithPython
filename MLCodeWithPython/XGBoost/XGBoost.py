#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:18:18 2018

@author: Venkata Kiran Menda
"""

# XGBoost


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

# Importing a dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Creating the matrix of Independent variables vector
X = dataset.iloc[:, 3:13].values   # taking indexes from 3 to 12(Upper bound exluded)

# Creating the dependent variables vector
Y = dataset.iloc[:, 13].values # For Purchased column 

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])

# Avoiding Dummy variable trap for Country, since contains 3 values, but gender not needed bcoz it contains two values 
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]



# Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X, Y, test_size=0.25, random_state=0)

# Fitting XGBoost to the training set
from xgboost import XGBClassifier
