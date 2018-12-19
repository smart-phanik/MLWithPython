#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:11:22 2018

@author: nisum Venkata kirna menda
"""
# Mutliple Linear regression
# Problem Statement: Model needs to predict the profit basing on R&D Spend,Administration, Marketing Spend,State

#Importing a Library

import numpy as np # this includes Mathematical Libraries
import matplotlib.pyplot as plt # this is used for plotting
import pandas as pd # this is used for import/manage datasets

# Importing a dataset
dataset = pd.read_csv('50_Startups.csv')

# Creating the matrix of Independent variables vector
X = dataset.iloc[:, :-1].values 

# Creating the dependent variables vector
Y = dataset.iloc[:, 4].values 

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


# Avoiding the dummy variables Trap
X = X[:, 1:]


# Splitting dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X, Y, test_size=0.2, random_state=0)

# Fitting multiple linear regression into Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# Building the model using Backward Elimination

import statsmodels.formula.api as sm
# we are appending a column of 1's for entire matrix because to satisfy formula constant
# b0*x0 +b1*x1 +..... + bn*xn
# Here we know x0 is always 1
X = np.append(arr =  np.ones((50,1)).astype(int), values = X, axis = 1)

# Creating an optimal matrix which contain only independent variables
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing the independent variable which is having more significance level 0.05
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# Repeating above step to get correct model which contains independent variable < 0.05

# Removing the independent variable which is having more significance level 0.05
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()





