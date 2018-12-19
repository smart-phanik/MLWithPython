#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:51:29 2018

@author: Venkata Kiran Menda
"""
# Random Forest


import numpy as np # this includes Mathematical Libraries
import matplotlib.pyplot as plt # this is used for plotting
import pandas as pd # this is used for import/manage datasets

# Importing a dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Creating the matrix of Independent variables vector
X = dataset.iloc[:, 1:2].values 

# Creating the dependent variables vector
Y = dataset.iloc[:, 2].values 


# Splitting dataset into Training set and Test set
# we are not creating training and test set because of less data only have 10 values
"""
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X, Y, test_size=1/3, random_state=0)"""


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # for train set we have to do first fit and then transform
X_test = sc_X.transform(X_test)"""



# Fitting the Regression Model into dataset
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, Y)


# Predicting a new result with Regression regression for the level 6.5
Y_pred = regressor.predict(6.5)

# For non continuous regression mode we should use higher resolution technicqu


# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

