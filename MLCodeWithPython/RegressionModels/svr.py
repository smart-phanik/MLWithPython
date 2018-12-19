#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:52:25 2018

@author: Venkata Kiran Menda
"""

# SVR

#Importing a Library

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X) 
Y = sc_Y.fit_transform(Y)



# Fitting the SVR into dataset
from sklearn.svm import SVR

# Create your new Regressor
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)


# Predicting a new result with Regression regression for the level 6.5
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]))))

#  Visualise the  SVR resutls

plt.scatter(X,Y,color = 'red')
plt.plot(X, regressor.predict(X),color = 'blue')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("position level")
plt.ylable("Salary")
plt.show()


# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()