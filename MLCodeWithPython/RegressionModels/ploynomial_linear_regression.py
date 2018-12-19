#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:00:07 2018

@author: Venkata kiran Menda
"""
# Polynomial Regression
# Problem : Predicting the salay basing on Level to predict employee is bluffing or not

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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # for train set we have to do first fit and then transform
X_test = sc_X.transform(X_test)"""


# Fitting Linear Regression into dataset
# we are creating two regressions bcoz to compare both

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


# Fitting Polynomial Regression into dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # we can increase degree basing on the curve
X_poly = poly_reg.fit_transform(X)

# Fitting Ploynomial regression  into Linear Regression
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)


# Visualise the Linear Regression resutls
plt.scatter(X,Y,color = 'red')
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("position level")
plt.ylable("Salary")
plt.show()


#  Visualise the Polynomial Regression resutls

plt.scatter(X,Y,color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("position level")
plt.ylable("Salary")
plt.show()


# To go for more advanced plot and good resolution we are doing this but for problem upto above 
# code is enough



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # we can increase degree basing on the curve
X_poly = poly_reg.fit_transform(X)

# Fitting Ploynomial regression  into Linear Regression
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

X_grid = np.arange(min(X), max(X), 0.1) # this gives vector but we need to give matrix
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("position level")
plt.ylable("Salary")
plt.show()


# Predicting a new result with Linear regression for the level 6.5
lin_reg.predict(6.5)


# Predicting a new result with Polynomial regression for the level 6.5
lin_reg2.predict(poly_reg.fit_transform(6.5))


