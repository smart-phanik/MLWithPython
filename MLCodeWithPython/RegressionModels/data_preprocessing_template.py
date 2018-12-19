#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:40:50 2018

@author: Venkata Kiran Menda
"""

#Please look into data_preprocessing_full_steps file to 
#understand more steps which has been removed from this to make this code as resualbe


#Importing a Library

import numpy as np # this includes Mathematical Libraries
import matplotlib.pyplot as plt # this is used for plotting
import pandas as pd # this is used for import/manage datasets

# Importing a dataset
dataset = pd.read_csv('Data.csv')

# Creating the matrix of Independent variables vector
X = dataset.iloc[:, :-1].values 

# Creating the dependent variables vector
Y = dataset.iloc[:, 3].values 


# Splitting dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X, Y, test_size=0.2, random_state=0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # for train set we have to do first fit and then transform
X_test = sc_X.transform(X_test)"""