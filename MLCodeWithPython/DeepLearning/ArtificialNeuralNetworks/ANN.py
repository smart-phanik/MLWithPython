#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 01:24:02 2018

@author: Venkata Kiran Menda
"""

# Installing Theaon,Tensorflow and keras



# Commands to follow:
""" conda install -c conda-forge keras only if python version < 3.7, for 3.7 there is no tensorflow package 

"""

# Installing Keras
# pip install --upgrade keras



# Problem : Customer Churn in Banks , 0 -> customer stays, 1-> customer leaves the bank
# this belongs to a classification problem

# Part 1 : Data Preprocessing



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


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # for train set we have to do first fit and then transform
X_test = sc_X.transform(X_test)


# Part 2 : Now Make ANN

# Importing the Keras Library
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN (Defining it as sequence of layers or Defining it as graph)
# Here we use layers
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Choosing number of layers is based on avg(no of input layers + no of output layers)
# for this eg : No of input layers = 11 (bcoz it contain 11 independent variable)
# output layer = 1  bcoz of 1 dependent variable
# no fo layers = (11+1)/2 = 6
# relu - rectifier funtion for hidden layer
# we need to specify input_dim only for first hidden layer
classifier.add(Dense(units = 6,init = 'uniform', activation = 'relu',input_dim = 11))

# Adding Second Layer
classifier.add(Dense(units = 6,init = 'uniform', activation = 'relu'))

# Adding output layer, so we have to use Sigmoid function for output layer
# if we have independent variable whcih contain 3 or more categorical variables the we have to choose softmax which is a sigmoid function

classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling ANN -> applying Stochastic Gradient
# optimiser -> nothing but algorithm which we are going to use, so we use Stochastic Gradient which contains famous type adam
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN into Training Set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)


# Part 3 Making the prediction and evaluvating the model

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
#cpnverting the predicting probablities into True or false like 0 or 1
Y_pred = (Y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)