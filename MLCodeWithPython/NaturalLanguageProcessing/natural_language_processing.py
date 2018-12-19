#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:01:32 2018

@author: Venkata Kiran Menda
"""
# NLP
# Problem: Predicting the reivew is correct or wrong

# Importing library
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

# Importing a dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # 3 for ignoring double quotes

# Cleaning the texts for single review
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) # removed character replaced by space

# Putting all letters in Lower case
review = review.lower()

# Remove non significant words(not useful for predicting positive or negative review) which are like and, or, the etc.,
review = review.split()

# looping through list
# set is using bcoz it is faster and helpful when we are using biiger texts , articles
# Stemming : taking root of the word (root of loved is Love)
# we use stemming to reduce the overhead of sparcity

ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # taking all words which are not in stopswords

# joining words together separated by space
review = ' '.join(review)

#------------- End for Single Review -------------#∫

# Now cleaning the entire 1000 reviews

import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] # creating empty list

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # removed character replaced by space
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # taking all words which are not in stopswords
    # joining words together separated by space
    review = ' '.join(review)
    # Append cleaned review to corpus
    corpus.append(review)
    

# Creating the Bag of words Model by using a process of Tokenisation

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
# Creating the sparse matrix
X = cv.fit_transform(corpus).toarray()


# here we will set parameter max_features for CountVectorizer to get only the relevant words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
# Creating the sparse matrix
X = cv.fit_transform(corpus).toarray()

# to train our dataset we need dependent variable
Y = dataset.iloc[:, 1].values # taking the column index

# Reducing sparcity can be done by decreasing the number of variables or by dimensonality reduction
# Common models used for NLP is Naive Bayes, Decison Tress, RandomForest
# We will use Naive Bayes 

# Splitting dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X, Y, test_size=0.20, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # for train set we have to do first fit and then transform
X_test = sc_X.transform(X_test)


# Fitting Maive Bayes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,  Y_train)


# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

accuracy =  54+87/200

    






