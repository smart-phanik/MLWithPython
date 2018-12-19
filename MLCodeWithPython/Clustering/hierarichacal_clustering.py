#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 20:57:12 2018

@author: Venkata Kiran Menda
"""

# Hierarchical Clustering

#Importing a Library

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing a dataset
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3, 4]].values 


# using the dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
# linkage is an algortihm
# ward is a method to minimise variance with in each cluster
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian Distance')
plt.show()

# Fitting Hierarical clustering to dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
Y_hc = hc.fit_predict(X)

# Visualise the clusters

plt.scatter(X[Y_hc == 0, 0], X[Y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[Y_hc == 1, 0], X[Y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[Y_hc == 2, 0], X[Y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[Y_hc == 3, 0], X[Y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[Y_hc == 4, 0], X[Y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

