#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:53:09 2018

@author: Venkata Kiran Menda
"""

# Apriori
# Problem : Implemeting algorithm to a store to look for the optimsation of sales...
# like cerals and milk should be in single basket or placing both next to each other

# Importing library
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing a dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Apriori will be expecting an input will be list of list, for that we will be prepraing dataset, so we will be using two for loops

transactions = []
for i in range(0,7501): 
    # lower bound is included but upeer bound is excluded (range is 0 to 7500)
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

 
# Training APriori into dataset
# this library is in folder of datasets
# min length = 2 is bcoz the products in basket is 2
# we need to try with diff values of support and confidence to get better rules which are having sense
    
# mins_support = 3*7/7500 (product purchased 3 times a day * total days in as week )/total transactions
# min_confidencd by default is 0.8 , but to get better rules we will apply strict like 0.2
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)