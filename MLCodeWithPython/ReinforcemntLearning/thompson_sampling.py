#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:27:51 2018

@author: Venkata Kiran Menda
"""

# Thompson Sampling
# Problem : Multi Armed Bandit Problem

# Importing library
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import random


# Importing a dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement Thompson Sampling

# N is totla no of rounds
N = 10000
d = 10  # No of ads
numbers_of_rewards_0 = [0] * d
numbers_of_rewards_1 = [0] * d
total_reward = 0
ads_selected = [] # gives ads selected at each round
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
        ads_selected.append(ad)
        reward = dataset.values[n, ad]
        if reward == 1:
            numbers_of_rewards_1[ad] =  numbers_of_rewards_1[ad] + 1
        else:
            numbers_of_rewards_0[ad] =  numbers_of_rewards_0[ad] + 1
        total_reward = total_reward + reward
        
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()