#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:54:48 2018

@author: Venkata Kiran Menda
"""

# UCB- Upper Confidence Bound
# Problem : Mutli Armed Bandit Problem
# dataset contains differenct version of ads to put in a social network
# they dont know whcih ad to put in social network to get more clicks (Click through rate = CTR)

# Importing library
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import math

# Importing a dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection to understand proper basic information 
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# Now Implementing UCB algorithm
# step1
numbers_of_selection = [0] * d  # creates a vector of zero ads
sum_of_rewards = [0] * d  # in first round sum of rewards is zero

# step2
# N is totla no of rounds
N = 10000
d = 10  # No of ads
total_reward = 0
ads_selected = [] # gives ads selected at each round
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selection[i] > 0):
            average_reward = sum_of_rewards[i]/numbers_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
        ads_selected.append(ad)
        reward = dataset.values[n, ad]
        numbers_of_selection[ad] = numbers_of_selection[ad]+1
        sum_of_rewards[ad] = sum_of_rewards[ad] + reward
        total_reward = total_reward + reward
        
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()