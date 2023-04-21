# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:42:19 2023

@author: jeffe
"""

### IMPLEMENTATION OF K-PROTOTYPES: A COMBINATION OF K-MEANS AND K-MODES 

## DATA PREPARATION

# import packages

import pandas as pd # working with data
import numpy as np # working with arrays
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization

# import data

df = pd.read_csv(r'C:\Users\jeffe\Desktop\PITT\ECON 2841 Capstone\capstone_data.csv')
print(df.head())

# check missing values in data

df.isna().sum()

# summary statistics

stat = df.describe()

# check variable data types

df.dtypes

# drop features that are not behavioral

df = df.drop(['customer_id', 'surname', 'age', 'geog', 'gender', 'jobs','est_salary', 'housing', 'products'], axis = 1)

## DATA TRANSFORMATION

df['saving'] = df['saving'].fillna('none') # replace na values with a new classification
df['checking'] = df['checking'].fillna('none') # replace na values with a new classification

# encode binary variables with strings
mapping = {0: 'no', 1: 'yes'}
df['cred_card'] = df['cred_card'].replace(mapping)
df['active'] = df['active'].replace(mapping)

# assign classifications for customer based on th length of tenure 0-3 = new, 4-7 long term, 8-10 loyal
df['tenure'] = np.where(df['tenure'].between(0,3), 1, df['tenure']) 
df['tenure'] = np.where(df['tenure'].between(4,7), 2, df['tenure']) 
df['tenure'] = np.where(df['tenure'].between(8,10), 3, df['tenure'])
mapping_tenure = {1: 'new', 2: 'long term', 3: 'loyal'}
df['tenure'] = df['tenure'].replace(mapping_tenure) # tenure is transformed from integer feature to categorical feature

# assign classifications for 'balance'
plt.hist(x = 'balance', bins = 50, data=df)
plt.show() # look at the distribution first
df['balance'] = np.where(df['balance'].between(1,120000), 1, df['balance'])
df['balance'] = np.where(df['balance'].between(120001,260000), 2, df['balance'])
mapping_bal = {0: 'no balance', 1: 'good balance', 2:'excellent balance'}
df['balance'] = df['balance'].replace(mapping_bal)

# MODEL IMPLEMENTATION - K PROTOTYPE

# check hopkins stat for cluster tendency, Hopkins Statistic is a way of measuring the cluster tendency of a data set.
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H

num_features = df.select_dtypes(include=[np.number]).columns
hopkins(df[num_features]) # Hopkins test gives 0.95, which is very high and means the continuous variables have a high tendency to be clustered.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
df[['credit_sco','credit_amo']]= scaler.fit_transform(df[['credit_sco','credit_amo']]) # standardize continuous variables

from kmodes.kprototypes import KPrototypes
cost = []
for num_clusters in list(range(2,7)):
    kproto = KPrototypes(n_clusters=num_clusters, init='Cao',random_state=42,n_jobs=10, max_iter=15, n_init=100) 
    kproto.fit_predict(df, categorical=[2,3,4,5,6,7])
    cost.append(kproto.cost_)

plt.plot(cost)
plt.xlabel('K (Number of Clusters)')
plt.ylabel('Cost')
plt.show

kproto = KPrototypes(n_clusters=4, init='Huang', verbose=0, random_state=42,max_iter=20, n_init=100, n_jobs=-2, gamma=.25) 
clusters = kproto.fit_predict(df, categorical=[2,3,4,5,6,7])

# convert assigned clusters to a dataframe and add it to the original dataframe 
clusters = pd.DataFrame(clusters)
clusters.columns = ['clusters']
df = pd.concat([df, clusters], axis=1)

# calculate means and compare between clusters
mapping_clusters = {0:'cluster one', 1:'cluster two', 2:'cluster three', 3:'cluster four'}
df['clusters'] = df['clusters'].replace(mapping_clusters)

# calculate within-cluster variations:
credit_sco_means = df.groupby('clusters')['credit_sco'].mean()
credit_amo_means = df.groupby('clusters')['credit_amo'].mean()
tenure_freq = df.groupby(['clusters','tenure']).size().reset_index(name='frequency')
balance_freq = df.groupby(['clusters','balance']).size().reset_index(name='frequency')
cred_card_freq = df.groupby(['clusters','cred_card']).size().reset_index(name='frequency')
active_freq = df.groupby(['clusters','active']).size().reset_index(name='frequency')
saving_freq = df.groupby(['clusters','saving']).size().reset_index(name='frequency')
checking_freq = df.groupby(['clusters','checking']).size().reset_index(name='frequency')

# import original df and merge the id and name columns

original_df = pd.read_csv(r'C:\Users\jeffe\Desktop\PITT\ECON 2841 Capstone\capstone_data.csv')
df['credit_sco'] = original_df['credit_sco']
df['credit_amo'] = original_df['credit_amo']
