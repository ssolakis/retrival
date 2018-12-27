# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 19:09:27 2018

@author: harter
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns


pal = sns.color_palette()

print('# File sizes')
for f in os.listdir('C:/Users/harter/Desktop/retrival/'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('C:/Users/harter/Desktop/retrival/' + f) / 1000000, 2)) + 'MB')

df_train = pd.read_csv('C:/Users/harter/Desktop/retrival/train_original.csv',nrows=364290)
df_test = pd.read_csv('C:/Users/harter/Desktop/retrival/train_original.csv',skiprows = range(1,364291))
df_train.head()


print('Total number of question pairs for training: {}'.format(len(df_train)))
print("\n")
print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
print("\n")
qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
print("\n")
print('Total number of questions in the training data: {}'.format(len(np.unique(qids))))
print("\n")
print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))
print("\n")
"PLOTTTTTT"
plt.figure(figsize=(12, 5))
print("\n")
plt.hist(qids.value_counts(), bins=50)
print("\n")
plt.yscale('log', nonposy='clip')
print("\n")
plt.title('Log-Histogram of question appearance counts')
print("\n")
plt.xlabel('Number of occurences of question')
print("\n")
plt.ylabel('Number of questions')
print("\n")


from sklearn.metrics import log_loss
p = df_train['is_duplicate'].mean() # Our predicted probability
print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))
sub = pd.DataFrame({'test_id': df_test['id'], 'is_duplicate': p})
sub.to_csv('naive_submission.csv', index=False)
sub.head()


print('Total number of question pairs for testing: {}'.format(len(df_test)))













