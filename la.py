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
#Fill NAN  
df_train['question1'].fillna('OK', inplace=True)
df_train['question2'].fillna('OK', inplace=True)

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

########
qi = pd.DataFrame(list(zip(df_train['qid1'],df_train['question1'])) + list(zip(df_train['qid2'],df_train['question2']))) 
qi.columns = ['qid','que']   
bloblist=qi.drop_duplicates(['qid','que'])
bloblist = bloblist.set_index(['qid'])
bloblist = bloblist.sort_values('qid')


bloblist.values.astype('U')


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(bloblist['que'].values.astype(str))

L=[]
for k in range (len(df_train)):
    i=df_train.loc[k,'qid1']
    j=df_train.loc[k,'qid2']

    
    from sklearn.metrics.pairwise import paired_cosine_distances
    ddsim_matrix=paired_cosine_distances(tfidf_matrix[i-1], tfidf_matrix[j-1])
    
    L.append(ddsim_matrix[0])
    print (k)
I=df_train.assign(cosine=L)
print(I)
I.to_csv('ok.csv', index=False)


chunk_size = 500
matrix_len = tfidf_matrix.shape[0] # Not sparse numpy.ndarray

def similarity_cosine_by_chunk(start, end):
    if end > matrix_len:
        end = matrix_len
    return cosine_similarity(X=tfidf_matrix[start:end], Y=tfidf_matrix) # scikit-learn function

for chunk_start in range(0, matrix_len, chunk_size):
    print(chunk_start)
    cosine_similarity_chunk = similarity_cosine_by_chunk(chunk_start, chunk_start+chunk_size)

A=[]
for k in range (len(df_train)):
    i=df_train.loc[k,'qid1']
    j=df_train.loc[k,'qid2']
    
    from sklearn.metrics.pairwise import cosine_similarity
    ddsim_matrix = cosine_similarity(tfidf_matrix[i-1], tfidf_matrix[j-1])
    
    A.append(ddsim_matrix[0][0])
    print (k)
A=df_train.assign(cosine=L)
print(A)
A.to_csv('cosine.csv', index=False)

# SVD
from scipy.sparse.linalg import svds
u, s, vt = svds(tfidf_matrix, k=3)
M = np.dot(tfidf_matrix,vt.transpose())



#TOKENS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

stop_words = set('?')
tokens=[]
for i in range(1,len(bloblist)):
    tokens.append(word_tokenize(bloblist.loc[i,'que']))
    print(i)

filtered_sentence=[]
for i in range(len(bloblist)-1):
    transposed_row = []
    for w in tokens[i]:
        if w not in stop_words:
            transposed_row.append(w)
    filtered_sentence.append(transposed_row)
    print(i)

#Common words 
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
#################

A=[]
for i in range(len(tokens)):
    words=tokens[i]
    tagged = nltk.pos_tag(words)
    A.append(tagged)
    print(i)
df = pd.DataFrame({'col':A})
df.to_csv('LLL.csv', index=False)
 = pd.read_csv('C:/Users/harter/Desktop/retrival/LLL.csv')

M=[]
E=[]
Y=[]
U=[]
for i,j in zip(df_train['qid1'],df_train['qid2']):
   Y=[]
   U=[]
   for k in A[i-1]:
       if (k[1]=='NN'or k[1]=='NNP'or k[1]=='NNS' or k[1]=='NNPS'):
           Y.append(k[0])
   for k in A[j-1]:
       if (k[1]=='NN'or k[1]=='NNP'or k[1]=='NNS' or k[1]=='NNPS'):
           U.append(k[0])
   if (set(U).intersection(Y)):
       print(i)
       M.append(set(U).intersection(Y))

from sklearn import HashingVectoriser
