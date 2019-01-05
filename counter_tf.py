from collections import Counter
import pandas as pd
import string
import numpy as np
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

df_train = pd.read_csv('C:/Users/Owner/Desktop/InformationRetrieval/input/train_original.csv')
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
tr = str.maketrans("", "", string.punctuation)
df_train['question1'] = df_train.question1.apply(lambda x: str(x).lower())
df_train['question2'] = df_train.question2.apply(lambda x: str(x).lower())
df_train['question1'] = df_train.question1.apply(lambda x: str(x).translate(tr))
df_train['question2'] = df_train.question2.apply(lambda x: str(x).translate(tr))

df_test = pd.read_csv('C:/Users/Owner/Desktop/InformationRetrieval/input/train_original.csv')
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
tr = str.maketrans("", "", string.punctuation)
df_test['question1'] = df_test.question1.apply(lambda x: str(x).lower())
df_test['question2'] = df_test.question2.apply(lambda x: str(x).lower())
df_test['question1'] = df_test.question1.apply(lambda x: str(x).translate(tr))
df_test['question2'] = df_test.question2.apply(lambda x: str(x).translate(tr))
# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

print('Most common words and weights: \n')
print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
print('\nLeast common words and weights: ')
print(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
df_train['tf_idf'] = tfidf_train_word_match
df_train['word_match'] = train_word_match
a = 0
for i in range(a,a+10):
    print(df_train.question1[i])
    print(df_train.question2[i])
    print(df_train.tf_idf[i])
    print(df_train.word_match[i])
    print()

x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match
x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)
y_train = df_train['is_duplicate'].values

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['id'] = df_test['id']
sub['is_duplicate'] = p_test
sub.to_csv('log-loss.csv', index=False)