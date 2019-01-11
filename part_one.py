import pandas as pd
import numpy as np
import pyemd
import scipy
import math
import gensim
import nltk
import editdistance
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from scipy import spatial
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

df2 = pd.read_csv('C:/Users/Owner/Desktop/InformationRetrieval/input/train_original.csv')
df2.drop(df2.columns[[0,1,2]], axis=1, inplace=True)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 500)

#αυτες οι ερωτησεις εχουν label 1
q1_dup = df2.loc[df2['is_duplicate'] == 1, 'question1']
q2_dup = df2.loc[df2['is_duplicate'] == 1, 'question2']
q1_dup = q1_dup.to_frame()
q2_dup = q2_dup.to_frame()
print(pd.concat([q1_dup,q2_dup], axis=1)[:10])

#αυτες οι ερωτησεις εχουν label 0
q1_nondup = df2.loc[df2['is_duplicate'] == 0, 'question1']
q2_nondup = df2.loc[df2['is_duplicate'] == 0, 'question2']
q1_nondup = q1_nondup.to_frame()
q2_nondup = q2_nondup.to_frame()
print(pd.concat([q1_nondup,q2_nondup], axis=1)[:10])

#μεθοδος προεπεξεργασιας - υποδειξεις απο kernel στο kaggle
def preprocessing_text(s):
    import re
    s = re.sub(r"[^A-Za-z0-9^,\*+-=]", " ",s)
    s = re.sub(r"(\d+)(k)", r"\g<1>000", s)
    s = re.sub(r"\;"," ",s)
    s = re.sub(r"\:"," ",s)
    s = re.sub(r"\,"," ",s)
    s = re.sub(r"\."," ",s)
    s = re.sub(r"\<"," ",s)
    s = re.sub(r"\^"," ",s)
    s = re.sub(r"(\d+)(/)", "\g<1> divide ", s)
    s = re.sub(r"\/"," ",s)
    s = re.sub(r"\+", " plus ", s)
    s = re.sub(r"\-", " minus ", s)
    s = re.sub(r"\*", " multiply ", s)
    s = re.sub(r"\=", "equal", s)
    s = re.sub(r"What's", "What is ", s)
    s = re.sub(r"what's", "what is ", s)
    s = re.sub(r"Who's", "Who is ", s)
    s = re.sub(r"who's", "who is ", s)
    s = re.sub(r"\'s", " ", s)
    s = re.sub(r"\'ve", " have ", s)
    s = re.sub(r"can't", "cannot ", s)
    s = re.sub(r"n't", " not ", s)
    s = re.sub(r"\'re", " are ", s)
    s = re.sub(r"\'d", " would ", s)
    s = re.sub(r"\'ll", " will ", s)
    s = re.sub(r"'m", " am ", s)
    s = re.sub(r"or not", " ", s)
    s = re.sub(r"What should I do to", "How can I", s)
    s = re.sub(r"How do I", "How can I", s)
    s = re.sub(r"How can you make", "What can make", s)
    s = re.sub(r"How do we", "How do I", s)
    s = re.sub(r"How do you", "How do I", s)
    s = re.sub(r"Is it possible", "Can we", s)
    s = re.sub(r"Why is", "Why", s)
    s = re.sub(r"Which are", "What are", s)
    s = re.sub(r"What are the reasons", "Why", s)
    s = re.sub(r"What are some tips", "tips", s)
    s = re.sub(r"What is the best way", "best way", s)
    s = re.sub(r"e-mail", "email", s)
    s = re.sub(r"e - mail", "email", s)
    s = re.sub(r"US", "America", s)
    s = re.sub(r"USA", "America", s)
    s = re.sub(r"us", "America", s)
    s = re.sub(r"usa", "America", s)
    s = re.sub(r"Chinese", "China", s)
    s = re.sub(r"india", "India", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip()
    return s


#το dataframe μετα την προεπεξεργασια
df2['question1'] = df2['question1'].astype(str)
df2['question1'] = df2['question1'].map(lambda x: preprocessing_text(x))
df2['question2'] = df2['question2'].astype(str)
df2['question2'] = df2['question2'].map(lambda x: preprocessing_text(x))


#μεθοδος για αφαιρεση των stowords
def remove_stopwords(string):
    word_list = [word.lower() for word in string.split()]
    from nltk.corpus import stopwords
    stopwords_list = list(stopwords.words("english"))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    return ' '.join(word_list)


#το dataframe μετα την αφαιρεση των stopwords
df2['question1'] = df2['question1'].astype(str)
df2['q1_without_stopwords'] = df2['question1'].apply(lambda x: remove_stopwords(x))
df2['question2'] = df2['question2'].astype(str)
df2['q2_without_stopwords'] = df2['question2'].apply(lambda x: remove_stopwords(x))

print(df2.head())


#επιστρεφει την αναγολια χαρακτηρων μεταξυ του ζευγαριου των ερωτησεων
def get_char_length_ratio(row):
    return len(row['question1'])/max(1,len(row['question2']))


df2['char_length_ratio'] = df2.apply(lambda row: get_char_length_ratio(row), axis=1)


#μεθοδος για επιστροφη των συνονυμων μιας λεξης
def get_synonyms(word):
    from nltk.corpus import wordnet as wn
    synonyms = []
    if wn.synsets(word):
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
    return list(set(synonyms))


#μεθοδος που επιστρεφει τις κοινες λεξεις μεταξυ του ζευγαριου των ερωτησεων
def get_row_syn_set(row):
    import nltk
    syn_set = [nltk.word_tokenize(row)]
    for token in nltk.word_tokenize(row):
        if get_synonyms(token):
            syn_set.append(get_synonyms(token))
    return set([y for x in syn_set for y in x])


#εφαρμογη της παραπανω μεθοδου στο dataframe - θα χρειαστει το αρχειο GoogleNews-vectors-negative300.bin.gz
df2['q1_tokens_syn_set'] = df2['q1_without_stopwords'].map(lambda row: get_row_syn_set(row))

df2['num_syn_words'] = df2.apply(lambda x:
                                 len(x['q1_tokens_syn_set'].intersection(set(nltk.word_tokenize(x['question2'])))), axis=1)

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/Owner/Desktop/InformationRetrieval/input/GoogleNews-vectors-negative300.bin.gz", binary=True)
word2vec_model.init_sims(replace=True)


#υπολογισμος word-movers-distance με βαση τα διανυσματα των λεξεων απο το dataset της Google
def get_wmd(string1,string2):
    return word2vec_model.wmdistance(string1.split(), string2.split())


df2['wmd'] = df2.apply(lambda row: get_wmd(row['q1_without_stopwords'],row['q2_without_stopwords']), axis=1)


#aποσταση Levenshtein - ο ελαχιστος αριθμος "αλλαγων" ωστε τα δυο string να ειναι ιδια
def get_Levenshtein(string1,string2):
    return editdistance.eval(string1,string2)

df2['Lev_dist'] = df2.apply(lambda row: get_Levenshtein(row['q1_without_stopwords'],row['q2_without_stopwords']),axis = 1)


def get_ave_token_length(string):
    import nltk
    sum_length = 0
    for token in nltk.word_tokenize(string):
        sum_length+=len(token)
    ave_length = sum_length/max(1,len(nltk.word_tokenize(string)))
    return ave_length


df2['q1_ave_token_length'] = df2['q1_without_stopwords'].apply(lambda x: get_ave_token_length(x))
df2['q2_ave_token_length'] = df2['q2_without_stopwords'].apply(lambda x: get_ave_token_length(x))
df2['diff_ave_token_length'] = abs(df2['q2_ave_token_length'] - df2['q1_ave_token_length'])
df2['diff_num_tokens'] = df2.apply(lambda row: abs(len(nltk.word_tokenize(row['q1_without_stopwords']))
                                                   - len(nltk.word_tokenize(row['q2_without_stopwords']))),
                                   axis = 1)


#δημιουργια του συνολου των λεξεων και εφαρμογη TfidfVectorizer
def generate_corpus(dataframe):
    corpus = []
    corpus.append(dataframe['q1_without_stopwords'].tolist())
    corpus.append(dataframe['q2_without_stopwords'].tolist())
    return [x for y in corpus for x in y]


corpus = generate_corpus(df2)
tf = TfidfVectorizer(analyzer='word',
                     min_df = 0, stop_words = 'english', sublinear_tf=True)
tfidf_matrix = tf.fit_transform(corpus)
feature_names = tf.get_feature_names()

feature_index = tfidf_matrix[0,:].nonzero()[1]
tfidf_scores = zip(feature_index, [tfidf_matrix[0, x] for x in feature_index])
print(tfidf_matrix)
feature_index = tfidf_matrix[0,:].nonzero()[1]
tfidf_scores = zip(feature_index, [tfidf_matrix[0, x] for x in feature_index])
for word, score in [(feature_names[i], s) for (i, s) in tfidf_scores]:
    print(word, score)


#ομοιοτητα συνιμητονου μεταξυ των διανυσματων
def cosine_similarity(vector1,vector2):
    vector1 = csr_matrix(vector1)
    vector2 = csr_matrix(vector2)
    result = vector1.dot(vector2.transpose())/(norm(vector1)*norm(vector2))
    return result.A[0][0]

#λιστα με ομοιοτητες
def get_cos_similarity_list(tfidf_matrix,dataframe):
    cos_list = []
    tfidf_matrix = csr_matrix (tfidf_matrix)
    for x in range(0,len(dataframe)):
        cos_list.append(cosine_similarity(tfidf_matrix[x,:],tfidf_matrix[x+len(dataframe),:]))
    return cos_list


cosine_similarity = get_cos_similarity_list(tfidf_matrix,df2)
df2['cos_similarity'] = pd.Series(cosine_similarity).values


def euclidean_distance(vector1,vector2):
    vector1 = csr_matrix(vector1)
    vector2 = csr_matrix(vector2)
    result = norm(vector1-vector2)
    return result

def get_euclidean_list(tfidf_matrix,dataframe):
    euclidean_list = []
    tfidf_matrix = csr_matrix (tfidf_matrix)
    for x in range(0,len(dataframe)):
        euclidean_list.append(euclidean_distance(tfidf_matrix[x,:],tfidf_matrix[x+len(dataframe),:]))
    return euclidean_list

euclidean_list = get_euclidean_list(tfidf_matrix,df2)
df2['euclidean_dis'] = pd.Series(euclidean_list).values

#κανονικοποιημενη αποσταση Manhattan(braycurtis)
def get_braycurtis(array1,array2):
    return scipy.spatial.distance.braycurtis(array1,array2)


def get_braycurtis_list(tfidf_matrix,dataframe):
    from scipy.sparse import csr_matrix
    braycurtis_list = []
    tfidf_matrix = csr_matrix (tfidf_matrix)
    for x in range(0,len(dataframe)):
        braycurtis_list.append(get_braycurtis(tfidf_matrix[x,:].toarray(),tfidf_matrix[x+len(dataframe),:].toarray()))
    return braycurtis_list

braycurtis_list = get_braycurtis_list(tfidf_matrix,df2)
df2['braycurtis_dis'] = pd.Series(braycurtis_list).values

#SENTIMENT ANALYSIS με βαση λεξεις με θετικο νοημα και λεξεις με αρνητικο - υποδειξη απο kernel στο kaggle
p_url = 'http://ptrckprry.com/course/ssd/data/positive-words.txt'
n_url = 'http://ptrckprry.com/course/ssd/data/negative-words.txt'

positive_words = requests.get(p_url).content.decode('latin-1')
positive_words = nltk.word_tokenize(positive_words)
positive_words.remove('not')
negative_words = requests.get(n_url).content.decode('latin-1')
negative_words = nltk.word_tokenize(negative_words)
positive_words = positive_words[413:]
negative_words = negative_words[418:]

def num_pos(sent):
    num_pos = 0
    word_list = [word.lower() for word in nltk.word_tokenize(sent)]
    for index, word in enumerate(word_list):
        if word in positive_words:
            if word_list[index-1] not in ['not','no']:
                num_pos += 1
    return num_pos


def num_neg(sent):
    num_neg = 0
    word_list = [word.lower() for word in nltk.word_tokenize(sent)]
    for index, word in enumerate(word_list):
        if word in negative_words:
            if word_list[index-1] not in ['not','no']:
                num_neg += 1
    return num_neg


df2['q1_num_pos'] = df2['q1_without_stopwords'].apply(lambda x: num_pos(x))
df2['q2_num_pos'] = df2['q2_without_stopwords'].apply(lambda x: num_pos(x))
df2['q1_num_neg'] = df2['q1_without_stopwords'].apply(lambda x: num_neg(x))
df2['q2_num_neg'] = df2['q2_without_stopwords'].apply(lambda x: num_neg(x))
df2['num_pos_diff'] = (df2['q1_num_pos'] - df2['q2_num_pos']).abs()
df2['num_neg_diff'] = (df2['q1_num_neg'] - df2['q2_num_neg']).abs()
df2.to_csv('df2.csv')

rcParams['figure.figsize'] = 12, 8

#το αρχειο df2 δημιουργειται πιο πριν
data = pd.read_csv('df2.csv')
data.head()

#df = data.sample(frac = 0.01)
df = data.sample(frac = 1).reset_index(drop = True)

#απαλοιφη των nα γραμμωμ
df2 = df.dropna()
df2.isnull().sum()

#Κανονικοποιηση των μετρικων
features = ['char_length_ratio','num_syn_words','wmd','Lev_dist','q1_ave_token_length','diff_ave_token_length','diff_num_tokens','cos_similarity','euclidean_dis','braycurtis_dis','num_pos_diff','num_neg_diff']

for col in features:
    df2.loc[:,col] = df2.loc[:,col].astype(float)

for col in features:
    if col != 'wmd':
        df2.loc[:,col] = (df2.loc[:,col]-df2.loc[:,col].min())/(df2.loc[:,col].max()-df2.loc[:,col].min())
    else:
        df2.loc[df2['wmd'] == np.inf, 'wmd'] = 1.386948
        df2.loc[:,col] = (df2.loc[:,col]-df2.loc[:,col].min())/(1.386948 - df2.loc[:,col].min())

#αφου δεν εχουμε διαθεσιμο το test.csv θα χωρισουμε το train σε train και test
divide = math.floor(len(df2)*0.8)
target = 'is_duplicate'
df_train = df2[:int(divide)]
df_test = df2[int(divide):]

#εκπαιδευση με gradient boosted decision trees με μετρικη logloss(binary)
xgb1 = XGBClassifier(
learning_rate = 0.01, n_estimators = 5000, max_depth = 3,
min_child_weight = 1, gamma = 0, subsample=0.8,
colsample_bytree=0.7,
reg_alpha = 0.05,
objective = 'binary:logistic',
n_jobs=5,
scale_pos_weight=1,
random_state=27)

xgb1.fit(df_train[features],df_train[target], eval_metric = 'logloss')

#πιθανοτητες για is_duplicate στο test
predictions = xgb1.predict(df_test[features])
pred_probs = xgb1.predict_proba(df_test[features])
print(metrics.accuracy_score(df_test[target], predictions)) #ακριβεια
print(metrics.roc_auc_score(df_test[target], pred_probs[:,1])) #roc-auc
print(metrics.log_loss(df_test[target], pred_probs[:,1])) #logloss
