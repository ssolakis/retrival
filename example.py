import numpy as np
import pandas as pd
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
from lsh import cache, minhash

####### μετατροπη του csv σε txt και εγγραφη τιμων #########
#df_train = pd.read_csv('C:/Users/Owner/Desktop/InformationRetrieval/input/train_original.csv')
#df_train = df_train.sample(1000)
#train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
#txt_file = open('C:/Users/Owner/Desktop/csv-txt/questions.txt','w', encoding='utf8')
#for i in range(0,1000):
    #txt_file.write(('{}\t{}\n'.format(i,train_qs[i])))

def shingles(text, char_ngram=5):
    return set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram))


def candidate_duplicates(document_feed, char_ngram=5, seeds=100, bands=5, hashbytes=4):
    char_ngram = 5
    sims = []
    hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, hashbytes=hashbytes)
    if seeds % bands != 0:
        raise ValueError('Seeds has to be a multiple of bands. {} % {} != 0'.format(seeds, bands))

    lshcache = cache.Cache(num_bands=bands, hasher=hasher)
    for i_line, line in enumerate(document_feed):
        line = line.decode('utf8')
        docid, headline_text = line.split('\t', 1)
        fingerprint = hasher.fingerprint(headline_text.encode('utf8'))

        # in addition to storing the fingerpring store the line
        # number and document ID to help analysis later on
        lshcache.add_fingerprint(fingerprint, doc_id=(i_line, docid))

    candidate_pairs = set()
    for b in lshcache.bins:
        for bucket_id in b:
            if len(b[bucket_id]) > 1:
                pairs_ = set(itertools.combinations(b[bucket_id], r=2))
                candidate_pairs.update(pairs_)

    return candidate_pairs


#### ομοιοτητα jaccard#######
def jaccard(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)

##### τεστ για 1000 ζευγαρια #######
hasher = minhash.MinHasher(seeds=50, char_ngram=5, hashbytes=4)
lshcache = cache.Cache(bands=2, hasher=hasher)

with open('C:/Users/Owner/Desktop/csv-txt/questions.txt', 'rb') as fh:
    feed = itertools.islice(fh, 1000)
    for line in feed:
        question_id, question = line.decode('utf8').split('\t', 1)
        lshcache.add_fingerprint(hasher.fingerprint(line), question_id)

candidate_pairs = set()
for b in lshcache.bins:
    for bucket_id in b:
        if len(b[bucket_id]) > 1:
            pairs_ = set(itertools.combinations(b[bucket_id], r=2))
            candidate_pairs.update(pairs_)

print(candidate_pairs)

####### τεστ για τον αριθμο των bands#######
num_candidates = []
bands = [2, 5, 10, 20]
for num_bands in bands:
    with open('C:/Users/Owner/Desktop/csv-txt/questions.txt', 'rb') as fh:
        feed = itertools.islice(fh, 1000)
        candidates = candidate_duplicates(feed, char_ngram=5, seeds=100, bands=num_bands, hashbytes=4)
        num_candidates.append(len(candidates))

fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(bands, num_candidates, align='center');
plt.title('Συγκριση Ζωνων(Bands)');
plt.xlabel('Αριθμος Ζωνων');
plt.ylabel('Αριθμος Υποψηφιων Διπλοτυπων');
plt.xticks(bands, bands);
plt.show();

##### υπολογισμος ανακλησης των true positives
lines = []
with open('C:/Users/Owner/Desktop/csv-txt/questions.txt', 'rb') as fh:
    for line in itertools.islice(fh, 1000):
        lines.append(line.decode('utf8'))

    fh.seek(0)
    feed = itertools.islice(fh, 1000)
    candidates = candidate_duplicates(feed, char_ngram=5, seeds=100, bands=20, hashbytes=4)

similarities = []
for ((line_a, qid_a), (line_b, qid_b)) in candidates:
    q_a, q_b = lines[line_a], lines[line_b]
    shingles_a = shingles(lines[line_a])
    shingles_b = shingles(lines[line_b])

    jaccard_sim = jaccard(shingles_a, shingles_b)
    fingerprint_a = set(hasher.fingerprint(q_a.encode('utf8')))
    fingerprint_b = set(hasher.fingerprint(q_b.encode('utf8')))
    minhash_sim = len(fingerprint_a & fingerprint_b) / len(fingerprint_a | fingerprint_b)
    similarities.append((qid_a, qid_b, jaccard_sim, minhash_sim))

import random

print('Υπαρχουν {} υποψηφια διπλοτυπα'.format(len(candidates)))
random.sample(similarities, k=15)

sims_all = np.zeros((1000, 1000), dtype=np.float64)
for i, line in enumerate(lines):
    for j in range(i + 1, len(lines)):
        shingles_a = shingles(lines[i])
        shingles_b = shingles(lines[j])
        jaccard_sim = jaccard(shingles_a, shingles_b)
        sims_all[i, j] = jaccard_sim

candidates_dict = {(line_a, line_b): (qid_a, qid_b) for ((line_a, qid_a), (line_b, qid_b)) in candidates}
found = 0
for i in range(len(lines)):
    for j in range(i+1, len(lines)):
        if sims_all[i, j] >= .8:
            found += ((i, j) in candidates_dict or (j, i) in candidates_dict)

print('Απο τα {} ζευγαρια με Jaccard ομοιοτητα >= 80% , βρεθηκαν τα  {}. Αντιστοιχει σε ποσοστο: {:.1%}'.format((sims_all >= .8).sum(), found, found / (sims_all >= .8).sum()))