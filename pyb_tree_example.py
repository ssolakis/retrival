import pybktree
import pandas as pd
import re
from simhash import Simhash

def get_features(s):
    width = 3
    s = s.lower()
    s = re.sub(r'[^\w]+', '', s)
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]


#tree = pybktree.BKTree(pybktree.hamming_distance, [7403924248468883405])
#df_train = pd.read_csv('C:/Users/Owner/Desktop/InformationRetrieval/input/train_original.csv',nrows=10)
#df_train = df_train.sample(10)
#train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
#for i in range(0,10):
    #tree.add(hash(train_qs[i]))
    #print(train_qs[i])
#x = hash("How can I be a good geologist?")
#print(sorted(tree.find(x, 3)))

import collections
#y = hash("This is a dummy sentence to be hashed!")
y = Simhash(get_features('This is a dummy sentence to be hashed!')).value
Item = collections.namedtuple('Item', 'bits id')

def item_distance(x, y):
    return pybktree.hamming_distance(x.bits, y.bits)


tree = pybktree.BKTree(item_distance, [Item(y, 'This is a dummy sentence to be hashed!')])

df_train = pd.read_csv('C:/Users/Owner/Desktop/InformationRetrieval/input/train_original.csv',nrows=100000)
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
for i in range(0,200000):
    #tree.add(Item(hash(train_qs[i]),train_qs[i]))
    tree.add(Item(Simhash(get_features(train_qs[i])).value, train_qs[i]))
z = "Can we hack Facebook?"
#x = hash(z)
x= Simhash(get_features(z)).value
print(sorted(tree.find(Item(x, z), 12)))
