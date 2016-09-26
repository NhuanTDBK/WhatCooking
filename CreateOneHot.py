# coding: utf-8

# In[38]:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from nltk.stem import *
from nltk import wordpunct_tokenize, word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import *
import string
from keras.layers import *
import itertools
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import csr_matrix


# In[2]:

train_dat = pd.read_json("data/train.json")
test_dat = pd.read_json("data/test.json")

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("english")
    def __call__(self, doc):
        return [self.convert_origin(t) for t in word_tokenize(doc)
                if t not in string.punctuation]
    def transform(self,doc):
        return self.__call__(doc)
    def convert_origin(self,word):
        word = word.replace(".","").replace('/','').replace('-','').replace("'s","").replace("'n","")
        return self.wnl.lemmatize(self.stemmer.stem(word).encode('ascii','ignore').lower())
# Smooth idf, sublinear df, norm
stemmer = SnowballStemmer("english")
lemma_normalizer = LemmaTokenizer()
encoder = LabelEncoder()

print "Extract words..."
total_dat = train_dat.drop(['cuisine'],axis=1).append(test_dat)


# In[3]:

# normalize_ingres = lambda ingres: str([lemma_normalizer.transform(ingre) for ingre in ingres])
def normalize_ingres(ingres):
    result = [lemma_normalizer.transform(ingre) for ingre in ingres]
    return ','.join(list(itertools.chain(*result)))

#total_dat['ingre_str'] = total_dat.ingredients.map(normalize_ingres) 
lemma = lambda x: x.strip().split(',')
ingredient_lemmatized = total_dat.ingredients.map(normalize_ingres).map(lemma)

# In[6]:
print "Build vocabulary"
vocab = []
result = [vocab.extend(recipe) for recipe in ingredient_lemmatized]
vocab = set(vocab)

# In[7]:

word2idx = dict((v, i) for i, v in enumerate(vocab))


# In[13]:

# idx2word = list(words)
vocab_size = len(vocab)
print vocab_size


# In[14]:

recipe_to_array = []
#convert word to integer
to_idx = lambda x: [word2idx[word] for word in x]


# In[16]:

#recipe_to_array = np.zeros((vocab_size))


# In[17]:

# vocbulary of word to one-hot vector


# In[18]:

word2hot = {}
for k,v in word2idx.iteritems():
    recipe_to_array = np.zeros((vocab_size))
    recipe_to_array[v] = 1
    word2hot[k] = recipe_to_array

# convert to k-hot vector
def to_k_hot(recipes):
    result = np.zeros((vocab_size))
    for recipe in recipes:
        result = result + word2hot[recipe]
#         print recipe
    return result


# In[34]:
train_len = train_dat.shape[0]
recipe_to_array = np.array(ingredient_lemmatized.map(to_k_hot).tolist(),dtype='int32')

# In[49]:

onehotEncoder = OneHotEncoder()
labels = onehotEncoder.fit_transform(encoder.fit_transform(train_dat.cuisine).reshape(-1,1))



