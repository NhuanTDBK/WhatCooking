
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
from keras.callbacks import EarlyStopping

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


# In[4]:

total_dat['ingre_str'] = total_dat.ingredients.map(normalize_ingres) 


# In[5]:

lemma = lambda x: x.strip().lower().split(',')
ingredient_lemmatized = total_dat.ingre_str.map(lemma)


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

to_idx = lambda x: [word2idx[word] for word in x]

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

recipe_to_array = np.array(ingredient_lemmatized.map(to_k_hot).tolist(),dtype='int32')


# In[49]:

onehotEncoder = OneHotEncoder()
labels = onehotEncoder.fit_transform(encoder.fit_transform(train_dat.cuisine).reshape(-1,1))

train_len = train_dat.shape[0]
word_embedding = 400
output_size = encoder.classes_.shape[0]


# In[53]:

X_train, X_val, y_train, y_val = train_test_split(recipe_to_array[:train_len],labels.toarray(), test_size = 0.2)


# In[68]:

# input_layer = Input(shape=(X_train))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=word_embedding,input_length=vocab_size)


# In[69]:

from keras.models import Sequential


# In[84]:
model = Sequential([
#        embedding_layer,
 #       Conv1D(128, 5, activation='relu'),
  #      MaxPooling1D(5),
   #     Flatten(),
	Dense(2048,input_dim=X_train.shape[1],activation='relu'),
	Dropout(0.5),
	Dense(1024,activation='relu'),
	Dropout(0.25),
	Dense(256,activation='relu'),
        Dense(output_size, activation='softmax')
])
from keras.optimizers import SGD
#sgd = SGD(lr=0.1,decay=1e-5,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer="adagrad",metrics=['acc'])
from keras.callbacks import EarlyStopping
earlyStopper = EarlyStopping(monitor='val_loss',patience=10)
# In[74]:
model.summary()
# In[78]:
model.fit(X_train,y_train,validation_split=0.1, nb_epoch=256, batch_size=32, verbose=2, callbacks=[earlyStopper])
model.save("RecipeEmbedding3")
print model.evaluate(X_val,y_val,batch_size=32)
