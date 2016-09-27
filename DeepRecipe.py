
# coding: utf-8

# In[24]:

import pandas as pd
import numpy as np
from utils import *
from keras.layers import Dense, Activation, Dropout, Embedding, Lambda, Flatten, Reshape
from keras.models import Sequential
from scipy.sparse import csr_matrix
from data_hub import *
from keras import backend as K
# from keras.utils.visualize_util import model_to_dot, plot


# In[2]:

vector_sum = lambda v: K.mean(v,axis=1)


# In[3]:

from CreateOneHot import *


# In[4]:

word_embedding = 300

# In[25]:

model = Sequential()
word_dims = word_embedding
model.add(Embedding(vocab_size, word_dims, input_length=vocab_size,init="glorot_uniform"))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.
model.add(Lambda(vector_sum, output_shape=(word_dims,)))
model.add(Dense(128,activation='relu',init="glorot_uniform"))
model.add(Dropout(0.5))
model.add(Dense(labels.shape[1],activation='softmax'))
# input_array = np.random.randint(400, size=(32, 10))
# model.compile('rmsprop', 'mse')

# In[27]
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['acc'])
# In[28]:

model.summary()


# In[29]:

from keras.callbacks import EarlyStopping,TensorBoard
early = EarlyStopping(monitor='val_loss',patience=20)
board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)


# In[31]:

history = model.fit(recipe_to_array[:train_len].toarray(),labels.toarray(),validation_split=0.1,nb_epoch=3,
                    batch_size=32,callbacks=[early,board],verbose=1)


# In[11]:

embedding = model.get_weights()[0]


# In[12]:

recipes_embed = {}
for embed_recipe,word in zip(embedding,vocab):
    recipes_embed[word] = embed_recipe


# In[35]:

recipes2vec = np.zeros((total_dat.ingre_str.shape[0],word_embedding))


# In[36]:

for idx,recipe in enumerate(total_dat.ingre_str):
    for word in recipe.split(','):
        recipes2vec[idx] = recipes2vec[idx] + recipes_embed[word]


# In[46]:

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
estimator = LogisticRegression(C=10)


# In[47]:

results = cross_val_score(estimator,X=recipes2vec[:train_len],y=train_dat.cuisine,cv=3)


# In[48]:

print results.mean()


# In[ ]:




# In[ ]:



