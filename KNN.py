
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import numpy as np
import pickle
from utils import *


# In[2]:

n_iter = 30
k_fold = 3
# cv = kfold
# initialize the classifier
X_train, X_val, y_train, y_val, cv = load_train_and_kfold(n_folds=k_fold)


# In[4]:

model = KNeighborsClassifier()
model_name = model.__class__.__name__
param_grid = {
      "n_neighbors": sp_randint(4,400),
      "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"],
}


# In[ ]:

search_GB = RandomizedSearchCV(model,param_grid,scoring=scoring,n_jobs=-1,
               n_iter=n_iter,cv=cv,verbose=True)
search_GB.fit(X_train,y_train.flatten())

log_model = search_GB.score(X_val,y_val.flatten())
print "Log loss = %s"%log_model
X_test = get_test()
y_pred = search_GB.predict(X_test)
save_submission(model_name,log_model,y_pred)


# In[ ]:



