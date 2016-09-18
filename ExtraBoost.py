
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import numpy as np
import pickle
from utils import *


# In[2]:

n_iter = 200
k_fold = 10
# cv = kfold
# initialize the classifier
X_train, X_val, y_train, y_val, cv = load_train_and_kfold(n_folds=k_fold)


# In[25]:

model = ExtraTreesClassifier(random_state=4111)
model_name = model.__class__.__name__
param_grid = {
      "max_depth": sp_randint(4, 100),
      "max_features": sp_randint(1, 11),
      "min_samples_split": sp_randint(1, 11),
      "min_samples_leaf": sp_randint(1, 11),
      "bootstrap": [True, False],
      "criterion": ["gini", "entropy"],
#       "n_estimators": sp_randint(100,600)
}


# In[6]:

search_GB = RandomizedSearchCV(model,param_grid,scoring='log_loss',n_jobs=-1,
               n_iter=2,cv=cv,verbose=True)
search_GB.fit(X_train,y_train.flatten())


# In[7]:

log_model = search_GB.score(X_val,y_val.flatten())
print "Log loss = %s"%log_model
X_test = get_test()
y_pred = search_GB.predict_proba(X_test)
save_submission(model_name,log_model,y_pred)

