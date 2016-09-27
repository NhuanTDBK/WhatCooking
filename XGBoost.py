
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import numpy as np
import pickle
from utils import *


# In[2]:

n_iter = 10
k_fold = 3
# cv = kfold
# initialize the classifier


# In[3]:
X_train, X_val, y_train, y_val, cv = load_train_and_kfold(n_folds=k_fold)

# In[4]:

GB = xgb.XGBClassifier(silent=False)
param_grid = {
              'max_depth': sp_randint(10, 200),
              'learning_rate': sp_uniform(loc=0e0,scale=1e0),
              'objective':['multi:softmax'],
              'nthread': [16],
              'missing': [np.nan],
              'reg_alpha': [0.01,0.056234133,\
                            0.1,0.56234133,1.,\
                            3.16227766,10.,\
                            17.7827941,100.],
              'colsample_bytree': sp_uniform(loc=0.2e0,scale=0.8e0),
              'subsample': sp_uniform(loc=0.6,scale=0.3),
              'n_estimators': sp_randint(100,1000),
}

print "Randomized XGBoost"
# In[ ]:
search_GB = RandomizedSearchCV(GB,param_grid,scoring=scoring,n_jobs=-1,
               n_iter=n_iter,cv=cv,verbose=True)
search_GB.fit(X_train,y_train)
log_model = search_GB.score(X_val,y_val)
print "Log loss = %s"%log_model
X_test = get_test()
save_submission('XGBoost',log_model,search_GB.predict(X_test))
