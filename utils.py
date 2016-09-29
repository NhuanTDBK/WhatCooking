import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from nltk.stem import *

scoring = "accuracy"
# Smooth idf, sublinear df, norm
def load_onehot_train_and_kfold(n_folds=5):
	X_train, X_test, y_train, y_test = load_train()
	train_len = len(y_train)
	kfold = StratifiedKFold(y_train, n_folds=n_folds, shuffle=True)
	labelEncoder = LabelEncoder()
	onehotEncoder = OneHotEncoder()
	total_target = np.append(y_train,y_test)
	categories = labelEncoder.fit_transform(total_target).reshape(-1,1)
	onehot_categorical = onehotEncoder.fit_transform(categories.reshape(-1,1)).toarray()
	y_train = onehot_categorical[:train_len]
	y_test = onehot_categorical[train_len:]
	return X_train,X_test,y_train, y_test, kfold
def load_train_and_kfold(n_folds=5):	
	X_train, X_test, y_train, y_test = load_train()
	cv = StratifiedKFold(y_train, n_folds=n_folds, shuffle=True)
	return X_train,X_test,y_train, y_test, cv

def save_submission(model_name,log_loss,y_test):
	# def save_submission(model_name,loss_model,y_test):
	test = pd.read_json('data/test.json')
	new_test = test.drop(['id'],axis=1)
	#y_test_sub = get_test()
	submisstion = pd.read_csv('data/sample_submission.csv')
	nn_sub = pd.DataFrame(columns=submisstion.columns, index=test.index)
	nn_sub.id = test.id
	nn_sub[nn_sub.columns[1:]] = y_test
	nn_sub.to_csv("results/%s_%s.csv"%(model_name,log_loss),index=None)
	print "Save submission completed"
def load_train(file_name="train_data.npz",todense=False):
	dat_file = np.load(file_name)
	if todense is False:
		return dat_file["X_train"].tolist().asfptype(),dat_file["X_val"].tolist().asfptype(),dat_file["y_train"].tolist(),dat_file["y_val"].tolist()
	else:
		return dat_file["X_train"].tolist().asfptype().toarray(),dat_file["X_val"].tolist().asfptype().toarray(),dat_file["y_train"].tolist(),dat_file["y_val"].tolist()
		
def get_test(normalize=True):
	dat_file = np.load("train_data.npz")
	vectorIngr = dat_file["X_test"]
	return vectorIngr.tolist()
