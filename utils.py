import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from nltk.stem import *

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = SnowballStemmer("english")
    def __call__(self, doc):
        return [self.wnl.stem(t) for t in doc.split(" ")]
train_dat = pd.read_json("data/train.json")
test_dat = pd.read_json("data/test.json")
total_dat = train_dat.drop(['cuisine'],axis=1).append(test_dat)
stemmer = SnowballStemmer("english")
tfIdfTrans = TfidfVectorizer(sublinear_tf=True, tokenizer=LemmaTokenizer())
total_dat["ingredient_str"] = total_dat.ingredients.map(lambda d: ' '.join(d))
total_dat_ing = total_dat.drop(['ingredients'],axis=1)
tfIdfTrans.fit(total_dat_ing.ingredient_str)
# Smooth idf, sublinear df, norm
def load_onehot_train_and_kfold(n_folds=5):
	train_dat = pd.read_csv('data/train.csv')
	labelEncoder = LabelEncoder()
	onehotEncoder = OneHotEncoder()
	new_dat = np.log(train_dat.drop(['id','target'],axis=1)+1)
	categories = labelEncoder.fit_transform(train_dat.target).reshape(-1,1)
	onehot_categorical = onehotEncoder.fit_transform(categories.reshape(-1,1)).toarray()
	X_train,X_test,y_train, y_test = train_test_split(new_dat,onehot_categorical,test_size=0.3,stratify=categories)
	kfold = StratifiedKFold(labelEncoder.fit_transform(train_dat.target[:X_train.shape[0]]), n_folds=n_folds, shuffle=True)
	return X_train,X_test,y_train, y_test, kfold
def load_train_and_kfold(n_folds=10):	
	vectorIngr = tfIdfTrans.transform(total_dat_ing.ingredient_str[:train_dat.shape[0]])
	labels = train_dat.cuisine
	kfold = StratifiedKFold(labels, n_folds=n_folds)
	X_train, X_test, y_train, y_test = train_test_split(vectorIngr, labels, train_size=0.8)
	return X_train,X_test,y_train, y_test, kfold 

def save_submission(model_name,loss_model,y_test):
	test = pd.read_csv('data/test.csv')
	new_test = test.drop(['id'],axis=1)
	#y_test = model.predict_proba(new_test.values)
	submisstion = pd.read_csv('data/sampleSubmission.csv')
	nn_sub = pd.DataFrame(columns=submisstion.columns, index=test.index)
	nn_sub.id = test.id
	nn_sub[nn_sub.columns[1:]] = y_test
	log_loss = loss_model
	nn_sub.to_csv("results/%s_%s.csv"%(model_name,log_loss),index=None)
	print "Save submission completed"
#>>>>>>> eb57fc4f01fd81cd0a07ff0a92a8d45b3c83ae4f
def load_train(file_name="train_data"):
	dat_file = np.load(file_name)
	return dat_file["X_train"],dat_file["X_test"],dat_file["y_train"],dat_file["y_test"]
def get_test(normalize=True):
	vectorIngr = tfIdfTrans.transform(total_dat_ing.ingredient_str[train_dat.shape[0]:])
	return vectorIngr
