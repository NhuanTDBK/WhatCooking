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
import itertools

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
        word = word.replace(".","").replace('/','').replace('-','').replace("'s","").replace("'n")
        return self.wnl.lemmatize(self.stemmer.stem(word).encode('ascii','ignore').lower())
def TfIdfComponent():
	# Smooth idf, sublinear df, norm
	stemmer = SnowballStemmer("english")
	lemma_normalizer = LemmaTokenizer()

	train_dat = pd.read_json("data/train.json")
	test_dat = pd.read_json("data/test.json")
	total_dat = train_dat.drop(['cuisine'],axis=1).append(test_dat)
	# Transform recipe into origin word
	def normalize_ingres(ingres):
	    result = [lemma_normalizer.transform(ingre) for ingre in ingres]
	    return ' '.join(list(itertools.chain(*result)))
	total_dat['ingre_str'] = total_dat.ingredients.map(normalize_ingres)
	# Build vocabulary for TF-IDF
	chain = list(itertools.chain(*set_ingredients))
	total_items = np.array(list(chain))
	total_items = np.unique(total_items)
	vocabulary = []
	for item in total_items:
	    for word in lemma_normalizer.transform(item):
		vocabulary.append(word)
	vocab = np.unique(vocabulary)
	del vocabulary, set_ingredients

	total_dat["ingredient_str"] = total_dat.ingredients.map(lambda d: ' '.join(d))
	total_dat_ing = total_dat.drop(['ingredients'],axis=1)
	# Build TF-IDF Transformer
	tfIdfTrans = TfidfVectorizer(tokenizer=LemmaTokenizer(), vocabulary=vocab)
	tfIdfTrans.fit(total_dat_ing.ingredient_str)
	vectorIngr = tfIdfTrans.transform(total_dat_ing.ingredient_str[:train_dat.shape[0]])
	labels = train_dat.cuisine
	cv = StratifiedKFold(labels, n_folds=3)
	X_train, X_test, y_train, y_test = train_test_split(vectorIngr, labels, train_size=0.8)
	return X_train, y_train, X_test, y_test

