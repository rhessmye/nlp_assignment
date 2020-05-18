# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:01:03 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
tqdm.pandas()

# NLP
from nltk import sent_tokenize, word_tokenize, RegexpParser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import mark_negation
import nltk
from sklearn import metrics

import spacy
import collections
import re
import sys
import time
import itertools
from wordcloud import WordCloud, STOPWORDS 
from math import pi

import gensim
import nltk
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora.dictionary import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from collections import defaultdict
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV





nlp_data = pd.read_csv("D:/_RADS611 NLP/Assignment/honest_doc_full_post_nlp.csv")
nlp_data.head()

# Create a series to store the labels: y
y = nlp_data.sentiment
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(nlp_data['docc'], y, test_size=0.20, random_state=53)

X_train.shape
X_train.head()
X_test.shape
X_test.head()

y_train.shape
y_test.shape



#SVC
#the model
clf_svc = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC(verbose=2))])
parameters = {'tfidf__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__C': (10,1,0.1)}

# GridSearch to find the best parameters
gs_clf = GridSearchCV(clf_svc, parameters, cv=5, n_jobs=-1)

gs_clf.fit(X_train,y_train)

y_pred = gs_clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
for param_name in sorted(parameters.keys()):
  print("Best parameters: \n %s: %r" % (param_name, gs_clf.best_params_[param_name]))

#Model with the best parameters
clf_svc_best = Pipeline([
      ('tfidf', TfidfVectorizer(ngram_range=(1,1),use_idf=True)),
      ('clf', SVC(C=10, verbose=1))])
    
clf_svc_best.fit(X_train,y_train)

# Create the predicted tags: pred
y_pred_svc = clf_svc_best.predict(X_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, y_pred_svc)
print(score)

print(classification_report(y_test,y_pred_svc))

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, y_pred_svc)
print(cm)


## Logistic regression
clf_lgr = Pipeline([
      ('tfidf', TfidfVectorizer()),
      ('clf', LogisticRegression(verbose=1,n_jobs=-1))])
parameters = {'tfidf__ngram_range': [(1, 1), (1, 2), (1,3)],
              'tfidf__use_idf': [True, False],
              'clf__C': [10,1,0.1,0.001],
              'clf__penalty':['l2','l1']}
gs_clf = GridSearchCV(clf_lgr, parameters, cv=5, n_jobs=-1)

gs_clf.fit(X_train,y_train)

y_pred = gs_clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
for param_name in sorted(parameters.keys()):
  print("Best parameters: \n %s: %r" % (param_name, gs_clf.best_params_[param_name]))

#Retrain with all values and best parameters
clf_lgr_best = Pipeline([
      ('tfidf', TfidfVectorizer(ngram_range=(1,2),use_idf=False)),
      ('clf', LogisticRegression(C=10 , penalty='l2' ,verbose=1,n_jobs=-1))])

clf_lgr_best.fit(X_train,y_train)

# Create the predicted tags: pred
y_pred_lgr = clf_lgr_best.predict(X_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, y_pred_lgr)
print(score)

print(classification_report(y_test,y_pred_lgr))

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, y_pred_lgr)
print(cm)


##
clf_mnb = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('mnb', MultinomialNB())])
    
parameters = {
  'mnb__alpha': np.linspace(0.5, 1.5, 6),
  'mnb__fit_prior': [True, False],  }

gs_clf_mnb = GridSearchCV(clf_mnb, parameters, cv=5, n_jobs=-1)

gs_clf_mnb.fit(X_train,y_train)

y_pred = gs_clf_mnb.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

for param_name in sorted(parameters.keys()):
  print("Best parameters: \n %s: %r" % (param_name, gs_clf_mnb.best_params_[param_name]))

#Retrain with all values and best parameters
clf_mnb_best = Pipeline([
      ('tfidf', TfidfVectorizer(ngram_range=(1,2),use_idf=False)),
      ('mnb', MultinomialNB( alpha=0.5, fit_prior=False, class_prior=None))])

clf_mnb_best.fit(X_train,y_train)

# Create the predicted tags: pred
y_pred_mnb = clf_mnb_best.predict(X_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, y_pred_mnb)
print(score)

print(classification_report(y_test,y_pred_mnb))

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, y_pred_mnb)
print(cm)




