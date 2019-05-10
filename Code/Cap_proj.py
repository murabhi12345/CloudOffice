# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 18:57:10 2017

@author: dell
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quandl
import math
import os
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.cluster import KMeans
from sklearn import metrics, datasets
from scipy.spatial.distance import cdist,pdist
from sklearn.preprocessing import scale
from sklearn import cross_validation, svm
from scipy.spatial.distance import cdist,pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from bs4 import BeautifulSoup
from sklearn.naive_bayes import  MultinomialNB
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import  MultinomialNB
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk import ne_chunk,pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import glob



#Read the text files to a list using a loop

path = r'D:\\Python\\Data Science\\Assignments\\Project\\text_topics\\'
filenames = glob.glob(path + '/*.txt')

#define an empty list
email_msg = []

#Open the filename, read each record in the specified folder and add to the list

for f in filenames:
    
    count = f 
    rd = open(f,encoding='utf8')
    df = rd.readlines()
    email_msg.append(df)
    rd.close()
#    print(count)

print(len(email_msg))
print(len(filenames))

#Create an empty list and clean the email text list

email_text = []

for i in range(0, len(email_msg)):
        sent = ''.join(email_msg[i])
        email_w = word_tokenize(sent)
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in email_w]
        words = [word for word in stemmed if word.isalpha()]
        email_text_lower = [element.lower() for element in words]    

        email_text_list.append(email_text_lower)


        
### FEATURE EXTRACTION
feature_algo = TfidfVectorizer()
text_features = feature_algo.fit_transform(email_text)
print(text_features)
text_features_raw_matrix = text_features.toarray()
feature_algo.vocabulary_

#
#
## Removing stop words
#feature_algo_wo_stopwords = TfidfVectorizer(stop_words='english')
#text_features_wo_stopwords = feature_algo_wo_stopwords.fit_transform(tweets)
#text_features__wo_stopwords_raw_matrix = text_features_wo_stopwords.toarray()
#feature_algo_wo_stopwords.vocabulary_
#
## Getting term frequency matrix without idf weighting
#feature_algo_tf = TfidfVectorizer(use_idf=False,norm=None)
#text_features_tf = feature_algo_tf.fit_transform(tweets)
#text_features_tf_raw_matrix = text_features_tf.toarray()
#feature_algo_tf.vocabulary_

