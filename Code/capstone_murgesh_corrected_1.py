# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:35:49 2018

@author: dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import quandl
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
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import glob
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(sentence):
	sentence = sentence.lower()
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	return tokens

#Read the text files to a list using a loop

path = r'data\\text_topics\\'
filenames = glob.glob(path + '/*.txt')


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

#Create an empty list and clean the email text list

email_text = []

for i in range(0, len(email_msg)):

        sent = ''.join(email_msg[i])

        email_text.append(sent)



email_text_processed = []
for email in email_text:
    email_text_processed.append(preprocess(email))

print(email_text_processed[0])    
type(email_text_processed[0])

#  getting all the words from the list to a single list

bag_of_words=[]

for i in range(len(email_text_processed)):
	for j in range(len(email_text_processed[i])):
		if email_text_processed[i][j] in bag_of_words:
			pass
		else:
			bag_of_words.append(email_text_processed[i][j])
			

Invoking TFIDF
   
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,4), min_df = 0, stop_words = 'english')
vectorizer = CountVectorizer()
tf_matrix = vectorizer.fit_transform(bag_of_words)

print(tf_matrix)
type(tf_matrix)


vect = CountVectorizer()
vect.fit(email_text)

       

### FEATURE EXTRACTION
feature_algo = TfidfVectorizer()
text_features = feature_algo.fit_transform(bag_of_words)
print(text_features)
text_features_raw_matrix = text_features.toarray()
feature_algo.vocabulary_
             


email_df = pd.DataFrame(text_features.toarray(),columns = vectorizer.get_feature_names())

email_df.head(2)