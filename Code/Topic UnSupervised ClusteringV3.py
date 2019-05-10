import pandas as pd
import numpy as np
import os
import nltk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import io
import re
from sklearn.model_selection import train_test_split
from sklearn import cluster
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import wordnet
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
import pickle
lemmatizer = WordNetLemmatizer()
actual = pd.DataFrame
from sklearn.neighbors import KNeighborsClassifier
#define an empty list
email_msg = []
data=[]
file_count =0
stop_words = set(stopwords.words('english'))

#Open the filename, read each record in the specified folder and add to the list
path = r"D:\Users\51678368\Documents\text_topics\text_topics\*.txt"
for name in glob.glob(path):
    with io.open(name, "r",encoding='ISO_8859-1:1987') as file:
        original_file = file.readlines()

    content = ''
    for line in original_file:
        line2 = re.sub('[^A-Za-z0-9]+', ' ',line)
        line1= []
        line1 = word_tokenize(line2)
        filtered_sentence = [w for w in line1 if not w in stop_words]
        l = ''
        for w in filtered_sentence:
            if len(w)>3 and not w.isdigit() and w.isalpha() :
                w_lem = lemmatizer.lemmatize(w)
                if w == w_lem:
                    w_lem = lemmatizer.lemmatize(w,pos='v')
                l +=  w_lem+ ' '
        content += l.lower()
    content = re.sub('[^A-Za-z0-9\-]+', ' ',content)
    data.append([file_count,name, content])
    file_count +=1
####################Email text put in dataframe, file number, name and its content after preprocessing
df = pd.DataFrame(data, columns=['FileNumber','Name', 'Content'])

print('-----------------Exmail text Representation in DATA Frame-----------------------')
print(df)


########### Ground Truth- Target file ################
actual = pd.read_csv(r"D:\Users\51678368\Documents\text_topics\target.csv")
df['label'] =actual.label

########## Transforming Files to Vector where columns represent terms (words in document) and rows represent tf*idf value( 1- row 1 document)
vectorizer = TfidfVectorizer(min_df=0.01,max_df = 0.7, sublinear_tf=True, use_idf =True, stop_words = 'english')
tf_matrix = vectorizer.fit_transform(df.Content)

### FEATURE EXTRACTION
features_list =  vectorizer.get_feature_names()

#Top most Features based on idf
indices = np.argsort(vectorizer.idf_)[::-1]
features = vectorizer.get_feature_names()
top_n = 100
top_features = [features[i] for i in indices[:top_n]]

print('----------------TOP 100 Features------------------------')

print(top_features)

############### Spliting Training and Testing Data ###############

X_train, X_test, y_train, y_test = train_test_split(tf_matrix, df.label, test_size=0.20)# 20% testing

print('-----------------Shape to Train and Test Data-----------------------')
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
print('--------------------------------------------------------------------')

##############Naive Bayes Classifier Training--80% used for Training

nb = GaussianNB()
nb_clf = nb.fit(X_train.toarray(),y_train)

##############Naive Bayes Classifier Testing--20% for testing

nb_pred = nb.predict(X_test.toarray())

##############Naive Bayes Classifier Accuracy Calculation
print('Testing Data Accuracy',accuracy_score(y_test,nb_pred))


