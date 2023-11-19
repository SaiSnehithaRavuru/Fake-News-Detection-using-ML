#---------------------------------------------------------IMPORTING THE LIBRARIES------------------------------------------------------------#
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import itertools
import numpy as np
import time

#---------------------------EXTRACTING THE DATASET----------------------------#
kbr = pd.read_csv("fake_or_real_news.csv")
z = kbr.label
kbr.drop("label", axis=1)

#--------------------DIVING TRAINING AND TEST SETS----------------------------#
Ktrain, Ktest, Btrain, Btest = train_test_split(kbr['title'].values.astype('U'), z, test_size=0.20, random_state=0)

#---------------------PERFORMING COUNT VECTORIZER OPERATIONS-----------------#
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(Ktrain)
count_test = count_vectorizer.transform(Ktest)

#-----------------------PERFORMING TFIDF VECTORIZER OPERATIONS----------------#
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(Ktrain)
tfidf_test = tfidf_vectorizer.transform(Ktest)

#------------------------------------HASH VECTORIZER----------------------#
hash_vectorizer = HashingVectorizer(stop_words='english')
hash_train = hash_vectorizer.fit_transform(Ktrain)
hash_test = hash_vectorizer.transform(Ktest)

#-----------------------PROGRAM FOR PLOTTING OF CONFUSION MATRIX--------#
def plot_confusion_matrix(cm, classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#-------------------------------------MULTINOMIAL BAYES USING TFIDF-------------------#
Alg = MultinomialNB()
Alg.fit(tfidf_train, Btrain)
pred = Alg.predict(tfidf_test)
score1 = metrics.accuracy_score(Btest, pred)
pre1 = metrics.precision_score(Btest, pred, pos_label='FAKE')
rec1 = metrics.recall_score(Btest, pred, pos_label='FAKE')
f11 = metrics.f1_score(Btest, pred, pos_label='FAKE')
print("accuracy: %0.3f" % score1)
cm = metrics.confusion_matrix(Btest, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

#--------------------------------MULTINOMIAL BAYES USING COUNT------------------#
Alg = MultinomialNB()
Alg.fit(count_train, Btrain)
pred = Alg.predict(count_test)
score2 = metrics.accuracy_score(Btest, pred)
pre2 = metrics.precision_score(Btest, pred, pos_label='FAKE')
rec2 = metrics.recall_score(Btest, pred, pos_label='FAKE')
f12 = metrics.f1_score(Btest, pred, pos_label='FAKE')
print("accuracy: %0.3f" % score2)
cm = metrics.confusion_matrix(Btest, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

#-----------------------PASSIVE AGGRESSIVE CLASSIFIER USING TFIDF----------------#
Alg1 = PassiveAggressiveClassifier()
Alg1.fit(tfidf_train, Btrain)
pred = Alg1.predict(tfidf_test)
score3 = metrics.accuracy_score(Btest, pred)
pre3 = metrics.precision_score(Btest, pred, pos_label='FAKE')
rec3 = metrics.recall_score(Btest, pred, pos_label='FAKE')
f3 = metrics.f1_score(Btest, pred, pos_label='FAKE')
print("accuracy: %0.3f" % score3)
cm = metrics.confusion_matrix(Btest, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

#-------------PASSIVE AGGRESSIVE CLASSIFIER USING HASH VECTORIZER-----#
Alg1 = PassiveAggressiveClassifier()
Alg1.fit(hash_train, Btrain)
pred = Alg1.predict(hash_test)
score4 = metrics.accuracy_score(Btest, pred)
pre4 = metrics.precision_score(Btest, pred, pos_label='FAKE')
rec4 = metrics.recall_score(Btest, pred, pos_label='FAKE')
f14 = metrics.f1_score(Btest, pred, pos_label='FAKE')
print("accuracy: %0.3f" % score4)
cm = metrics.confusion_matrix(Btest, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

#-------------------XGBOOST CLASSIFIER USING THE TFIDF VECTORIZER------------#
Alg2 = XGBClassifier()
Alg2.fit(tfidf_train, Btrain)
pred = Alg2.predict(tfidf_test)
score5 = metrics.accuracy_score(Btest, pred)
pre5 = metrics.precision_score(Btest, pred, pos_label='FAKE')
rec5 = metrics.recall_score(Btest, pred, pos_label='FAKE')
f15 = metrics.f1_score(Btest, pred, pos_label='FAKE')
print("accuracy: %0.3f" % score5)
cm = metrics.confusion_matrix(Btest, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

#-----------------XGBOOST CLASSIFIER USING THE COUNT VECTORIZER------------#
Alg2 = XGBClassifier()
Alg2.fit(count_train, Btrain)
pred = Alg2.predict(count_test)
score6 = metrics.accuracy_score(Btest, pred)
pre6 = metrics.precision_score(Btest, pred, pos_label='FAKE')
rec6 = metrics.recall_score(Btest, pred, pos_label='FAKE')
f16 = metrics.f1_score(Btest, pred, pos_label='FAKE')
print("accuracy: %0.3f" % score6)
cm = metrics.confusion_matrix(Btest, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

#----------------- VISUALIZATION OF THE RECALL VALUE-----------------------#
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
data = [rec1, rec3, rec5]
data2 = ['NB', 'PGC', 'XGB']
ax.bar(data2, data)
plt.title('recall comparison for tfidf algorithms')
plt.xlabel('Algorithms')
plt.ylabel('value')
plt.show()

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
data = [rec2, rec4, rec6]
data2 = ['NB', 'PGC', 'XGB']
ax.bar(data2, data)
plt.title('recall comparison for count algorithms')
plt.xlabel('Algorithms')
plt.ylabel('value')
plt.show()

#-------------------------EXTRACTING AN ARTICLE FROM NEWSPAPER-----------------#
from newspaper import Article
url ="####COPY AND PASTE THE URL OF NEWS ARTICLE####"
article = Article(url, language="en")
article.download()
article.parse()
article.nlp()
text= article.summary

#---------------FINDING FAKE OR REAL USING MNB COUNTVECTORIZER------------#
count_test2 = count_vectorizer.transform([text])
Alg.fit(count_train, Btrain)
pred = Alg.predict(count_test2)

#---------------FINDING FAKE OR REAL USING MNB TFIDF VECTORIZER----------#
tfidf_test2 = tfidf_vectorizer.transform([text])
Alg.fit(tfidf_train, Btrain)
pred2 = Alg.predict(tfidf_test2)

#---------------FINDING FAKE OR REAL USING PAC TFIDF VECTORIZER----------#
tfidf_test2 = tfidf_vectorizer.transform([text])
Alg1.fit(tfidf_train, Btrain)
pred3 = Alg.predict(tfidf_test2)

#---------------FINDING FAKE OR REAL USING HASH VECTORIZER------------------#
hash_test2 = hash_vectorizer.transform([text])
Alg1.fit(hash_train, Btrain)
pred4 = Alg1.predict(hash_test2)

#---------------FINDING FAKE OR REAL USING XGB COUNT VECTORIZER------------#
count_test2 = count_vectorizer.transform([text])
Alg2.fit(count_train, Btrain)
pred5 = Alg.predict(count_test2)

#----------------FINDING FAKE OR REAL USING XGB TFIDF VECTORIZER----#
tfidf_test2 = tfidf_vectorizer.transform([text])
Alg.fit(tfidf_train, Btrain)
pred6 = Alg.predict(tfidf_test2)
