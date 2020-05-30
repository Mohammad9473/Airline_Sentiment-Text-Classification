# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.svm import SVC


#load dataset
df = pd.read_csv('PATH TO CSV FILE', encoding='latin-1')



#Select airline sentiment except the neutral ones
df = df[df['airline_sentiment'] != 'neutral']

print(df['airline_sentiment'].value_counts())


df_majority = df[df['airline_sentiment'] == 'negative']
df_minority = df[df['airline_sentiment'] == 'positive']

from sklearn.utils import resample

df_majority_downsampled = resample(df_majority, replace = True, n_samples = 2363, random_state = 4)

df = pd.concat([df_minority,df_majority_downsampled])

print(df['airline_sentiment'].value_counts())


X = df['text'] #target 
Y = df['airline_sentiment'] #label

#split train test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2, stratify= Y) #tries to split equally coressponding to labels
#print(X_train)


vectorizer = CountVectorizer(stop_words='english', min_df = 0.02)



X_train_words = vectorizer.fit_transform(X_train) 

print(len(Y_train))
print(len(Y_test))
#print(X_train_words['data'])
#print(newsgroups_train['target'][1000])

vocabulary_dict = vectorizer.vocabulary_ 

vocabulary_list = vectorizer.get_feature_names() 

print(vocabulary_list)




vocabulary = np.asarray(vocabulary_list) 


stopwords = vectorizer.stop_words_ 



doc_term_train = X_train_words.todense().getA()
print(doc_term_train)
stopwords_list = list(stopwords);
print(stopwords_list)


#creating classifeir
clf = SVC(gamma='auto',probability=True) 

clf.fit(X_train_words,Y_train); 


Y_train_words_pred = clf.predict(X_train_words) 

train_accuracy = np.mean(Y_train_words_pred == Y_train)

print(train_accuracy) 



#train_accuracy1 = clf.score(X_train_words,Y_train);
#print(train_accuracy1)

print(metrics.classification_report(Y_train, Y_train_words_pred)); 

#n_train = len(X.data);


train_conf_mat = confusion_matrix(Y_train, Y_train_words_pred) 

print(train_conf_mat)

X_test_words = vectorizer.transform(X_test)


doc_term_test =X_test_words.todense().getA()
#print(doc_term_test)
                                    

Y_test_words_pred = clf.predict(X_test_words)
test_accuracy = np.mean(Y_test_words_pred == Y_test)
print(test_accuracy)



print(metrics.classification_report(Y_test, Y_test_words_pred))

test_conf_mat = metrics.confusion_matrix(Y_test, Y_test_words_pred)



print(test_conf_mat)




#Another Methods
#naive bayees classifier for multinoial models
#is suitable mostly for discrete features



#Building the model: multinomial 
print('multinomial')
Model = MultinomialNB()
Model.fit(X_train_words , Y_train) 



#Evaluate Model on Train set
Y_train_predict = Model.predict(X_train_words) 


print(accuracy_score(Y_train, Y_train_predict)) 
#almost same result accuracy close to 80%

#Evaluate Model on Test set
X_test_words = vectorizer.transform(X_test)

Y_test_predict = Model.predict(X_test_words)

print(accuracy_score(Y_test, Y_test_predict))

#Building the model: logistic Regression 
print('logistic')
Model = LogisticRegression(solver = 'lbfgs')
Model.fit(X_train_words , Y_train)

#Evaluate Model on Train set
Y_train_predict = Model.predict(X_train_words)

print(accuracy_score(Y_train, Y_train_predict))

#Evaluate Model on Test set
X_test_words = vectorizer.transform(X_test)

Y_test_predict = Model.predict(X_test_words)

print(accuracy_score(Y_test, Y_test_predict))

#print(Y_train.value_counts())

#print(vectorizer.vocabulary_)
