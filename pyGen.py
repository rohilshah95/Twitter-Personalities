import csv
import array
import pandas
import pickle
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
csvFile=open('newfrequency300.csv', 'rt')
csvReader=csv.reader(csvFile)
mydict={row[1]: int(row[0]) for row in csvReader}

y=[]
with open ('PJFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('PJFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
print(train.shape)
print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('BNPJFinal.sav', 'wb'))
del result

y=[]
with open ('IEFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('IEFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
print(train.shape)
print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('BNIEFinal.sav', 'wb'))
del result

y=[]
with open ('TFFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('TFFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
print(train.shape)
print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('BNTFFinal.sav', 'wb'))
del result

y=[]
with open ('SNFinaltest.csv', 'rt') as f:
	reader=csv.reader(f)
	corpus=[rows[0] for rows in reader]

with open ('SNFinaltest.csv', 'rt') as f:
	csvReader1=csv.reader(f)
	for rows in csvReader1:
		y.append([int(rows[1])])
vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(corpus).toarray()
result=np.append(x,y,axis=1)
X=pandas.DataFrame(result)
model=GaussianNB()
train = X.sample(frac=0.8, random_state=1)
test=X.drop(train.index)
y_train=train[301]
y_test=test[301]
print(train.shape)
print(test.shape)
xtrain=train.drop(301,axis=1)
xtest=test.drop(301,axis=1)
model.fit(xtrain,y_train)
pickle.dump(model, open('BNSNFinal.sav', 'wb'))
