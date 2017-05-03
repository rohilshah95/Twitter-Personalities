#  -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import tweepy
import sys
import os
import nltk 
import re
import numpy as np
import string
from unidecode import unidecode
import csv
from itertools import islice
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

ckey='...'
csecret='...'
atoken='...'
asecret='...'
auth=tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api=tweepy.API(auth)

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
def preproc(s):
	#s=emoji_pattern.sub(r'', s) # no emoji
	s= unidecode(s)
	POSTagger=preprocess(s)
	#print(POSTagger)

	tweet=' '.join(POSTagger)
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(tweet)
	#filtered_sentence = [w for w in word_tokens if not w in stop_words]
	filtered_sentence = []
	for w in POSTagger:
	    if w not in stop_words:
	        filtered_sentence.append(w)
	#print(word_tokens)
	#print(filtered_sentence)
	stemmed_sentence=[]
	stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
	for w in filtered_sentence:
		stemmed_sentence.append(stemmer2.stem(w))
	#print(stemmed_sentence)

	temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation) 
	preProcessed=temp.split(" ")
	final=[]
	for i in preProcessed:
		if i not in final:
			if i.isdigit():
				pass
			else:
				if 'http' not in i:
					final.append(i)
	temp1=' '.join(c for c in final)
	#print(preProcessed)
	return temp1

def getTweets(user):
	csvFile = open('user.csv', 'a', newline='')
	csvWriter = csv.writer(csvFile)
	try:
		for i in range(0,4):
			tweets=api.user_timeline(screen_name = user, count = 1000, include_rts=True, page=i)
			for status in tweets:
				tw=preproc(status.text)
				if tw.find(" ") == -1:
					tw="blank"
				csvWriter.writerow([tw])
	except tweepy.TweepError:
		print("Failed to run the command on that user, Skipping...")
	csvFile.close()



username=input("Please Enter Twitter Account handle: ")
getTweets(username)
with open('user.csv','rt') as f:
	csvReader=csv.reader(f)
	tweetList=[rows[0] for rows in csvReader]
os.remove('user.csv')
with open('newfrequency300.csv','rt') as f:
	csvReader=csv.reader(f)
	mydict={rows[1]: int(rows[0]) for rows in csvReader}

vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
x=vectorizer.fit_transform(tweetList).toarray()
df=pd.DataFrame(x)


model_IE = pickle.load(open("BNIEFinal.sav", 'rb'))
model_SN = pickle.load(open("BNSNFinal.sav", 'rb'))
model_TF = pickle.load(open('BNTFFinal.sav', 'rb'))
model_PJ = pickle.load(open('BNPJFinal.sav', 'rb'))

answer=[]
IE=model_IE.predict(df)
SN=model_SN.predict(df)
TF=model_TF.predict(df)
PJ=model_PJ.predict(df)


b = Counter(IE)
value=b.most_common(1)
print(value)
if value[0][0] == 1.0:
	answer.append("I")
else:
	answer.append("E")

b = Counter(SN)
value=b.most_common(1)
print(value)
if value[0][0] == 1.0:
	answer.append("S")
else:
	answer.append("N")

b = Counter(TF)
value=b.most_common(1)
print(value)
if value[0][0] == 1:
	answer.append("T")
else:
	answer.append("F")

b = Counter(PJ)
value=b.most_common(1)
print(value)
if value[0][0] == 1:
	answer.append("P")
else:
	answer.append("J")
mbti="".join(answer)
print(mbti)
