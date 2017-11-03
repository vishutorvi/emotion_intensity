#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:33:18 2017

@author: soumak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
import nltk

def get_tokens(datapre):
    #U+1F35E   U+1F9C0 U+2B50
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F97F"      # 64F  # emoticons
        u"\U0001F300-\U0001F9FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F173-\U0001F1FF"  # flags (iOS)
        u"\u2B50\u2600-\u26FF\u2700-\u27BF]+", 
            flags=re.UNICODE)
    i = 0
    for text in datapre['sentence']:
        text = text.lower()
        text = text.translate(str.maketrans('','',string.punctuation))
        text = emoji_pattern.sub(r'', text)
        datapre['sentence'][i] = nltk.word_tokenize(text)
        datapre['intensity'][i] = str(datapre['intensity'][i]).split(':')[0]
        i = i + 1
    #datapre['intensity'] = str(datapre[intensity]).split(':')[0]
    return datapre[['id','sentence','intensity']]

def get_score(wordtable,emotion,wordstring):
    #if not isinstance(wordtable, pd.DataFrame): #check if file exists
     #   return 0
    wordtable=wordtable.loc[wordtable['emotion']==emotion] #subset only the matching emotion, may want to do this before to reduce complexity for every word search
    return wordtable.loc[wordtable['word']==wordstring,'score'] #return only the float score value, empty if not found

def check_emotion(emotiontable,emotion,wordstring):
    #if not isinstance(emotiontable, pd.DataFrame): #check if file exists
     #   return 0
    try:
        return emotiontable.loc[wordstring,emotion]
    except:
        return 0


wordtable=pd.read_table('NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2/NRC-Hashtag-Emotion-Lexicon-v0.2.txtget',header=None)
wordtable.columns = ['emotion','word','score']

emotiontable=pd.read_table('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',header=None)
emotiontable.columns = ['word','emotion','score']
emotiontable=pd.pivot_table(emotiontable,index=['word'],columns=['emotion'],values=['score'])['score']



data = pd.read_table('../datasets/EI-oc-En-train/EI-oc-En-anger-train.txt')
data.columns = ['id', 'sentence','emotion','intensity']
#tokenize the msg
data_tokenized = get_tokens(data)
#data_copy = data_tokenized

#stop_words = pd.read_table('../datasets/stop_words.txt', header=None)
stop_words = np.loadtxt('../datasets/stop_words.txt', dtype=str)
i=0
for text in data['sentence']:
    for word in text:
        if(np.any(stop_words[:] == np.str_(word))):
            text.remove(word)
    data['sentence'][i] = text
    i = i + 1
        
        
        



