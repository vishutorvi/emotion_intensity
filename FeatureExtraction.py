#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:33:18 2017

@author: soumak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

emotion = 3
def get_score(wordtable,emotion,wordstring):
    wordtable1=wordtable[wordtable['emotion']==emotion] #subset only the matching emotion, may want to do this before to reduce complexity for every word search
    wordemotion = np.array(wordtable1[['word','score']],dtype=str)
    wordemotionstring = wordemotion[:,0]
    if(np.alen(wordemotion[wordemotionstring[:] == wordstring])>0):
        scorearray = wordemotion[wordemotionstring[:] == wordstring]
        return float(scorearray[0,1])
    else:
        return 0.0

def check_emotion(emotiontable,emotion,wordstring):
    try:
        return emotiontable.loc[wordstring,emotion]
    except:
        return 0

features = ['id','sentence','intensity']
wordtable=pd.read_table('./NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2/NRC-Hashtag-Emotion-Lexicon-v0.2.txt',header=None)
wordtable.columns = ['emotion','word','score']
emotiontable=pd.read_table('./NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',header=None)
emotiontable.columns = ['word','emotion','score']
emotiontable=pd.pivot_table(emotiontable,index=['word'],columns=['emotion'],values=['score'])['score']

emotiontext = ''
emotionscores = './scoreddata/'

if emotion == 0:
    processeddataset = pd.read_csv('./processeddata/angertrainset.txt',names = features, sep='\t')
    emotiontext = 'anger'
    emotionscores += emotiontext+'emotionscores.txt'
elif emotion == 1:
    processeddataset = pd.read_csv('./processeddata/feartrainset.txt',names = features, sep='\t')
    emotiontext = 'fear'
    emotionscores += emotiontext+'emotionscores.txt'
elif emotion == 2:
    processeddataset = pd.read_csv('./processeddata/joytrainset.txt',names = features, sep='\t')
    emotiontext = 'joy'
    emotionscores += emotiontext+'emotionscores.txt'
elif emotion == 3:
    processeddataset = pd.read_csv('./processeddata/sadnesstrainset.txt',names = features, sep='\t')
    emotiontext = 'sadness'
    emotionscores += emotiontext+'emotionscores.txt'
else:
    processeddataset = pd.read_csv('./processeddata/valencetrainset.txt',names = features, sep='\t')

angerFeatureFrame = pd.DataFrame()
i = 0
for row in processeddataset['sentence']:
    row = row.replace('[','')
    row = row.replace(']','')
    row = row.replace('\'','')
    row = row.replace(' ','')
    score = 0.0
    for text in row.split(','):
        score = score + get_score(wordtable, emotiontext, text)
    h = pd.Series([str(processeddataset['id'][i]),str(score),str(processeddataset['intensity'][i])])
    angerFeatureFrame = angerFeatureFrame.append(h,ignore_index = True)
    i = i + 1

angerFeatureFrame.to_csv(emotionscores,header=['id','score','intensity'], index = False, sep='\t')