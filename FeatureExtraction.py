#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:33:18 2017

@author: soumak
"""

import numpy as np
import pandas as pd
from pathlib import Path

#Method to score emotion intensity
def get_score(wordtable,emotion,wordstring):
    wordtable1=wordtable[wordtable['emotion']==emotion] #subset only the matching emotion, may want to do this before to reduce complexity for every word search
    wordemotion = np.array(wordtable1[['word','score']],dtype=str)
    if(len(wordemotion[wordtable1['word']==wordstring])>0):
        scorearray = wordemotion[wordtable1['word']==(wordstring)]
        return float(scorearray[0,1])
    else:
        return 0.0

#Method to score sentiment intensity
def get_valence_score(sentimentwordtable,emotion,wordstring):
    if(len(sentimentwordtable[sentimentwordtable['emotion'].str.contains(wordstring) == True])>0):
        if len(sentimentwordtable[sentimentwordtable['emotion'] == (wordstring)])>0:
            scorearray = np.array(sentimentwordtable[sentimentwordtable['emotion'] == wordstring])
        else:
            scorearray = np.array(sentimentwordtable[sentimentwordtable['emotion'].str.contains(wordstring) == True])

        return float(scorearray[0][1])
    else:
        return 0.0

#Method to extract features for specific emotions
def extracting(emotion,features):
    features = ['id','sentence','intensity']
    #NRC Hashtag Emotion Lexicons
    wordtable=pd.read_table('./NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2/NRC-Hashtag-Emotion-Lexicon-v0.2.txt',header=None)
    wordtable.columns = ['emotion','word','score']
    #NRC Hashtag Sentiment Lexicons
    sentimentwordtable = pd.read_table('./NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-unigrams.txt',header=None)
    sentimentwordtable.columns = ['emotion','score','count','intensity']
    
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
        emotiontext = 'valence'
        emotionscores += emotiontext+'emotionscores.txt'
        
    angerFeatureFrame = pd.DataFrame()
    i = 0
    
    if emotion > 3:
        my_file = Path(emotionscores)
        if my_file.is_file():
            pass
        else:
            for row in processeddataset['sentence']:
                importantwordcount = 0
                totalWords = 0
                row = row.replace('[','')
                row = row.replace(']','')
                row = row.replace('\'','')
                row = row.replace(' ','')
                score = 0.0
                hashtagfinalscore = 0
                for t in row.split(','):
                    hashtagscore = 0.0
                    text = t.replace(' ','')
                    if len(text) > 0:
                        wordscore = get_valence_score(sentimentwordtable, emotiontext, text)
                        if(wordscore >= 1):
                            importantwordcount += 1
                        score = score + wordscore
        
                h = pd.Series([str(processeddataset['id'][i]), score, str(processeddataset['intensity'][i])])#importantwordcount#hashtagfinalscore
                angerFeatureFrame = angerFeatureFrame.append(h,ignore_index = True)
                i = i + 1
    
    else:
        my_file = Path(emotionscores)
        if my_file.is_file():
            pass
        else:
            for row in processeddataset['sentence']:
                importantwordcount = 0
                totalWords = 0
                row = row.replace('[','')
                row = row.replace(']','')
                row = row.replace('\'','')
                row = row.replace(' ','')
                score = 0.0
                hashtagfinalscore = 0
                for text in row.split(','):
                    totalWords += 1
                    hashtagscore = 0.0
                    if('#' in text):
                        hashtagscore = get_score(wordtable, emotiontext, text)
                        if hashtagscore>=0.5:
                            hashtagfinalscore += hashtagscore
                    else:
                        wordscore = get_score(wordtable, emotiontext, text)
                    if(wordscore >= 0.5):
                        importantwordcount += 1
                    score = score + wordscore + hashtagscore
                if features == 1:
                    h = pd.Series([str(processeddataset['id'][i]),score,str(processeddataset['intensity'][i])])#importantwordcount#hashtagfinalscore
                if features == 2:
                    h = pd.Series([str(processeddataset['id'][i]),score,importantwordcount,str(processeddataset['intensity'][i])])#importantwordcount#hashtagfinalscore
                angerFeatureFrame = angerFeatureFrame.append(h,ignore_index = True)
                i = i + 1
    if len(angerFeatureFrame)>1:
        if len(angerFeatureFrame.columns) == 3:
            angerFeatureFrame.to_csv(emotionscores,header=['id','score','intensity'], index = False, sep='\t')
        elif len(angerFeatureFrame.columns) == 4:
            angerFeatureFrame.to_csv(emotionscores,header=['id','score','importantwordcount','intensity'], index = False, sep='\t')
    
    print('Done with Feature Extraction....Training set')