# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:31:17 2017

@author: vishw
"""

import numpy as np
import pandas as pd
import string
from pathlib import Path
import re

def wordtokenize(text):
    sentence = []
    for x in text.split(" "):
        sentence += [x]
    return sentence

def stopWordRemoval(data):
    stop_words = np.loadtxt('./stop_words.txt', dtype=str)
    new_stop_words = []
    for x in range(len(stop_words)):
        new_stop_words +=[stop_words[x].translate(str.maketrans('','',string.punctuation.replace("#","").replace('!','')))]
    
    new_stop_words = np.array(new_stop_words)
    i=0
    for text in data['sentence']:
        new_text = []
        for word in text:
            if(np.any(new_stop_words[:] == word)):
                continue
            else:
                new_text += [word]
        data['sentence'][i] = new_text
        i = i + 1
    return data

# # Data Preprocessing:
#     Step 1: Stemmer Removal
#     Step 2: Emotion Removal
#     Step 3: Puntuation Removal
def stemEmotionRemoval(datapre):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F97F"      # 64F  # emoticons
        u"\U0001F300-\U0001F9FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F173-\U0001F1FF"  # flags (iOS)
        u"\u2B50\u2600-\u26FF\u2700-\u27BF]+", 
            flags=re.UNICODE)
    i = 0
    for text in datapre['sentence']:
        lowers = text.lower()
        no_punctuation = lowers.translate(str.maketrans('','',string.punctuation.replace("#","")))
        no_puntuation_in = emoji_pattern.sub(r'', no_punctuation)
        datapre['sentence'][i] = wordtokenize(no_puntuation_in)
        datapre['intensity'][i] = int(str(datapre['intensity'][i]).split(':')[0])
        i = i + 1
    return datapre[['id','sentence','intensity']]

def dataframecreator(filename,createname):
    datapre = pd.DataFrame({'A' : []})
    my_file = Path(createname)
    if my_file.is_file():
        features = ['id','sentence','intensity']
        datapre = pd.read_table(createname,names=features,sep='\t')
    else:
        features = ['id','sentence','emotion','intensity']
        datapre = pd.read_table(filename,names=features)
        datapre = stemEmotionRemoval(datapre)
        datapre = stopWordRemoval(datapre)
        textfile = open(createname,'w',encoding='utf-8')
        for index, row in datapre.iterrows():
            textfile.write(str(row['id'])+'\t')
            textfile.write(str(row['sentence'])+'\t')
            textfile.write(str(row['intensity'])+'\n')
        textfile.close()
    return datapre[['id','sentence','intensity']]
    
def preprocessingData(emotion):
    print(emotion)
    if emotion == 0:    
        #Anger dataframe creation
        data = dataframecreator('./testdata/2018-EI-oc-En-dev/2018-EI-oc-En-anger-dev.txt','./processedtestdata/angertestset.txt')
    elif emotion == 1:    
        #Fear dataframe creation
        data = dataframecreator('./testdata/2018-EI-oc-En-dev/2018-EI-oc-En-fear-dev.txt','./processedtestdata/feartestset.txt')
    elif emotion == 2:    
        #Joy dataframe creation
        data = dataframecreator('./testdata/2018-EI-oc-En-dev/2018-EI-oc-En-joy-dev.txt','./processedtestdata/joytestset.txt')
    elif emotion == 3:    
        #Sadness dataframe creation
        data = dataframecreator('./testdata/2018-EI-oc-En-dev/2018-EI-oc-En-sadness-dev.txt','./processedtestdata/sadnesstestset.txt')
    else:
        #valence dataframe creation",
        valencedataframe = dataframecreator('./testdata/2018-Valence-oc-En-dev/2018-Valence-oc-En-dev.txt','./processedtestdata/valenceset.txt')