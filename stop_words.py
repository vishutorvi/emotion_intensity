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




data = pd.read_table('./trainingdata/EI-oc-En-train/EI-oc-En-anger-train.txt', header=None)
data.columns = ['id', 'sentence','emotion','intensity']
#tokenize the msg
data_tokenized = get_tokens(data)
#data_copy = data_tokenized

#stop_words = pd.read_table('../datasets/stop_words.txt', header=None)
stop_words = np.loadtxt('./stop_words.txt', dtype=str)

i=0
for text in data['sentence']:
    new_text=[]
    for word in text:
        if(np.any(stop_words[:] == np.str_(word))):
            continue
        else:
            new_text += [word]
    data['sentence'][i] = new_text
    i = i + 1
        
        
        



