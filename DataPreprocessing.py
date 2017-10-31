
# coding: utf-8

# # Sentiment Intensity Affects on Twitter Data
#     Step 1: Data Preprocessing
#             a) Removal of Stem words
#             b) Removal of Stop words
#             c) Removal of Puntuations
#             d) Removal of Emoticons
#     Step 2: <--To be done-->

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
import nltk
from nltk.stem.snowball import SnowballStemmer
import string
from pathlib import Path
import re


# In[2]:


get_ipython().magic('matplotlib inline')


# In[3]:


features = ['id','sentence','emotion','intensity']
stemmer = SnowballStemmer("english")


# # Data Preprocessing:
#     Step 1: Stemmer Removal
#     Step 2: Emotion Removal
#     Step 3: Puntuation Removal

# In[4]:


def stemEmotionRemoval(datapre):
    tokens = []
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
        lowers = text.lower()
        no_punctuation = lowers.translate(str.maketrans('','',string.punctuation))
        no_puntuation_in = emoji_pattern.sub(r'', no_punctuation)
        datapre['sentence'][i] = nltk.word_tokenize(no_puntuation_in)
        datapre['intensity'][i] = str(datapre['intensity'][i]).split(':')[0]
        i = i + 1
    #datapre['intensity'] = str(datapre[intensity]).split(':')[0]
    return datapre[['id','sentence','intensity']]


# In[5]:


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
        textfile = open(createname,'w',encoding='utf-8')
        for index, row in datapre.iterrows():
            textfile.write(str(row['id'])+'\t')
            textfile.write(str(row['sentence'])+'\t')
            textfile.write('\t'+str(row['intensity'])+'\n')
        textfile.close()
        return datapre


# # Create dataframes for Anger, Fear, Joy, Sadness datasets

# In[ ]:


#Anger dataframe creation
angerdataframe = dataframecreator('./EI-oc-En-train/EI-oc-En-anger-train.txt','angertrainset.txt')


# In[ ]:


#Fear dataframe creation
feardataframe = dataframecreator('./EI-oc-En-train/EI-oc-En-fear-train.txt','feartrainset.txt')


# In[ ]:


#Joy dataframe creation
joydataframe = dataframecreator('./EI-oc-En-train/EI-oc-En-joy-train.txt','joytrainset.txt')


# In[ ]:


#Sadness dataframe creation
sadnessdataframe = dataframecreator('./EI-oc-En-train/EI-oc-En-sadness-train.txt','sadnesstrainset.txt')


# In[ ]:


print(angerdataframe)

