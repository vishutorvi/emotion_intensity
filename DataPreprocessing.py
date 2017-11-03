# # Sentiment Intensity Affects on Twitter Data
#     Step 1: Data Preprocessing
#             a) Removal of Stem words
#             b) Removal of Stop words
#             c) Removal of Puntuations
#             d) Removal of Emoticons
#     Step 2: Feature Extraction
#             a) Extracting using NRC-Hashtag emotion lexicon

import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import string
from pathlib import Path
import re

emotion = 0 #0=anger, 1=fear, 2=joy, 3=sadness
features = ['id','sentence','emotion','intensity']
stemmer = SnowballStemmer("english")
emotion = 0 

def wordtokenize(text):
    sentence = []
    for x in text.split(" "):
        sentence += [x]
    return sentence
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
        datapre['intensity'][i] = str(datapre['intensity'][i]).split(':')[0]
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
        textfile = open(createname,'w',encoding='utf-8')
        for index, row in datapre.iterrows():
            textfile.write(str(row['id'])+'\t')
            textfile.write(str(row['sentence'])+'\t')
            textfile.write(str(row['intensity'])+'\n')
        textfile.close()
    return datapre[['id','sentence','intensity']]

# # Create dataframes for Anger, Fear, Joy, Sadness datasets
if emotion == 0:    
#Anger dataframe creation
    data = dataframecreator('./trainingdata/EI-oc-En-train/EI-oc-En-anger-train.txt','./processeddata/angertrainset.txt')
elif emotion == 1:    
#Fear dataframe creation
    data = dataframecreator('./trainingdata/EI-oc-En-train/EI-oc-En-fear-train.txt','./processeddata/feartrainset.txt')
elif emotion == 2:    
#Joy dataframe creation
    data = dataframecreator('./trainingdata/EI-oc-En-train/EI-oc-En-joy-train.txt','./processeddata/joytrainset.txt')
else:    
#Sadness dataframe creation
    data = dataframecreator('./trainingdata/EI-oc-En-train/EI-oc-En-sadness-train.txt','./processeddata/sadnesstrainset.txt')
#valence dataframe creation",
#valencedataframe = dataframecreator('./trainingdata/2018-Valence-oc-En-train/2018-Valence-oc-En-train.txt','./processeddata/valencetrainset.txt')

stop_words = np.loadtxt('./stop_words.txt')
i=0
for text in data['sentence']:
    for word in text:
        if(np.any(stop_words[:] == np.str_(word))):
            text.remove(word)
    data['sentence'][i] = text
    i = i + 1
