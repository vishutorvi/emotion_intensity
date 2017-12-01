# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:52:32 2017

@author: vishw
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np

def get_emotion_score(wordtable,wordstring,emotion):
    wordtable1=wordtable[wordtable['emotion']==emotion] #subset only the matching emotion, may want to do this before to reduce complexity for every word search
    wordemotion = np.array(wordtable1[['word','score']],dtype=str)
    if(len(wordemotion[wordtable1['word'] == wordstring])>0):
        if(len(wordemotion[wordtable1['word']==wordstring])>0):
            scorearray = wordemotion[wordtable1['word']==(wordstring)]
        else:
            scorearray = wordemotion[wordtable1['word'].str.contains(wordstring)]
        return float(scorearray[0,1])
    else:
        return 0.0

def get_sentiment_score(sentimentwordtable,wordstring):
    if(len(sentimentwordtable[sentimentwordtable['emotion'].str.contains(wordstring) == True])>0):
        if len(sentimentwordtable[sentimentwordtable['emotion'] == (wordstring)])>0:
            scorearray = np.array(sentimentwordtable[sentimentwordtable['emotion'] == wordstring])
        else:
            scorearray = np.array(sentimentwordtable[sentimentwordtable['emotion'].str.contains(wordstring) == True])
        return float(scorearray[0][1])
    else:
        return 0.0

def processingData(wordtable,processedataset,emotiontext):
    i = 0
    angerFeatureFrame = pd.DataFrame()
    for row in processedataset['sentence']:
        scorefeatures = 0.0
        row = row.replace('[','')
        row = row.replace(']','')
        row = row.replace('\'','')
        row = row.replace(' ','')
        importantcount = 0
        for text in row.split(','):
            wordscore = get_emotion_score(wordtable,text,emotiontext)
            if wordscore > 0.5:
                importantcount += 1
                scorefeatures += wordscore
        
        h = pd.Series([scorefeatures,importantcount, str(processedataset['intensity'][i])])
        angerFeatureFrame = angerFeatureFrame.append(h,ignore_index = True)
        i = i + 1
    return angerFeatureFrame

def processingValenceData(sentimentwordtable,processedataset):
    i = 0
    print('Processing Sentiment data.........')
    angerFeatureFrame = pd.DataFrame()
    for row in processedataset['sentence']:
        scorefeatures = 0.0
        row = row.replace('[','')
        row = row.replace(']','')
        row = row.replace('\'','')
        row = row.replace(' ','')
        importantcount = 0
        for text in row.split(','):
            wordscore = get_sentiment_score(sentimentwordtable,text)
            if wordscore > 0.5:
                importantcount += 1
                scorefeatures += wordscore
        
        h = pd.Series([scorefeatures,importantcount, str(processedataset['intensity'][i])])
        angerFeatureFrame = angerFeatureFrame.append(h,ignore_index = True)
        i = i + 1
    return angerFeatureFrame
  
def classify(emotion,features):
    sentimentwordtable = pd.read_table('./NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-unigrams.txt',header=None)
    sentimentwordtable.columns = ['emotion','score','count','intensity']
    wordtable=pd.read_table('./NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2/NRC-Hashtag-Emotion-Lexicon-v0.2.txt',header=None)
    wordtable.columns = ['emotion','word','score']
    features = ['id','sentence','intensity']
    if emotion == 0:
        processeddataset = pd.read_csv('./processeddata/angertrainset.txt',names = features, sep='\t')
        processedtestset = pd.read_csv('./processedtestdata/angertestset.txt',names = features, sep='\t')
        emotiontext = 'anger'
    
    if emotion == 1:
        processeddataset = pd.read_csv('./processeddata/feartrainset.txt',names = features, sep='\t')
        processedtestset = pd.read_csv('./processedtestdata/feartestset.txt',names = features, sep='\t')
        emotiontext = 'fear'
    
    if emotion == 2:
        processeddataset = pd.read_csv('./processeddata/joytrainset.txt',names = features, sep='\t')
        processedtestset = pd.read_csv('./processedtestdata/joytestset.txt',names = features, sep='\t')
        emotiontext = 'joy'
    
    if emotion == 3:
        processeddataset = pd.read_csv('./processeddata/sadnesstrainset.txt',names = features, sep='\t')
        processedtestset = pd.read_csv('./processedtestdata/sadnesstestset.txt',names = features, sep='\t')
        emotiontext = 'sadness'
    
    else:
        processeddataset = pd.read_csv('./processeddata/valencetrainset.txt',names = features, sep='\t')
        processedtestset = pd.read_csv('./processedtestdata/valenceset.txt',names = features, sep='\t')
        emotiontext = 'valence'
    
    
    if emotion == 0 or emotion == 1 or emotion == 2 or emotion == 3:
        trainfrequencyOfWords = np.array(processingData(wordtable, processeddataset,emotiontext))
        testfrequencyOfWords = np.array(processingData(wordtable, processedtestset,emotiontext)) 
    else:
        trainfrequencyOfWords = np.array(processingValenceData(sentimentwordtable,processeddataset))
        testfrequencyOfWords = np.array(processingValenceData(sentimentwordtable,processedtestset)) 
    
    #train and test labels    
    y_train = np.array(trainfrequencyOfWords[:,-1],dtype='int32')
    y_test = np.array(testfrequencyOfWords[:,-1],dtype='int32')
    
    #train and test categorical labels
    labelstrain = np.zeros((len(y_train),4))
    labelstest = np.zeros((len(y_test),4))
    #insert value at there index
    j = 0
    for a in y_train:
        a = int(a)
        labelstrain[j,a]=1
        j = j + 1
        
    #insert value at there index
    k = 0
    for l in y_test:
        l = int(l)
        labelstest[k,l]=1
        k = k + 1
    #features = 1 or features = 2
    if features == 1:
        x_train = trainfrequencyOfWords[:,0:1]
        x_test = testfrequencyOfWords[:,0:1]
    else:
        x_train = trainfrequencyOfWords[:,0:2]
        x_test = testfrequencyOfWords[:,0:2]
    
    #splitting labels
    labelstrain1, labelstrain2 = np.split(labelstrain, [(len(labelstrain)*5)//6])
    x_train = x_train.reshape((len(x_train),1,len(x_train[0])))
    x_test = x_test.reshape((len(x_test),1,len(x_test[0])))
    print('Model Training started.....')
    #Model creation
    model = Sequential()
    neurons = len(y_test)
    model.add(LSTM(neurons,input_shape=(len(x_train[0]),len(x_train[0][0])),dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
    model.add(LSTM(50, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(4))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    
    #fitting the model
    model.fit(x_train, labelstrain,
              batch_size=1,
              epochs=5)
    
    score, acc = model.evaluate(x_test, labelstest,
                                batch_size=1)
    print('Test score:', score)
    print('Test accuracy:', acc)