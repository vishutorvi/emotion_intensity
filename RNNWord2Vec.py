# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:52:32 2017

@author: vishw
"""

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping

def classify(emotion):    
    features = ['id','sentence','intensity']
    if emotion > 3:
        processeddataset = pd.read_csv('./processeddata/valencetrainset.txt',names = features, sep='\t')
        processedtestset = pd.read_csv('./processedtestdata/valenceset.txt',names = features, sep='\t')
        print(processeddataset.head())
        angerframe2 = pd.read_csv('./scoreddata/rnndata/valenceprocessedtrain.csv',header=0, sep=',')
        angerframetest2 = pd.read_csv('./scoreddata/rnndata/valenceprocessedtest.csv', header=0, sep=',')
    
    if emotion == 0:
        processeddataset = pd.read_csv('./processeddata/angertrainset.txt',names = features, sep='\t')
        processedtestset = pd.read_csv('./processedtestdata/angertestset.txt',names = features, sep='\t')
        print(processeddataset.head())
        angerframe2 = pd.read_csv('./scoreddata/rnndata/angerprocessedtrain.csv',header=0, sep=',')
        angerframetest2 = pd.read_csv('./scoreddata/rnndata/angerprocessedtest.csv', header=0, sep=',')
    
    elif emotion == 1:
        processeddataset = pd.read_csv('./processeddata/feartrainset.txt',names = features, sep='\t')
        processedtestset = pd.read_csv('./processedtestdata/feartestset.txt',names = features, sep='\t')
        print(processeddataset.head())
        angerframe2 = pd.read_csv('./scoreddata/rnndata/fearprocessedtrain.csv',header=0, sep=',')
        angerframetest2 = pd.read_csv('./scoreddata/rnndata/fearprocessedtest.csv', header=0, sep=',')
    
    if emotion == 2:
        processeddataset = pd.read_csv('./processeddata/joytrainset.txt',names = features, sep='\t')
        processedtestset = pd.read_csv('./processedtestdata/joytestset.txt',names = features, sep='\t')
        print(processeddataset.head())
        angerframe2 = pd.read_csv('./scoreddata/rnndata/joyprocessedtrain.csv',header=0, sep=',')
        angerframetest2 = pd.read_csv('./scoreddata/rnndata/joyprocessedtest.csv', header=0, sep=',')
        
    if emotion == 3:
        processeddataset = pd.read_csv('./processeddata/sadnesstrainset.txt',names = features, sep='\t')
        processedtestset = pd.read_csv('./processedtestdata/sadnesstestset.txt',names = features, sep='\t')
        print(processeddataset.head())
        angerframe2 = pd.read_csv('./scoreddata/rnndata/sadnessprocessedtrain.csv',header=0, sep=',')
        angerframetest2 = pd.read_csv('./scoreddata/rnndata/sadnessprocessedtest.csv', header=0, sep=',')    
    
    trainfrequencyOfWords = np.array(angerframe2)
    testfrequencyOfWords = np.array(angerframetest2)
    
    y_train = np.array(processeddataset['intensity'])
    y_test = np.array(processedtestset['intensity'])
    y_train = keras.utils.to_categorical(y_train, num_classes=4)
    y_test = keras.utils.to_categorical(y_test, num_classes=4)
    #test labels
    labelstrain = y_train
    labelstest = y_test
    labelstrain = labelstrain.reshape(len(labelstrain),1,4)
    labelstest = labelstest.reshape(len(labelstest),1,4)
    #insert value at there index
    x_train = sequence.pad_sequences(trainfrequencyOfWords[:,:], maxlen=400, dtype='float64')
    x_test = sequence.pad_sequences(testfrequencyOfWords[:,:], maxlen=400, dtype='float64')
    
    x_train = x_train.reshape(len(x_train),1,len(x_train[0]))
    x_test = x_test.reshape(len(x_test),1,len(x_test[0]))

    lenX2 = len(x_train[0][0])
    neurons = 100
    model = Sequential()
    
    model.add(LSTM(neurons,input_shape=(1,lenX2), dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(LSTM(50, dropout=0.2,input_shape=(neurons, 1), recurrent_dropout=0.2, return_sequences=True))
    model.add(Dense(4,input_shape=(200,1)))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    stop = [EarlyStopping(monitor='acc', mode='max', min_delta=0.002, patience=5)]
    #fitting the model
    model.fit(x_train, labelstrain, batch_size=1, shuffle=False,epochs=15,callbacks=stop)
    
    score, acc = model.evaluate(x_test, labelstest,
                                batch_size=10,
                                verbose=2)

    print('Test accuracy:', acc)