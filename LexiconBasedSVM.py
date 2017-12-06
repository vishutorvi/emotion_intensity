# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:36:54 2017

@author: vishw
"""
import numpy as np
import pandas as pd
from sklearn import svm

def classify(emotion,featureCount):
    if featureCount == 3:
        features = ['id','sentence','intensity']
        if emotion > 3:
            processeddataset = pd.read_csv('./processeddata/valencetrainset.txt',names = features, sep='\t')
            processedtestset = pd.read_csv('./processedtestdata/valenceset.txt',names = features, sep='\t')
            angerframe2 = pd.read_csv('./scoreddata/rnndata/valenceprocessedtrain.csv',header=0, sep=',')
            angerframetest2 = pd.read_csv('./scoreddata/rnndata/valenceprocessedtest.csv', header=0, sep=',')
            
        if emotion == 0:
            processeddataset = pd.read_csv('./processeddata/angertrainset.txt',names = features, sep='\t')
            processedtestset = pd.read_csv('./processedtestdata/angertestset.txt',names = features, sep='\t')
            angerframe2 = pd.read_csv('./scoreddata/rnndata/angerprocessedtrain.csv',header=0, sep=',')
            angerframetest2 = pd.read_csv('./scoreddata/rnndata/angerprocessedtest.csv', header=0, sep=',')
        
        elif emotion == 1:
            processeddataset = pd.read_csv('./processeddata/feartrainset.txt',names = features, sep='\t')
            processedtestset = pd.read_csv('./processedtestdata/feartestset.txt',names = features, sep='\t')
            angerframe2 = pd.read_csv('./scoreddata/rnndata/fearprocessedtrain.csv',header=0, sep=',')
            angerframetest2 = pd.read_csv('./scoreddata/rnndata/fearprocessedtest.csv', header=0, sep=',')
        
        if emotion == 2:
            processeddataset = pd.read_csv('./processeddata/joytrainset.txt',names = features, sep='\t')
            processedtestset = pd.read_csv('./processedtestdata/joytestset.txt',names = features, sep='\t')
            angerframe2 = pd.read_csv('./scoreddata/rnndata/joyprocessedtrain.csv',header=0, sep=',')
            angerframetest2 = pd.read_csv('./scoreddata/rnndata/joyprocessedtest.csv', header=0, sep=',')
            
        if emotion == 3:
            processeddataset = pd.read_csv('./processeddata/sadnesstrainset.txt',names = features, sep='\t')
            processedtestset = pd.read_csv('./processedtestdata/sadnesstestset.txt',names = features, sep='\t')
            angerframe2 = pd.read_csv('./scoreddata/rnndata/sadnessprocessedtrain.csv',header=0, sep=',')
            angerframetest2 = pd.read_csv('./scoreddata/rnndata/sadnessprocessedtest.csv', header=0, sep=',')    
        
        angerframetest1 = np.array(angerframetest2)
        angerframe1 = np.array(angerframe2)
        labeltrain = np.array(processeddataset['intensity'])
        labeltrain = np.array([labeltrain[:len(labeltrain)-200]])#np.array([list(processeddataset['intensity'])])
        labeltest = np.array([list(processedtestset['intensity'])])
        #print(len(labeltrain.T))
        #print(len(angerframe1))
        angerframe1 = np.append(labeltrain.T, angerframe1, axis = 1)
        angerframetest1 = np.append(labeltest.T, angerframetest1, axis = 1)
    else:    
        if emotion == 0:
            angerframe1 = pd.read_csv('./scoreddata/angeremotionscores.txt', sep='\t')
            angerframetest1 = pd.read_csv('./scoreddata/angertestemotionscores.txt', sep='\t')
        elif emotion == 1:
            angerframe1 = pd.read_csv('./scoreddata/fearemotionscores.txt', sep='\t')
            angerframetest1 = pd.read_csv('./scoreddata/feartestemotionscores.txt', sep='\t')
        elif emotion == 2:
            angerframe1 = pd.read_csv('./scoreddata/joyemotionscores.txt', sep='\t')
            angerframetest1 = pd.read_csv('./scoreddata/joytestemotionscores.txt', sep='\t')
        elif emotion == 3:
            angerframe1 = pd.read_csv('./scoreddata/sadnessemotionscores.txt', sep='\t')
            angerframetest1 = pd.read_csv('./scoreddata/sadnesstestemotionscores.txt', sep='\t')
        else:
            angerframe1 = pd.read_csv('./scoreddata/valenceemotionscores.txt', sep='\t')
            angerframetest1 = pd.read_csv('./scoreddata/valencetestemotionscores.txt', sep='\t')


    angerframe1 = np.array(angerframe1)
    angerframetest1 = np.array(angerframetest1)
    angerframetest1 = np.array(angerframetest1[:,1:],dtype='float64')
    angerframe1 = np.array(angerframe1[:,1:],dtype='float64')    
    
    clf = svm.SVC()
    if featureCount == 1:
        X_train = np.array(angerframe1[:,0:1])
        X_test = np.array(angerframetest1[:,0:1])
    elif featureCount == 2:
        X_train = np.array(angerframe1[:,0:2])
        X_test = np.array(angerframetest1[:,0:2])
    else:
        X_train = angerframe1
        X_test = angerframetest1
    
    if featureCount == 1 or featureCount == 2:    
        Y_train = angerframe1[:,-1]
        Y_test = angerframetest1[:,-1]
    else:
        Y_train = labeltrain[0]
        Y_test = labeltest[0]
        
    clf.fit(X_train, Y_train)
    
    predictions = clf.predict(X_test)
    
    print('Actual Accuracy=',np.sum(predictions == Y_test)/len(angerframetest1))