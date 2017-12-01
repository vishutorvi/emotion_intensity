# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 00:22:46 2017

@author: vishw
"""

from FeatureExtractionForTest import extracting as extractingtest
from FeatureExtraction import extracting as extractingtrain
from DataPreprocessing import preprocessingData as trainpreprocessing
from DataPreprocessingTest import preprocessingData as testpreprocessing
from RNNLexiconBased import classify as RNNClassify
from RNNWord2Vec import classify as RNNWord2VecClassify
from LexiconBasedSVM import classify as SVMClassify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from time import time
import numpy as np
import sys

def main():
    emotion = 0
    argument = str(sys.argv)
    argument = argument.replace('[','')
    argument = argument.replace(']','')
    argument = argument.replace("'",'')
    argument = argument.replace(' ','')
    arguments = argument.split(',')
    print(arguments)
    if len(arguments)<4:
        raise ValueError('Minimum number of arguments is:',3)
        

    #First Argument is Emotion    
    emotionText = arguments[1]#0=anger, 1=fear, 2=joy, 3=sadness
    print(emotionText)
    if emotionText == 'anger':
        emotion = 0
    elif emotionText == 'fear':
        emotion = 1
    elif emotionText == 'joy':
        emotion = 2
    elif emotionText == 'sadness':
        emotion = 3
    else:
        emotion = 4
    
    print(emotion)
    #Step 1: Datapreprocessing
    print('Data Preprocessing Phase')
    print('Starting........')
    #training data
    trainpreprocessing(emotion)
    #testing data
    testpreprocessing(emotion)
    
    #Step 2: FeatureExtraction
    #Second Argument Type of Feature Extraction
    print('Feature Extraction Process')
    print('Starting.....')
    typeOfFeatureExtraction = arguments[2]
    print(typeOfFeatureExtraction)
    if typeOfFeatureExtraction == 'Lexicon-Single':
        extractingtrain(emotion, 1)
        extractingtest(emotion, 1)
    if typeOfFeatureExtraction == 'Lexicon-Two':
        extractingtrain(emotion, 2)
        extractingtest(emotion, 2)
    
    #Step 3: Classification
    #Type of Classifier to call
    print('Process of classifying')
    print('Starting.....')
    classifierType = arguments[3]
    print(classifierType)
    
    if classifierType =='svm':
        if typeOfFeatureExtraction == 'Lexicon-Single':
            SVMClassify(emotion, 1)
        elif typeOfFeatureExtraction == 'Lexicon-Two':
            SVMClassify(emotion, 2)
        elif typeOfFeatureExtraction == 'Ngrams':
            print('NGrams')
        else:
            SVMClassify(emotion, 3)
        
    
    if classifierType == 'lstm':
        if typeOfFeatureExtraction == 'Lexicon-Single':
            RNNClassify(emotion, 1)
        elif typeOfFeatureExtraction == 'Lexicon-Two':
            RNNClassify(emotion, 2)
        else:
            RNNWord2VecClassify(emotion)

if __name__ == "__main__":
    main()