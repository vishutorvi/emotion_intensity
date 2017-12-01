# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:05:01 2017

@author: vishw
"""
from __future__ import print_function
from time import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import sys

argument = str(sys.argv)
argument = argument.replace('[','')
argument = argument.replace(']','')
argument = argument.replace("'",'')
argument = argument.replace(' ','')
arguments = argument.split(',')
print(arguments)
if len(arguments)<2:
    raise ValueError('Minimum number of arguments is:',1)

emotion = 0
emotiontext = arguments[1]
if emotiontext == 'anger':
    emotion = 0
elif emotiontext == 'fear':
    emotion = 1
elif emotiontext == 'joy':
    emotion = 2
elif emotiontext == 'sadness':
    emotion = 3
elif emotiontext == 'valence':
    emotion = 4
    
#Training data
print('SVM Ngrams classify.....',emotion)
features = ['id','sentence','intensity']
if emotion == 0:
    processeddataset = pd.read_csv('./processeddata/angertrainset.txt',names = features, sep='\t')
    processedtestdataset = pd.read_csv('./processedtestdata/angertestset.txt',names = features, sep='\t')
elif emotion == 1:
    processeddataset = pd.read_csv('./processeddata/feartrainset.txt',names = features, sep='\t')
    processedtestdataset = pd.read_csv('./processedtestdata/feartestset.txt',names = features, sep='\t')
elif emotion == 2:
    processeddataset = pd.read_csv('./processeddata/joytrainset.txt',names = features, sep='\t')
    processedtestdataset = pd.read_csv('./processedtestdata/joytestset.txt',names = features, sep='\t')
elif emotion == 3:
    processeddataset = pd.read_csv('./processeddata/sadnesstrainset.txt',names = features, sep='\t')
    processedtestdataset = pd.read_csv('./processedtestdata/sadnesstestset.txt',names = features, sep='\t')
else:
    processeddataset = pd.read_csv('./processeddata/valencetrainset.txt',names = features, sep='\t')
    processedtestdataset = pd.read_csv('./processedtestdata/valenceset.txt',names = features, sep='\t')

data = []
for row in processeddataset['sentence']:
    sentencedata = ''

    row = row.replace('[','')
    row = row.replace(']','')
    row = row.replace('\'','')
    row = row.replace(' ','')
    
    for text in row.split(','):
        sentencedata += text+' '
    data += [sentencedata]

labels = np.array(processeddataset['intensity'])

#test data

datatest = []
for row in processedtestdataset['sentence']:

    sentencedata = ''
    row = row.replace('[','')
    row = row.replace(']','')
    row = row.replace('\'','')
    row = row.replace(' ','')
    for text in row.split(','):
        sentencedata += text+' '
    datatest += [sentencedata]

testlabels = np.array(processedtestdataset['intensity'])

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000),#50000
    'vect__ngram_range': ((1, 2), (2, 2), (1, 3)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__max_iter': (10, 50, 80)#500,100
}
# multiprocessing requires the fork to happen in a __main__ protected
# block

# find the best parameters for both the feature extraction and the
# classifier
if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    grid_search.fit(data, labels.astype(int))
    print("done in %0.3fs" % (time() - t0))
    print()
    
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    prediction = grid_search.best_estimator_.predict(datatest)
    print('predictions = ',(np.sum(prediction == testlabels)/len(datatest))*100)        

