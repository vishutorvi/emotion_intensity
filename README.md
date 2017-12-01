# Sentiment Intensity Affects on Twitter Data
    
    # Step 1: Data Preprocessing
	a) Removal of Stem words
        b) Removal of Stop words
        c) Removal of Puntuations
        d) Removal of Emoticons
	f) Removal of Stop words -In progress
			
    # Step 2: Feature Selection <--Yet to Start-->
	1) Using NRC Hashtag Emotion Association Lexicon (NRC-Hash-Emo)
		2 Approach Feature Extraction
			a) Score Feature Only
			b) Score Feature and ImportantWordCount 
	2) Bag of N-Grams 
	3) Using Word2Vec Word Embedding
		
    # Step 3: Sentiment Classification <--Yet to Start-->
	1) SVM Classifier
	2) LSTM Neural Network
		
    # Step 4: Experiment Results <--Yet to Start-->
		File named PerformanceDataset.txt has results
		
    # Final Step: Final Report

	# Steps to follow for running this project
	
	1) SVM Classifier
		a) Only Score Feature
			i) Detecting Intensity of Emotion
			Anger:
				Run:
					python DetectEmotionIntensity.py anger Lexicon-Single svm
					
			Fear:
				Run:
					python DetectEmotionIntensity.py fear Lexicon-Single svm
					
			Joy:
				Run:
					python DetectEmotionIntensity.py joy Lexicon-Single svm
					
			Sadness:
				Run:
					python DetectEmotionIntensity.py sadness Lexicon-Single svm
				
			ii) Detecting Intensity of Sentiment
				Run:
					python DetectEmotionIntensity.py valence Lexicon-Single svm
				
		b) Score, ImportantWordCount Feature
			i) Detecting Intensity of Emotion
			Anger:
				Run:
					python DetectEmotionIntensity.py anger Lexicon-Two svm
					
			Fear:
				Run:
					python DetectEmotionIntensity.py fear Lexicon-Two svm
					
			Joy:
				Run:
					python DetectEmotionIntensity.py joy Lexicon-Two svm
					
			Sadness:
				Run:
					python DetectEmotionIntensity.py sadness Lexicon-Two svm
					
			ii) Detecting Intensity of Sentiment
				Run:
					python DetectEmotionIntensity.py valence Lexicon-Two svm
				
		c) Word of N-Grams
			i) Detecting Intensity of Emotion
			Anger:
				Run:
					python SVM_Ngrams.py anger
					
			Fear:
				Run:
					python SVM_Ngrams.py fear
					
			Joy:
				Run:
					python SVM_Ngrams.py joy
					
			Sadness:
				Run:
					python SVM_Ngrams.py sadness
					
			ii) Detecting Intensity of Sentiment
				Run:
					python SVM_Ngrams.py valence
				
		d) Word2Vec Word Embedding
			i) Detecting Intensity of Emotion
			Anger:
				Run:
					python DetectEmotionIntensity.py anger Word2Vec svm
					
			Fear:
				Run:
					python DetectEmotionIntensity.py fear Word2Vec svm
					
			Joy:
				Run:
					python DetectEmotionIntensity.py joy Word2Vec svm
					
			Sadness:
				Run:
					python DetectEmotionIntensity.py sadness Word2Vec svm
					
			ii) Detecting Intensity of Sentiment
				Run:
					python DetectEmotionIntensity.py valence Lexicon-Single svm
				
	2) LSTM Classifier
		a) Only Score Feature
			i) Detecting Intensity of Emotion
			Anger:
				Run:
					python DetectEmotionIntensity.py anger Lexicon-Single lstm
					
			Fear:
				Run:
					python DetectEmotionIntensity.py fear Lexicon-Single lstm
					
			Joy:
				Run:
					python DetectEmotionIntensity.py joy Lexicon-Single lstm
					
			Sadness:
				Run:
					python DetectEmotionIntensity.py sadness Lexicon-Single lstm
					
			ii) Detecting Intensity of Sentiment
				Run:
					python DetectEmotionIntensity.py valence Lexicon-Single lstm
				
		b) Score, ImportantWordCount Feature
			i) Detecting Intensity of Emotion
			Anger:
				Run:
					python DetectEmotionIntensity.py anger Lexicon-Two lstm
					
			Fear:
				Run:
					python DetectEmotionIntensity.py fear Lexicon-Two lstm
					
			Joy:
				Run:
					python DetectEmotionIntensity.py joy Lexicon-Two lstm
					
			Sadness:
				Run:
					python DetectEmotionIntensity.py sadness Lexicon-Two lstm
					
			ii) Detecting Intensity of Sentiment
				Run:
					python DetectEmotionIntensity.py valence Lexicon-Two lstm
				
		c) Word2Vec Word Embedding
			i) Detecting Intensity of Emotion
			Anger:
				Run:
					python DetectEmotionIntensity.py anger Word2Vec lstm
					
			Fear:
				Run:
					python DetectEmotionIntensity.py fear Word2Vec lstm
					
			Joy:
				Run:
					python DetectEmotionIntensity.py joy Word2Vec lstm
					
			Sadness:
				Run:
					python DetectEmotionIntensity.py sadness Word2Vec lstm
					
			ii) Detecting Intensity of Sentiment
				Run:
					python DetectEmotionIntensity.py valence Word2Vec lstm
				