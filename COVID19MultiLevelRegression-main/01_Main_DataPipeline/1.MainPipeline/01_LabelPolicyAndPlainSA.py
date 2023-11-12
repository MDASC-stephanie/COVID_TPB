
# STEP 01
#
# OBJECTIVE
#  - THIS .PY IS FOR CONDUCTING THE POLICY CLASSIFICATION AND ALSO THE PLAIN SENTIMENT ANALYSIS
#
# MAIN INPUT
#  - TWEETS FOR DIFFERENT COUNTRIES (INPUT PER COUNTRY)
# 
# MAIN OUTPUT
#  - TWEETS FOR DIFFERENT COUNTRIES WITH PREDICTED POLICY LABELS AND OVERALL SENTIMENT SCORES (OUTPUT PER COUNTRY)


# IMPORT THE REQUIRED PACKAGES
print('START IMPORTING THE PACKAGES')
import numpy as np
import pandas as pd
import topicClassifier
import sentimentAnalysis
import common
import regressionAnalysis
import topicModelling
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import KeyedVectors
import tensorflow
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")

# READ THE DATAFILE ABOUT THE COLLECTED TWEETS
print('LOAD THE TWEETS')
tweetFile = pd.read_csv(r'./AUSImportantPeriodTweets_.csv', lineterminator='\n') # INPUT THE DATA

# TOPIC CLASSIFIER
print('==================================================')
print('START THE TOPIC CLASSIFICATION SESSION')
print('==================================================')
print("RENAME AND PRE-PROCESS THE DATA")
tweetsFileDF = topicClassifier.preprocessDF(tweetFile, 'Australia')

# WORD EMBEDDING
print('LOAD THE WORDEMBEDDING MODEL')
word2VecmodelLoad = KeyedVectors.load('./interimData/embedding_word2vec.txt', mmap='r')
embeddingDictionary = topicClassifier.embeddingDictionaryPrep(word2VecmodelLoad)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(embeddingDictionary)

# LOAD THE TRAINED MODEL
print('LOAD THE TRAINED TOPIC CLASSIFICATION MODEL')
model_word2vec = tensorflow.keras.models.load_model('model01')

# LABEL THE DATA
print('START LABELLING THE TWEETS')
tweetsFileDF = topicClassifier.topicClassificationModel(tokenizer, tweetsFileDF, model_word2vec)
tweetsFileDF.to_csv('./AUSImportantPeriodTweetsTopicNew.csv', index = False)
print('GROUP BY RESULT:')
print(tweetsFileDF.groupby('policyLabel_pred')['country'].count())

# SENTIMENT ANALYSIS
print('==================================================')
print('START THE SENTIMENT ANALYSIS SESSION')
print('==================================================')

# PROCESS THE DATA
print('START PREPROCESSING THE DATA')
tweetsFileDFProceed = sentimentAnalysis.sa_pre(tweetsFileDF)
# CONDUCTING SENTIMENT ANALYSIS
print('START CONDUCTING SENTIMENT ANALYSIS')
tweetsFileDFwithSA = sentimentAnalysis.sa_textblob(tweetsFileDFProceed)
tweetsFileDFwithSA.to_csv('./AUSImportantPeriodTweetsTopicNewwithSA.csv', index = False)

print('=== INFORMATION ON THE TWEETS WITH SA OUTPUT ===')
print('NUMBER OF ROWS = ' + str(tweetsFileDFwithSA.shape[0]))
print('NUMBER OF COLUMNS = ' + str(tweetsFileDFwithSA.shape[1]))
print('COLUMN NAMES: ' + str(tweetsFileDFwithSA.columns.to_list()))
print('SAMPLE DATA: ')
print(tweetsFileDFwithSA.head(5))