
# STEP 02
#
# OBJECTIVE
#  - THIS .PY IS FOR CONDUCTING THE ASPECT BASED SENTIMENT ANALYSIS
#
# MAIN INPUT
#  - TWEETS FOR DIFFERENT COUNTRIES WITH PREDICTED POLICY LABELS AND OVERALL SENTIMENT SCORES (INPUT PER COUNTRY)
# 
# MAIN OUTPUT
#  - TWEETS FOR DIFFERENT COUNTRIES WITH TPB SCORES (OUTPUT PER COUNTRY)

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
import quantitative_SA
import TPBSetReturn

print('LOAD THE DATASETS')
print('START WORKING ON THE SET')
setLst = TPBSetReturn.dataSet()
print('LOAD THE TWEETS')
tweetsFileDFwithSA = pd.read_csv('./AUSImportantPeriodTweetsTopicNewwithSA.csv') # INPUT THE FILE FOR PROCESSING, PROCESSING THE FILE FROM ONE COUNTRY PER EXECUTION

print('PRE-PROCESSING THE DATA')
tweetsFilewithSA_PreProcessed = quantitative_SA.pre_absa_tpb(tweetsFileDFwithSA)
print(tweetsFilewithSA_PreProcessed.head(5))
tweetsFilewithSA_PreProcessed.to_csv('./Interim/AUSImportantPeriodTweetsTopicNewwithSA1.csv', index = False) # SAVE THE INTERIM RESULTS

print('ABSA')
tweetsFilewithSA_PreProcessed = pd.read_csv('./Interim/AUSImportantPeriodTweetsTopicNewwithSA1.csv')  # LOAD THE INTERIM RESULTS
print('TOTAL NUMBER OF ROWS: ')
print(len(tweetsFilewithSA_PreProcessed))
ABSADataSet = quantitative_SA.ABSA(tweetsFilewithSA_PreProcessed)
print(ABSADataSet.head(5))
ABSADataSet.to_csv('./Interim/AUSImportantPeriodTweetsTopicNewwithSA2.csv', index = False) # SAVE THE INTERIM RESULTS

print('TPB SCORE')
ABSADataSet = pd.read_csv('./Interim/AUSImportantPeriodTweetsTopicNewwithSA2.csv') # LOAD THE INTERIM RESULTS
print('TOTAL NUMBER OF ROWS: ')
print(len(ABSADataSet))
tpbScoreData = quantitative_SA.tpb_score(ABSADataSet, setLst)
print(tpbScoreData.head(5))
tpbScoreData.to_csv('./AUSImportantPeriodTweetsTopicwithTPBNew.csv', index = False) # SAVE THE RESULTS TO PERFORMING THE REGRESSION