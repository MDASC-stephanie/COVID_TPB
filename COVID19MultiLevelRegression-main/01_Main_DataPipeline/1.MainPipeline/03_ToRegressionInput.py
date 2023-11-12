
# STEP 03
#
# OBJECTIVE
#  - THIS .PY IS FOR MERGING THE TWEETS FOR DIFFERENT COUNTRIES WITH TPB SCORES TO THE REGRESSION INPUT
#
# MAIN INPUT
#  - TWEETS FOR DIFFERENT COUNTRIES WITH TPB SCORES (INPUT IN SEPARATED BUT AT ONCE)
#  - SELECTED DATE SCOPE
#  - POLICY IMPLEMENTATION INFORAMTION
# 
# MAIN OUTPUT
#  - REGRESSION MODEL INPUT (OUTPUT AS A WHOLE)

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

print('LOAD THE DATASET')
UStweetsFileDFwithSA = pd.read_csv('./USImportantPeriodTweetsTopicwithTPB.csv')
UStweetsFileDFwithSA = common.alignDataFrame(UStweetsFileDFwithSA)

UKtweetsFileDFwithSA = pd.read_csv('./UKImportantPeriodTweetsTopicwithTPB.csv')
UKtweetsFileDFwithSA = common.alignDataFrame(UKtweetsFileDFwithSA)

AUStweetsFileDFwithSA = pd.read_csv('./AUSImportantPeriodTweetsTopicwithTPBNew.csv')
AUStweetsFileDFwithSA = common.alignDataFrame(AUStweetsFileDFwithSA)

CANtweetsFileDFwithSA = pd.read_csv('./CANImportantPeriodTweetsTopicwithTPB.csv')
CANtweetsFileDFwithSA = common.alignDataFrame(CANtweetsFileDFwithSA)

tweetsFileDFwithSA = pd.concat([UStweetsFileDFwithSA, UKtweetsFileDFwithSA, AUStweetsFileDFwithSA, CANtweetsFileDFwithSA])
tweetsFileDFwithSA = tweetsFileDFwithSA.dropna(axis = 0, how = 'any')

print('==================================================')
print('PREPARE THE INFORMATION FOR THE REGRESSION ANALYSIS SESSION')
print('==================================================')
# CONVERT THE DATAFRAME TO REGRESSION MODEL ACCEPT FORMAT
print('START PREPARING THE INPUT FOR THE REGRESSION MODEL')
print('PREPARE THE MASTER DATE TABLES')
masterDate = pd.read_csv('./masterDate.csv')
countryList = list()
dateList = list()
dateGroupList = list()
for x in range(len(masterDate)):
    # READ THE CONFIGURATION TABLE
    country = masterDate['Country'].iloc[x]
    startDate = common.stringToDate(masterDate['StartDate'].iloc[x])
    endDate = common.stringToDate(masterDate['EndDate'].iloc[x])
    timeGroup = masterDate['TimeGroup'].iloc[x]
    # APPEND THE INFORMATION INTO TABLE
    countryList.append(country)
    dateList.append(startDate)
    dateGroupList.append(timeGroup)
    initialDate = startDate
    while initialDate + timedelta(days=1) <= endDate:
        countryList.append(country)
        dateList.append(initialDate + timedelta(days=1))
        dateGroupList.append(timeGroup)
        initialDate = initialDate + timedelta(days=1)

dateMaster = pd.DataFrame(data = {'country': countryList, 'date': dateList, 'timeGroup': dateGroupList})
dateMaster.to_csv('./dataMaster.csv', index = False)

print('CONVERT THE RESULT')
dateMaster = pd.read_csv('./dataMaster.csv')

##########################################################################################
# USERS CAN CHOOSE PREPARAING THE REGRESSION MODLE INPUT WITH TWO APPROACHES
#   - AVERAGE THE TWEETS PER DAY AND THEN OVER THE PERIOD: resultToRegression
#   - AVERAGE THE TWEETNS DIRECTLY OVER THE PERIOD: resultToRegressionAveragingTweets
##########################################################################################

regressionSentimentInput = common.resultToRegressionAveragingTweets(tweetsFileDFwithSA, dateMaster, ignoreZero = False) # AVERAGING BY TWEETS
# regressionSentimentInput = common.resultToRegressionAveragingTweets(tweetsFileDFwithSA, dateMaster, ignoreZero = True) # AVERAGING BY TWEETS WITH IGNORING THE ZEROS
# regressionSentimentInput = common.resultToRegression(tweetsFileDFwithSA, ignoreZero = False) # AVERAGING BY DAY
# regressionSentimentInput = common.resultToRegression(tweetsFileDFwithSA, ignoreZero = True) # AVERAGING BY DAY WITH IGNORING THE ZEROS
try:
    regressionSentimentInput['tweetDate'] = regressionSentimentInput['tweetDate'].apply(common.stringToDate2)
except:
    None
regressionSentimentInput.to_csv('./regressionSentimentInput.csv', index = False)

print('READ THE DATA')
if ('timeGroup' in list(regressionSentimentInput.columns)):
    regressionSentimentInput = regressionSentimentInput.drop(['timeGroup'], axis = 1)
print('PREPARE THE ADDITIONAL INFORMATION')
policyImplementationbyDay = common.readPolicyImplementation()

print('COMBINE THE FULL SET OF INFORMATION')
regressionInput = common.combinedForRegression(dateMaster, regressionSentimentInput, policyImplementationbyDay)
regressionInput.to_csv('./regressionInput.csv', index = False)