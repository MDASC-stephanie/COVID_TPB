# %% Sentiment analysis
#import required packages
import pandas as pd
from textblob import TextBlob

#preprocessing for sa, data after processing for policy classification
def sa_pre(data):
    #creating country key, e.g. United States0, United States1,... will all become 'United States'
    data['key'] = data['country'] #.str[:-1]
    #only keep those tweets where country of the user is in our study's scope
    scope = ['United States', 'United Kingdom' , 'Canada', 'Australia']
    data = data[data['key'].isin(scope)]
    return data

#using TextBlob for sentiment scoring
def sa_textblob(data):
    #sentiment score
    data['sentiment'] = data['text'].apply(lambda t: TextBlob(t).sentiment.polarity)

    #sentiment label (negative/neutral/positive) based on sentiment score column
    data['SA'] = 'neu'
    data.loc[data['sentiment'] > 0, 'SA'] = 'pos'
    data.loc[data['sentiment'] < 0, 'SA'] = 'neg'
    return data

# %%