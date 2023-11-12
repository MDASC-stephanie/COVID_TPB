# %% Sentiment analysis

# import required packages
import pandas as pd
from textblob import TextBlob


# ======================================================  basic Sentiment Analaysis ====================================================== #
# preprocessing for sa, data after processing for policy classification
def sa_pre(data):
    # creating country key, e.g. United States0, United States1,... will all become 'United States'
    data['key'] = data['country'].str[:-1]

    # only keep those tweets where country of the user is in our study's scope
    scope = ['United States', 'United Kingdom', 'Canada', 'Australia']
    data = data[data['key'].isin(scope)]

    return data


# using TextBlob for sentiment scoring, should run sa_pre() function first
def sa_textblob(data):
    # sentiment score
    data['sentiment'] = data['text'].apply(
        lambda t: TextBlob(t).sentiment.polarity
    )

    # sentiment label (negative/neutral/positive) based on sentiment score column
    data['SA'] = 'neu'
    data.loc[data['sentiment'] > 0, 'SA'] = 'pos'
    data.loc[data['sentiment'] < 0, 'SA'] = 'neg'

    return data


# ======================================================  ABSA ====================================================== #
# data preprocessing steps before conducting absa and tpb scoring
# tweet policy is the dataframe. This assume basic sentiment analysis (using sa_textblob function) has already been completed
def pre_absa_tpb(tweet_policy):
    # cleaning tweet_policying tweets before ABSA
    from nltk.tokenize import sent_tokenize
    from nltk.stem import SnowballStemmer
    import re

    snow_stemmer = SnowballStemmer(language='english')

    tweet_policy['clean_text'] = tweet_policy['text'].astype(str).str.lower()  # standardize into lower case
    # remove URLs and special characters from the result
    tweet_policy['clean_text'] = tweet_policy['clean_text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '',
                                                                                                        regex=True)
    tweet_policy['clean_text'] = tweet_policy['clean_text'].replace(r'@\S+', '', regex=True)
    tweet_policy['clean_text'] = tweet_policy['clean_text'].replace(r'\n', '. ', regex=True)  # remove new lines

    tweet_policy['clean_text'] = tweet_policy['clean_text'].apply(
        sent_tokenize)  # tokenize into sentences (before remove we special characters)
    tweet_policy['clean_text'] = tweet_policy['clean_text'].map(
        lambda x: [re.sub(r'[^a-zA-Z\s]', '', s) for s in x])  # remove special characters
    tweet_policy['clean_text'] = tweet_policy['clean_text'].map(
        lambda x: [re.sub(r' +', ' ', s).strip() for s in x])  # remove duplicated and trailing white space

    # create and initialize some new columns
    tweet_policy['aspect'], tweet_policy['description'], tweet_policy['ABSA'], tweet_policy['SN'], tweet_policy['A'], \
    tweet_policy['PBC'], tweet_policy['O'] = ["", "", "", 0.0, 0.0, 0.0, 0.0]

    # setting value of the 'attitude' variable, which is based on sentiment score
    tweet_policy['A'] = tweet_policy['sentiment']  # set A value = sentiment score initially
    tweet_policy.loc[tweet_policy[
                         'sentiment'] < 0, 'A'] = 0  # if sentiment is negative, this attitude will contribute to 0 behaviour intention

    return tweet_policy


# absa function using ner with entity linking for additional info when doing tpb scoring
# ABSA function: given clean tweets, return ABSA score, the aspects, and description of the aspect
def ABSA(df):
    from textblob import TextBlob
    import re
    import spacy

    # initialize language model
    # nlp = spacy.load("en_core_web_trf")
    nlp = spacy.load("en_core_web_sm")

    # add pipeline
    # for the first time using entityLinker, run these in terminal first to install
    # pip install spacy-entity-linker
    # python -m spacy_entity_linker "download_knowledge_base"
    nlp.add_pipe("entityLinker", last=True)

    # ABSA steps
    # entities we're interested in
    entity = {'PERSON', 'NORP', 'ORG', 'PRODUCT', 'EVENT', 'LAW'}

    df = df.astype({'aspect': 'object', 'description': 'object', 'ABSA': 'object'})
    for index, row in df.iterrows():
        # PRINT THE NUMBER OF ROW PROCESSED FOR MONITORING THE STATUS
        if index% 1000 == 0:
            print('NUMBER OF ROW PROCESSED: ' + str(index))

        # only do ABSA if there're sentiment expressed in the tweet
        if row.sentiment != 0:
            tweet = row.clean_text  # tweet
            aspect_in = list()  # the aspectn name, corresponding absa score, and description  to be added to the df
            absa_in = list()
            describe_in = list()

            tweet = stringToList(tweet)
            tweet = set(tweet)
            tweet = list(tweet)

            for sents in tweet:  # for each sentence in the tweet
                labels = []  # for storing entity label detected from the tweet's sentences
                try:
                    sents = re.sub(r' +', ' ', sents).strip()  # remove any remaining duplicated white spaces
                    doc = nlp(sents)
                    labels = [word.label_ for word in doc.ents]  # labels (e.g. 'ORG', 'NORP') detected in the sentence
                except:
                    pass

                if entity.intersection(labels):  # check if the labels detected are of interest
                    # the candidates aspects detected (would only include in final aspect list if there're wikidata details)
                    candidates = [[word.text, word.label_] for word in doc.ents if word.label_ in entity]

                    for item in candidates:
                        interest_check = nlp(str(item[0]))
                        try:  # check superclass (wikidata) of the candidate aspect
                            wikidata = [enti._.linkedEntities[0] for enti in interest_check.sents][0]
                            describe = str(wikidata.get_super_entities()[0])  # superclass of the aspect
                        except:
                            wikidata = None
                        if wikidata != None:  # sentiment analysis for aspects that have wikidata info
                            # sentiment score of that aspect's corresponding sentence
                            score = TextBlob(sents).sentiment.polarity
                            if score == 0:  # if insufficient info from sentence, then analyze the whole tweet
                                score = TextBlob(" ".join(tweet)).sentiment.polarity

                            aspect_in.append(item[0])  # append the aspect name into the "in" list
                            describe_in.append(describe)
                            absa_in.append(score)

            # once finish processing a tweet, update that row's df ABSA col related value using those 'in' list
            df.at[index, 'aspect'] = aspect_in
            df.at[index, 'description'] = describe_in
            df.at[index, 'ABSA'] = absa_in

    return df


# ======================================================  TPB ====================================================== #
# returns tpb scoring based on absa and sa results. Assume pre_absa_tpb() and ABSA() functions were already executed
# df is the dataframe
# set_lst is the keyword sets of each tpb variables; i.e.:
#    set_lst = [sn_set, pbc_set, o_set]
def tpb_score(df, set_lst):
    from statistics import mean

    df = df.fillna(value={'aspect': '[]', 'description': '[]', 'ABSA': '[]'})

    for index, row in df.iterrows():
        # PRINT THE NUMBER OF ROW PROCESSED FOR MONITORING THE STATUS
        if index% 1000 == 0:
            print('NUMBER OF ROW PROCESSED: ' + str(index))

        row['aspect'] = stringToList1(row['aspect'], stringOnly = True)
        row['description'] = stringToList1(row['description'], stringOnly = True)
        row['ABSA'] = stringToList1(row['ABSA'])
        row['clean_text'] = stringToList1(row['clean_text'], stringOnly = True)

        # only do tpb scoring if there's sentiment expressed in the tweet, otherwise the BI will be nan
        if row.sentiment != 0:
            # for positive sentiment tweets
            if row.sentiment > 0:
                tweet = " ".join(row.clean_text)  # rejoin cleaned sentence back into a complete tweet
                tweet = tokenizeStemmer(tweet)
                # sn scoring, see if the SN keywords & phrases appeared in tokens/tweet; same for pbc and o
                for i, col in enumerate(['SN', 'PBC', 'O']):
                    for eachElement in set_lst[i]:
                        if tweet.lower().find(eachElement.lower()) != -1:  # IF THE ELEMENT IS FOUND IN THE TWEET
                            df.at[index, col] = row.sentiment
                            break  # BREAK IF THE ELEMENT HAS ALREADY BE FOUND

            # if there're absa result, overwrite the general result
            if len(row.aspect) > 0:
                rating_lst = [[], [], []]  # for storing each aspect's TPB variable value
                for num, aspect in enumerate(row.aspect):
                    ABSAScore = row.ABSA[num]
                    currentAspect = tokenizeStemmer(aspect)
                    currentDescription = tokenizeStemmer(row.description[num])
                    for i, col in enumerate(['SN', 'PBC', 'O']):
                        for eachElement in set_lst[i]:
                            if currentAspect.lower() == eachElement.lower() or currentDescription.lower().find(eachElement.lower()) != -1:
                                rating_lst[i].append(ABSAScore)

                # set TPB variable value after iterating through the aspects
                for i, col in enumerate(['SN', 'PBC', 'O']):
                    if len(rating_lst[i]) > 0:
                        df.at[index, col] = mean(rating_lst[i])

    return df

def stringToList(string, stringOnly = False):
    stringList = string.replace('[', '').replace(']', '').split(',')
    string = list()
    for eachElement in stringList:
        eachElement = eachElement.strip().replace("'", '')
        if len(eachElement) > 0:
            if stringOnly == True:
                string.append(eachElement)
            else:
                try:
                    string.append(float(eachElement))
                except:
                    string.append(eachElement)
    return string

def tokenizeStemmer(string):
    from nltk.tokenize import word_tokenize
    from nltk.stem import SnowballStemmer

    snow_stemmer = SnowballStemmer(language='english')
    stringList = word_tokenize(string)
    stringListProcessed = list()

    for eachElement in stringList:
        stringListProcessed.append(snow_stemmer.stem(eachElement))

    string = ' '.join(stringListProcessed)

    return string

def stringToList1(string, stringOnly = False):
    stringList = string.replace('[', '').replace(']', '').split(',')
    string = list()
    for eachElement in stringList:
        eachElement = eachElement.strip().replace("'", '')
        if stringOnly == True:
            string.append(eachElement)
        else:
            try:
                string.append(float(eachElement))
            except:
                string.append(eachElement)
    return string