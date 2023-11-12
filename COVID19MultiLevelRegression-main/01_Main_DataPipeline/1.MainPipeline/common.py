import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
# pd.set_option('display.width', 1000)

def resultToRegression(df, ignoreZero = False):
    df = df[pd.to_numeric(df['sentiment'], errors='coerce').notnull()] # REMOVE NON-NUMERIC ROWS
    df['sentiment_'] = df['sentiment'].apply(lambda x: float(x))
    df['tweetDate'] = df['tweetTimeStamp'].apply(lambda x: str(x)[:10])
    df = df[df['key'].isin(['United States', 'United Kingdom', 'Canada', 'Australia'])]

    # AGGREGATE THE INFORMATION FOR SA
    regressionDataframe = df.groupby(['key', 'tweetDate', 'policyLabel_pred'])['sentiment_'].mean().reset_index()
    # PIVOT THE INFORMATION
    regressionDataframe = regressionDataframe.pivot(index = ['key', 'tweetDate'], columns = 'policyLabel_pred')['sentiment_'].reset_index()
    regressionDataframe = regressionDataframe.rename(columns = {'key': 'country'})
    for eachColumn in list(regressionDataframe.columns):
        if eachColumn != 'country' and eachColumn != 'tweetDate':
            regressionDataframe = regressionDataframe.rename(columns = {eachColumn:eachColumn + '_Sentiment'})

    # AGGREGATE THE INFORMATION FOR SN, A, PBC AND O
    if ignoreZero == True:
        df['SN_'] = df['SN'].apply(lambda x: float(x) if float(x) != 0 else np.NaN)
        df['A_'] = df['A'].apply(lambda x: float(x) if float(x) != 0 else np.NaN)
        df['PBC_'] = df['PBC'].apply(lambda x: float(x) if float(x) != 0 else np.NaN)
        df['O_'] = df['O'].apply(lambda x: float(x) if float(x) != 0 else np.NaN)
    else:
        df['SN_'] = df['SN'].apply(lambda x: float(x))
        df['A_'] = df['A'].apply(lambda x: float(x))
        df['PBC_'] = df['PBC'].apply(lambda x: float(x))
        df['O_'] = df['O'].apply(lambda x: float(x))

    # FOR GENERAL SN, A, PBC AND O
    regressionDataframeSupplement = df.groupby(['key', 'tweetDate'])[['SN_', 'A_', 'PBC_', 'O_']].mean().reset_index()
    regressionDataframeSupplement = regressionDataframeSupplement.rename(columns={'key':'country', 'SN_':'SN', 'A_':'A', 'PBC_':'PBC', 'O_':'O'})

    # AGGREGATE THE INFORMATION FOR SN
    regressionDataframeSN = df.groupby(['key', 'tweetDate', 'policyLabel_pred'])['SN_'].mean().reset_index()
    # PIVOT THE INFORMATION
    regressionDataframeSN = regressionDataframeSN.pivot(index=['key', 'tweetDate'], columns='policyLabel_pred')['SN_'].reset_index()
    regressionDataframeSN = regressionDataframeSN.rename(columns={'key': 'country'})
    for eachColumn in list(regressionDataframeSN.columns):
        if eachColumn != 'country' and eachColumn != 'tweetDate':
            regressionDataframeSN = regressionDataframeSN.rename(columns={eachColumn: eachColumn + '_SN'})

    # AGGREGATE THE INFORMATION FOR A
    regressionDataframeA = df.groupby(['key', 'tweetDate', 'policyLabel_pred'])['A_'].mean().reset_index()
    # PIVOT THE INFORMATION
    regressionDataframeA = regressionDataframeA.pivot(index=['key', 'tweetDate'], columns='policyLabel_pred')['A_'].reset_index()
    regressionDataframeA = regressionDataframeA.rename(columns={'key': 'country'})
    for eachColumn in list(regressionDataframeA.columns):
        if eachColumn != 'country' and eachColumn != 'tweetDate':
            regressionDataframeA = regressionDataframeA.rename(columns={eachColumn: eachColumn + '_A'})

    # AGGREGATE THE INFORMATION FOR PBC
    regressionDataframePBC = df.groupby(['key', 'tweetDate', 'policyLabel_pred'])['PBC_'].mean().reset_index()
    # PIVOT THE INFORMATION
    regressionDataframePBC = regressionDataframePBC.pivot(index=['key', 'tweetDate'], columns='policyLabel_pred')['PBC_'].reset_index()
    regressionDataframePBC = regressionDataframePBC.rename(columns={'key': 'country'})
    for eachColumn in list(regressionDataframePBC.columns):
        if eachColumn != 'country' and eachColumn != 'tweetDate':
            regressionDataframePBC = regressionDataframePBC.rename(columns={eachColumn: eachColumn + '_PBC'})

    # AGGREGATE THE INFORMATION FOR O
    regressionDataframeO = df.groupby(['key', 'tweetDate', 'policyLabel_pred'])['O_'].mean().reset_index()
    # PIVOT THE INFORMATION
    regressionDataframeO = regressionDataframeO.pivot(index=['key', 'tweetDate'], columns='policyLabel_pred')['O_'].reset_index()
    regressionDataframeO = regressionDataframeO.rename(columns={'key': 'country'})
    for eachColumn in list(regressionDataframeO.columns):
        if eachColumn != 'country' and eachColumn != 'tweetDate':
            regressionDataframeO = regressionDataframeO.rename(columns={eachColumn: eachColumn + '_O'})

    ###
    dataFrameReturn = regressionDataframe.merge(regressionDataframeSupplement, on = ['country', 'tweetDate'], how = 'inner')
    dataFrameReturn = dataFrameReturn.merge(regressionDataframeSN, on = ['country', 'tweetDate'], how = 'inner')
    dataFrameReturn = dataFrameReturn.merge(regressionDataframeA, on = ['country', 'tweetDate'], how = 'inner')
    dataFrameReturn = dataFrameReturn.merge(regressionDataframePBC, on = ['country', 'tweetDate'], how = 'inner')
    dataFrameReturn = dataFrameReturn.merge(regressionDataframeO, on = ['country', 'tweetDate'], how = 'inner')

    return dataFrameReturn

def resultToRegressionAveragingTweets(df, dateMaster, ignoreZero = False):
    dateMaster['dateString'] = dateMaster['date'].apply(lambda x: str(x)[:10].replace('-', ''))
    df['tweetDate'] = df['tweetTimeStamp'].apply(lambda x: str(x)[:10].replace('-', ''))

    # COMBINED THE INFORMATION OF DATE GROUP
    df = df.merge(dateMaster, left_on = ['country', 'tweetDate'], right_on = ['country', 'dateString'], how = 'left')

    ###
    df = df[pd.to_numeric(df['sentiment'], errors='coerce').notnull()] # REMOVE NON-NUMERIC ROWS
    df['sentiment_'] = df['sentiment'].apply(lambda x: float(x))
    df['tweetDate'] = df['tweetTimeStamp'].apply(lambda x: str(x)[:10])
    df = df[df['key'].isin(['United States', 'United Kingdom', 'Canada', 'Australia'])]

    # AGGREGATE THE INFORMATION FOR SA
    regressionDataframe = df.groupby(['key', 'timeGroup', 'policyLabel_pred'])['sentiment_'].mean().reset_index()
    # PIVOT THE INFORMATION
    regressionDataframe = regressionDataframe.pivot(index = ['key', 'timeGroup'], columns = 'policyLabel_pred')['sentiment_'].reset_index()
    regressionDataframe = regressionDataframe.rename(columns = {'key': 'country'})
    for eachColumn in list(regressionDataframe.columns):
        if eachColumn != 'country' and eachColumn != 'timeGroup':
            regressionDataframe = regressionDataframe.rename(columns={eachColumn: eachColumn + '_Sentiment'})

    # AGGREGATE THE GENERAL INFORMATION FOR SN, A, PBC AND O
    if ignoreZero == True:
        df['SN_'] = df['SN'].apply(lambda x: float(x) if float(x) != 0 else np.NaN)
        df['A_'] = df['A'].apply(lambda x: float(x) if float(x) != 0 else np.NaN)
        df['PBC_'] = df['PBC'].apply(lambda x: float(x) if float(x) != 0 else np.NaN)
        df['O_'] = df['O'].apply(lambda x: float(x) if float(x) != 0 else np.NaN)
    else:
        df['SN_'] = df['SN'].apply(lambda x: float(x))
        df['A_'] = df['A'].apply(lambda x: float(x))
        df['PBC_'] = df['PBC'].apply(lambda x: float(x))
        df['O_'] = df['O'].apply(lambda x: float(x))

    regressionDataframeSupplement = df.groupby(['key', 'timeGroup'])[['SN_', 'A_', 'PBC_', 'O_']].mean().reset_index()
    regressionDataframeSupplement = regressionDataframeSupplement.rename(columns={'key':'country', 'SN_':'SN', 'A_':'A', 'PBC_':'PBC', 'O_':'O'})

    # AGGREGATE THE INFORMATION FOR SN
    regressionDataframeSN = df.groupby(['key', 'timeGroup', 'policyLabel_pred'])['SN_'].mean().reset_index()
    # PIVOT THE INFORMATION
    regressionDataframeSN = regressionDataframeSN.pivot(index=['key', 'timeGroup'], columns='policyLabel_pred')['SN_'].reset_index()
    regressionDataframeSN = regressionDataframeSN.rename(columns={'key': 'country'})
    for eachColumn in list(regressionDataframeSN.columns):
        if eachColumn != 'country' and eachColumn != 'timeGroup':
            regressionDataframeSN = regressionDataframeSN.rename(columns={eachColumn: eachColumn + '_SN'})

    # AGGREGATE THE INFORMATION FOR A
    regressionDataframeA = df.groupby(['key', 'timeGroup', 'policyLabel_pred'])['A_'].mean().reset_index()
    # PIVOT THE INFORMATION
    regressionDataframeA = regressionDataframeA.pivot(index=['key', 'timeGroup'], columns='policyLabel_pred')['A_'].reset_index()
    regressionDataframeA = regressionDataframeA.rename(columns={'key': 'country'})
    for eachColumn in list(regressionDataframeA.columns):
        if eachColumn != 'country' and eachColumn != 'timeGroup':
            regressionDataframeA = regressionDataframeA.rename(columns={eachColumn: eachColumn + '_A'})

    # AGGREGATE THE INFORMATION FOR PBC
    regressionDataframePBC = df.groupby(['key', 'timeGroup', 'policyLabel_pred'])['PBC_'].mean().reset_index()
    # PIVOT THE INFORMATION
    regressionDataframePBC = regressionDataframePBC.pivot(index=['key', 'timeGroup'], columns='policyLabel_pred')['PBC_'].reset_index()
    regressionDataframePBC = regressionDataframePBC.rename(columns={'key': 'country'})
    for eachColumn in list(regressionDataframePBC.columns):
        if eachColumn != 'country' and eachColumn != 'timeGroup':
            regressionDataframePBC = regressionDataframePBC.rename(columns={eachColumn: eachColumn + '_PBC'})

    # AGGREGATE THE INFORMATION FOR O
    regressionDataframeO = df.groupby(['key', 'timeGroup', 'policyLabel_pred'])['O_'].mean().reset_index()
    # PIVOT THE INFORMATION
    regressionDataframeO = regressionDataframeO.pivot(index=['key', 'timeGroup'], columns='policyLabel_pred')['O_'].reset_index()
    regressionDataframeO = regressionDataframeO.rename(columns={'key': 'country'})
    for eachColumn in list(regressionDataframeO.columns):
        if eachColumn != 'country' and eachColumn != 'timeGroup':
            regressionDataframeO = regressionDataframeO.rename(columns={eachColumn: eachColumn + '_O'})

    ###
    dataFrameReturn = regressionDataframe.merge(regressionDataframeSupplement, on = ['country', 'timeGroup'], how = 'inner')
    dataFrameReturn = dataFrameReturn.merge(regressionDataframeSN, on = ['country', 'timeGroup'], how = 'inner')
    dataFrameReturn = dataFrameReturn.merge(regressionDataframeA, on = ['country', 'timeGroup'], how = 'inner')
    dataFrameReturn = dataFrameReturn.merge(regressionDataframePBC, on = ['country', 'timeGroup'], how = 'inner')
    dataFrameReturn = dataFrameReturn.merge(regressionDataframeO, on = ['country', 'timeGroup'], how = 'inner')

    dataFrameReturn = dataFrameReturn.merge(dateMaster, on = ['country', 'timeGroup'], how = 'inner')
    dataFrameReturn = dataFrameReturn.drop(['dateString'], axis = 1)
    dataFrameReturn = dataFrameReturn.rename(columns={'date':'tweetDate'})

    return dataFrameReturn

def readPolicyImplementation():
    columnRenameDict = {'Period of interest': 'timePeriod', 'I3': 'policy1', 'I4': 'policy2', 'I8': 'policy3', 'I9': 'policy4', 'I11': 'policy5', 'I14': 'policy6', 'I20': 'policy7'}
    policyImplementationUSA = pd.read_excel('./dataInput/policyImplementation/policyImplementation.xlsx', sheet_name = 'USA').rename(columns = columnRenameDict)
    policyImplementationUSA = policyImplementationDataFrame(policyImplementationUSA, 'United States')
    policyImplementationUK = pd.read_excel('./dataInput/policyImplementation/policyImplementation.xlsx', sheet_name = 'UK').rename(columns=columnRenameDict)
    policyImplementationUK = policyImplementationDataFrame(policyImplementationUK, 'United Kingdom')
    policyImplementationCAN = pd.read_excel('./dataInput/policyImplementation/policyImplementation.xlsx', sheet_name = 'Can').rename(columns=columnRenameDict)
    policyImplementationCAN = policyImplementationDataFrame(policyImplementationCAN, 'Canada')
    policyImplementationAUS = pd.read_excel('./dataInput/policyImplementation/policyImplementation.xlsx', sheet_name = 'AUS').rename(columns=columnRenameDict)
    policyImplementationAUS = policyImplementationDataFrame(policyImplementationAUS, 'Australia')
    policyImplementation = pd.concat([policyImplementationUSA, policyImplementationUK, policyImplementationCAN, policyImplementationAUS])
    return policyImplementation

def policyImplementationDataFrame(df, country):
    df['timePeriodStart'] = df['timePeriod'].apply(lambda x: x.split('~')[0].strip())
    df['timePeriodStart'] = df['timePeriodStart'].apply(stringToDate2)
    df['timePeriodEnd'] = df['timePeriod'].apply(lambda x: x.split('~')[1].strip())
    df['timePeriodEnd'] = df['timePeriodEnd'].apply(stringToDate2)
    df = df.drop(['timePeriod'], axis=1)

    countryList = list()
    dateList = list()
    policy1List = list()
    policy2List = list()
    policy3List = list()
    policy4List = list()
    policy5List = list()
    policy6List = list()
    policy7List = list()
    firstDevList = list()
    secondDevList = list()
    for x in range(len(df)):
        # READ THE CONFIGURATION TABLE
        country = country
        startDate = df['timePeriodStart'].iloc[x]
        endDate = df['timePeriodEnd'].iloc[x]
        policy1 = float(df['policy1'].iloc[x])
        policy2 = float(df['policy2'].iloc[x])
        policy3 = float(df['policy3'].iloc[x])
        policy4 = float(df['policy4'].iloc[x])
        policy5 = float(df['policy5'].iloc[x])
        policy6 = float(df['policy6'].iloc[x])
        policy7 = float(df['policy7'].iloc[x])
        firstDev = float(df['Number of case (First deri)'].iloc[x])
        secondDev = float(df['Number of case (second deri)'].iloc[x])
        # APPEND THE INFORMATION INTO TABLE
        countryList.append(country)
        dateList.append(startDate)
        policy1List.append(policy1)
        policy2List.append(policy2)
        policy3List.append(policy3)
        policy4List.append(policy4)
        policy5List.append(policy5)
        policy6List.append(policy6)
        policy7List.append(policy7)
        firstDevList.append(firstDev)
        secondDevList.append(secondDev)
        initialDate = startDate
        while initialDate + timedelta(days=1) <= endDate:
            countryList.append(country)
            dateList.append(initialDate + timedelta(days=1))
            policy1List.append(policy1)
            policy2List.append(policy2)
            policy3List.append(policy3)
            policy4List.append(policy4)
            policy5List.append(policy5)
            policy6List.append(policy6)
            policy7List.append(policy7)
            firstDevList.append(firstDev)
            secondDevList.append(secondDev)

            initialDate = initialDate + timedelta(days=1)

    dfProcessed = pd.DataFrame(data={'country': countryList, 'date': dateList, 'Government Policies on Pandemic Prevention (Government Policies)_I': policy1List, 'Biological Environment_I': policy2List, 'Industrial and Economic Stability Indicator (Economy barriers)_I': policy3List, 'Industrial and economic stability indicator (Economy support)_I': policy4List, 'Multimedia and Technology Infrastructure_I': policy5List, 'Government Policies on Pandemic Prevention (Individual Behaviour)_I': policy6List, 'Pandemic Education_I': policy7List, 'firstDev': firstDevList, 'secondDev': secondDevList})
    return dfProcessed

def combinedForRegression(dateMaster, regressionSentimentInput, policyImplementationbyDay):
    dateMaster['dateString'] = dateMaster['date'].apply(lambda x: x.replace('-', ''))
    regressionSentimentInput['tweetDate'] = regressionSentimentInput['tweetDate'].apply(lambda x: x.replace('-', ''))
    policyImplementationbyDay['date'] = policyImplementationbyDay['date'].apply(lambda x: str(x).replace('-', '')[:8])
    regressionInput = dateMaster.merge(regressionSentimentInput, left_on = ['country', 'dateString'], right_on = ['country', 'tweetDate'], how = 'left')
    regressionInput = regressionInput.merge(policyImplementationbyDay, left_on = ['country', 'dateString'], right_on = ['country', 'date'], how = 'left')
    regressionInput = regressionInput.drop(['dateString', 'date_y', 'tweetDate'], axis = 1)
    regressionInput = regressionInput.rename(columns={'date_x': 'date'})

    listOfFloatColumns = list()
    for eachColumn in list(regressionInput.columns):
        if eachColumn != 'country' and eachColumn != 'date' and eachColumn != 'timeGroup':
            regressionInput[eachColumn] = regressionInput[eachColumn].apply(lambda x: float(x))
            listOfFloatColumns.append(eachColumn)

    regressionInputAggregated = regressionInput.groupby(['country', 'timeGroup'])[listOfFloatColumns].mean().reset_index()
    regressionInputMin = regressionInput.groupby(['country', 'timeGroup'])['date'].min().reset_index()
    regressionInputMin = regressionInputMin.rename(columns={'date': 'startDate'})
    regressionInputMax = regressionInput.groupby(['country', 'timeGroup'])['date'].max().reset_index()
    regressionInputMax = regressionInputMax.rename(columns={'date': 'endDate'})

    regressionInputAggregated = regressionInputAggregated.merge(regressionInputMin, on = ['country', 'timeGroup'], how='inner')
    regressionInputAggregated = regressionInputAggregated.merge(regressionInputMax, on=['country', 'timeGroup'], how='inner')

    regressionInputAggregated = regressionInputAggregated.drop(['timeGroup'], axis = 1)

    return regressionInputAggregated

def dateMonthZeroPadding(date, sep):
    dateList = date.split(sep)
    if len(dateList[1]) == 1:
        dateList[1] = '0' + dateList[1]
    dateProcessed = sep.join(dateList)
    return dateProcessed

def stringToDate(date):
    date = dateMonthZeroPadding(date, '/')
    return datetime.strptime(date, '%d/%m/%Y')

def stringToDate2(date):
    date = dateMonthZeroPadding(date, '-')
    return datetime.strptime(date, '%Y-%m-%d')

def alignDataFrame(data):
    data = data[['country', 'text', 'processedText', 'tweetTimeStamp', 'policyLabel_pred', 'policyPredictionScore', 'key', 'sentiment', 'SA', 'SN', 'A', 'PBC', 'O']]
    data['tweetTimeStamp'] = data['tweetTimeStamp'].apply(lambda x: str(x)[:19])
    return data