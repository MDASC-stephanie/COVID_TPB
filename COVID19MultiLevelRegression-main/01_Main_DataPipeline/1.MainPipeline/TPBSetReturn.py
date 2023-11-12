from nltk.stem import SnowballStemmer
import pandas as pd

def dataSet():
    tpb = pd.read_excel('./dataInput/full_phrases2.xlsx',  sheet_name='final', usecols="C:D")

    snow_stemmer = SnowballStemmer(language='english')
    #if it's a single word, then stem; else keep the phrase as it is
    # REMOVE THE CONDITION ABOUT KEEPING THE PHRASE AS IT IS
    # tpb['keywords'] = tpb['full_phrases'].apply(lambda x:  snow_stemmer.stem(x) if " " not in x else x)
    tpb['full_phrases_'] = tpb['full_phrases'].apply(lambda x: x.split(' '))
    tpb['keywords'] = ''
    for index, row in tpb.iterrows():
        tmpList = list()
        for eachElement in row['full_phrases_']:
            tmpList.append(snow_stemmer.stem(eachElement))
        tpb.at[index, 'keywords'] = ' '.join(tmpList)

    #create sets of keywords
    sn_set = set(tpb['keywords'][tpb['Type'] == 'SN'].values)
    pbc_set = set(tpb['keywords'][tpb['Type'] == 'PBC'].values)
    o_set = set(tpb['keywords'][tpb['Type'] == 'O'].values)

    return sn_set, pbc_set, o_set