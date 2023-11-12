import pandas as pd
import os
import json

def jsonColumnSelector(dataRead, dataWrite):
    with open(dataWrite, 'a') as the_file:
        with open(dataRead) as infile:
            for line in infile:
                elementSelect = dict()
                # print(line)
                elementRead = json.loads(line)
                # print(elementRead)
                columnList = ['created_at', 'id_str', 'full_text', 'lang', 'retweet_count', 'favorite_count']
                for eachColumn in columnList:
                    elementSelect[eachColumn] = elementRead[eachColumn]
                elementWrite = json.dumps(elementSelect)
                # print(elementWrite)
                the_file.write(elementWrite)
                the_file.write('\n')
    return 0

dataRead = '/AustraliaTweetID.jsonl' # INPUT PATH
dataWrite = '/AustraliaTweetID_.jsonl' # OUTPUT PATH
jsonColumnSelector(dataRead, dataWrite)