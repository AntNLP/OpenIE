from faulthandler import disable
import pandas as pd
import json
import spacy
dataset_type = 'dev'
method = '.pipeline'
conll_path = '/home/weidu/OPENIE/data/corups/oie2016/' + dataset_type +'.oie.conll'
fw = open('/home/weidu/OPENIE/data/corups/oie2016/oie2016.new' + method + '.'+ dataset_type + '.txt','w')
# gw = open('/home/weidu/OPENIE/data/corups/oie2016/oie2016' + method + dataset_type + '.gold','w')
nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
dic = {}
mx_len = 0
tot = 0
with open(conll_path, 'r') as f:
    entries = f.read().strip().split('\n\n')
    for entry in entries:
        tag = [line.strip().split('\t')[-1] for line in entry.split('\n')]
        text = [line.strip().split('\t')[1] for line in entry.split('\n')]
        for i in range(len(tag)):
            # if(tag[i][0] == 'A'):
            #     if(tag[i][-1] == 'B'):
            #         tag[i] = 'A-B'
            #     else:
            #         tag[i] = 'A-I'
            if(tag[i][0] == 'A' ):
                if(tag[i][1] == '0' or tag[i][1] == '1' or tag[i][1] == '2' or tag[i][1] == '3'):
                    tag[i] = tag[i]
                else:
                    tag[i] = 'O'
            if(tag[i][0] == 'P'):
                if(tag[i][-1] == 'B'):
                    tag[i] = 'P-B'
                else:
                    tag[i] = 'P-I'
            fw.write(str(text[i]) + '\t' + str(tag[i]) + '\t' + 'O' + '\n')
        fw.write('\n')
print(mx_len)





