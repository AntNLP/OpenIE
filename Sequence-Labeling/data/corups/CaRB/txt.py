from faulthandler import disable
import pandas as pd
import json
import spacy
dataset_type = 'dev'
fr = '/home/weidu/OPENIE/data/corups/CaRB/' + dataset_type + '.txt'
fw = open('/home/weidu/OPENIE/data/corups/CaRB/carb.' + dataset_type + '.np.txt','w')
# gw = open('/home/weidu/OPENIE/data/corups/oie2016/oie2016' + method + dataset_type + '.gold','w')
dic = {}
mx_len = 0
cnt = 0
nlp = spacy.load("en_core_web_trf", disable=[ "lemmatizer"])
   
with open(fr, 'r') as f:
    entries = f.read().strip().split('\n')
    for entry in entries:
        doc = nlp(entry)
        text = [line for line in entry.split(' ')]
        seg = ['O'] * len(text)
        for chunk in doc.noun_chunks:
            for i in range(chunk.start, chunk.end):
                if(i >= len(seg)):
                    break
                seg[i] = 'A-I'
                seg[chunk.start] = 'A-B'
        
        for i in range(len(text)):
            fw.write(text[i] + '\t' + 'O' + '\t' + seg[i] + '\n')
        fw.write('\n')





