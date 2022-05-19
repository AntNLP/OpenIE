from faulthandler import disable
import pandas as pd
import json
import spacy
dataset_type = 'train'
fr = 'oie2016.' + dataset_type
fw = open('oie2016.gold.' + dataset_type, 'w')
dic = {}
mx_len = 0
tot = 0
with open(fr, 'r') as f:
    entries = f.read().strip().split('\n\n')
    for entry in entries:
        lines = entry.split('\n')
        for line in lines:
            fw.write(line + '\t' + line.split('\t')[1] + '\n')
        fw.write('\n')
print(mx_len)





