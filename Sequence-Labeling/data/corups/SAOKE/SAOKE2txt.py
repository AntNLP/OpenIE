import json
from re import I

from numpy import float16, true_divide
import os
import numpy as np
import logging
from regex import P
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

bert_model = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_model)
data = open('/home/weidu/OPENIE/data/corups/SAOKE/json/SAOKE_DATA.json')

f1 = open('/home/weidu/OPENIE/data/corups/SAOKE/txt/SAOKE_NER.train','w')
f2 = open('/home/weidu/OPENIE/data/corups/SAOKE/txt/SAOKE_REL.train','w')
cnt = 0
tot = 0
for line in data:  # 获取属性列表
    dic = json.loads(line)
    tot += 1
    if(len(dic['logic']) == 0):
        continue
    # for i in range(len(dic['logic'])):
    #     if( (len(dic['logic'][i]['subject']) ==0 ) or (len(dic['logic'][i]['object']) == 0) or (len(dic['logic'][i]['predicate']) == 0) ):
    #         continue
    #     if( ('|' in dic['logic'][i]['subject'][0]) or ('|' in dic['logic'][i]['object'][0]) or ('|' in dic['logic'][i]['predicate']) ):
    #         continue
    #     if( (dic['logic'][i]['subject'][0] == '_') or (dic['logic'][i]['object'][0] == '_') or (dic['logic'][i]['predicate']) == '_'):
    #         continue
    #     cnt += 1
    if( (len(dic['logic'][0]['subject']) ==0 ) or (len(dic['logic'][0]['object']) == 0) or (len(dic['logic'][0]['predicate']) == 0) ):
        continue
    if( ('|' in dic['logic'][0]['subject']) or ('|' in dic['logic'][0]['object'][0]) or ('|' in dic['logic'][0]['predicate']) ):
        continue
    if( (dic['logic'][0]['subject'][0] == '_') or (dic['logic'][0]['object'][0] == '_') or (dic['logic'][0]['predicate']) == '_'):
        continue
    cnt += 1
print(cnt)
cntt = 0
data.close()
data = open('/home/weidu/OPENIE/data/corups/SAOKE/json/SAOKE_DATA.json')
for line in data:  # 获取属性列表
    dic = json.loads(line)
    if(len(dic['logic']) == 0):
        continue
    if( (len(dic['logic'][0]['subject']) <=0 ) or (len(dic['logic'][0]['object'][0]) == 0) or (len(dic['logic'][0]['predicate']) ==0) ):
        continue
    if( ('|' in dic['logic'][0]['subject']) or ('|' in dic['logic'][0]['object'][0]) or ('|' in dic['logic'][0]['predicate']) ):
        continue
    if( (dic['logic'][0]['subject'] == '_') or (dic['logic'][0]['object'][0] == '_') or (dic['logic'][0]['predicate']) == '_'):
        continue
    cntt += 1
    if(cntt == int(cnt*0.8)):
        print(cntt)
        f1.close()
        f2.close()
        f1 = open('/home/weidu/OPENIE/data/corups/SAOKE/txt/SAOKE_NER.dev','w')
        f2 = open('/home/weidu/OPENIE/data/corups/SAOKE/txt/SAOKE_REL.dev','w')
    if(cntt == int(cnt*0.9)):
        print(cntt)
        f1.close()
        f2.close()
        f1 = open('/home/weidu/OPENIE/data/corups/SAOKE/txt/SAOKE_NER.test','w')
        f2 = open('/home/weidu/OPENIE/data/corups/SAOKE/txt/SAOKE_REL.test','w')
    # f1.write(str(cnt)+'\n')
    s = str(dic['natural'])
    sub = str(dic['logic'][0]['subject'])
    obj = str(dic['logic'][0]['object'][0])
    pre = str(dic['logic'][0]['predicate'])
    tag = ['O' for i in range(len(s))]
    for i in range(len(s)):
        if(s[i] == sub[0] ):
            flag = True
            for j in range(len(sub)):
                if(i+j >= len(s) or s[i+j] != sub[j]):
                    flag = False
                    break
            if (flag):
                tag[i] = 'BS'
                for j in range(1, len(sub)):
                    tag[i+j] = 'IS'
        elif(s[i] == obj[0] ) :
            flag = True
            for j in range(len(obj)):
                if(i+j >= len(s) or s[i+j] != obj[j] ):
                    flag = False
                    break
            if (flag):
                tag[i] = 'BO'
                for j in range(1, len(obj)):
                    tag[i+j] = 'IO'
    for i in range(len(s)):
        f1.write(s[i] + '\t' + tag[i] + '\n')
    f1.write('\n')
    tag = ['O' for i in range(len(s))]
    for i in range(len(s)):
        if(s[i] == sub[0] ):
            flag = True
            for j in range(len(sub)):
                if(i+j >= len(s) or s[i+j] != sub[j]):
                    flag = False
                    break
            if (flag):
                tag[i] = 'BS'
                for j in range(1, len(sub)):
                    tag[i+j] = 'IS'
        elif(s[i] == obj[0] ) :
            flag = True
            for j in range(len(obj)):
                if(i+j >= len(s) or s[i+j] != obj[j]):
                    flag = False
                    break
            if (flag):
                tag[i] = 'BO'
                for j in range(1, len(obj)):
                    tag[i+j] = 'IO'
        elif(s[i] == pre[0] ):
            flag = True
            for j in range(len(pre)):
                if(i+j >= len(s) or s[i+j] != pre[j]):
                    flag = False
                    break
            if (flag):
                tag[i] = 'BR'
                for j in range(1, len(pre)):
                    tag[i+j] = 'IR'
    for i in range(len(s)):
        f2.write(s[i] + '\t' + tag[i] + '\n')
    f2.write('\n')
    

data.close()
f1.close()
f2.close()
# data = open('/home/weidu/OPENIE/NER/data/SAOKE_DATA.json')
# for line in data:  # 获取属性列表
#     print(line)
#     break