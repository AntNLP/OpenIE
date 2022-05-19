
from faulthandler import disable
from statistics import median
import pandas as pd
import json
import spacy
# conll_path = '/home/weidu/OPENIE/data/ori/OIE2016/train.oie.conll'
# fw = open('/home/weidu/OPENIE/data/txt/oie2016_train.txt','w')
# conll_path = '/home/weidu/OPENIE/data/ori/OIE2016/test.oie.conll'
# fw = open('/home/weidu/OPENIE/data/txt/oie2016_test.txt','w')
# conll_path = '/home/weidu/OPENIE/data/ori/OIE2016/train.noisy.oie.conll'
# fw = open('/home/weidu/OPENIE/data/txt/oie2016_train_noisy.txt','w')
task = 'OIE2016'
cnt = 0
if(task in ['NYT', 'WEB', 'PENN']):
    fname = task.lower() 
    frpath = '/home/weidu/OPENIE/data/ori/' + task + '/' + fname + '.oie'
    fwpath = '/home/weidu/OPENIE/data/json/' + '/' + fname + '.json'
    print(frpath)
    fr = open(frpath, 'r')
    fw = open(fwpath, 'w')
    new_json = []
    entries = fr.read().strip().split('\n')
    for entry in entries:
        lines = entry.split('\t')
        new_dict = {}
        new_dict['text'] = lines[0]
        new_dict['predicate'] = lines[1]
        new_dict['subject'] = lines[2]
        new_dict['object'] = lines[3]
        new_json.append(new_dict)
    json.dump(new_json,fw)
    fr.close()
    fw.close()
elif (task == 'OIE2016'):
    dataset_type = '_dev'
    method = '_np+g'
    
    # conll_path = '/home/weidu/public/corpus/OIE2016/train.oie.conll'
    # fw = open('/home/weidu/OPENIE/data/oie2016.train','w')
    conll_path = '/home/weidu/OPENIE/data/corups/oie2016/dev' + '.oie.conll'
    fw = open('/home/weidu/OPENIE/data/corups/oie2016/oie2016' + method + dataset_type + '.txt' ,'w')
    # conll_path = '/home/weidu/public/corpus/OIE2016/dev.oie.conll'
    # fw = open('/home/weidu/OPENIE/data/oie2016.dev','w')
    cnt = 0
    if(method == '_np'):
        # 提取np 并且 每个句子只给三个extraction
        # sent = 'QVC Network Inc. said it completed its acquisition of CVN Cos. for about $ 423 million .'
        nlp = spacy.load("en_core_web_trf")
        # doc = nlp(sent)
        # for chunk in doc.noun_chunks:
        #     print(chunk.start, chunk.end, chunk.label_)
    elif(method == '_np+g'):
        # np + gold pre\
        r = 4
        method = method + '_' + str(r)
        conll_path = '/home/weidu/OPENIE/data/corups/oie2016/dev' + '.oie.conll'
        fw = open('/home/weidu/OPENIE/data/corups/oie2016/oie2016' + method + dataset_type + '.txt' ,'w')
        nlp = spacy.load("en_core_web_trf", disable=[ "lemmatizer", 'ner'])
        dic = {}
        with open(conll_path, 'r') as f:
            entries = f.read().strip().split('\n\n')
            for entry in entries:
                cnt += 1
                text = " ".join([line.strip().split('\t')[1] for line in entry.split('\n')]);
                if(text in dic):
                    dic[text] += 1
                else:
                    dic[text] = 0
                if(dic[text] >= r):
                    continue
                seg_tag = ['O'] * len(entry.split('\n'))
                
                doc = nlp(text)
                tag = []
                text = []
                i = 0
                for line in entry.split('\n'):
                    l = line.strip().split('\t')
                    sentpic = [x for x in nlp.tokenizer(l[1])]
                    text.extend(sentpic)
                    tag.extend([l[-1]]*len(sentpic))
                seg_tag = ['O'] * len(tag)
                for chunk in doc.noun_chunks:
                    for i in range(chunk.start, chunk.end):
                        seg_tag[i] = 'A'
                for i in range(len(tag)):
                    fw.write(str(text[i]) + '\t')
                    if(tag[i] != 'O'):
                        tag[i] = tag[i][0] + '-' + tag[i][-1]
                    fw.write(str(tag[i]) + '\t' )
                    if(tag[i][0] == 'P'):
                        seg_tag[i] = 'P'
                    fw.write(str(seg_tag[i]) + '\n')
                fw.write('\n')

