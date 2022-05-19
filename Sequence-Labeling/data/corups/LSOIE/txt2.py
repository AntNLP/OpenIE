from faulthandler import disable
import pandas as pd
import json
import spacy
dataset_type = '_train'
method = '_a+p'
conll_path = '/home/weidu/OPENIE/data/corups/LSOIE/lsoie_wiki' + dataset_type +'.conll'
fw = open('/home/weidu/OPENIE/data/corups/LSOIE/lsoie_wiki' + method + dataset_type + '.txt','w')
# gw = open('/home/weidu/OPENIE/data/corups/LSOIE/lsoie_wiki'+ method + dataset_type + '.gold','w')
# conll_path = '/home/weidu/OPENIE/data/ori/OIE2016/test.oie.conll'
# fw = open('/home/weidu/OPENIE/data/txt/oie2016_test.txt','w')
# conll_path = '/home/weidu/OPENIE/data/ori/OIE2016/train.noisy.oie.conll'
# fw = open('/home/weidu/OPENIE/data/txt/oie2016_train_noisy.txt','w')

nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
# conll_path = '/home/weidu/public/corpus/OIE2016/train.oie.conll'
# fw = open('/home/weidu/OPENIE/data/oie2016.train','w')
# conll_path = '/home/weidu/OPENIE/data/corups/LSOIE/lsoie_wiki_' + dataset_type + '.conll'
# fw = open('/home/weidu/OPENIE/data/corups/oie2016/oie2016.' + dataset_type,'w')
# conll_path = '/home/weidu/public/corpus/OIE2016/dev.oie.conll'
# fw = open('/home/weidu/OPENIE/data/oie2016.dev','w')
cnt = 0
nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
dic = {}
mx = 0
with open(conll_path, 'r') as f:
    entries = f.read().strip().split('\n\n')
    for entry in entries:
        cnt+=1
        text = [line.strip().split('\t')[1] for line in entry.split('\n')]
        arg_tag = []
        pre_tag = []
        for line in entry.split('\n'):
            tag = line.strip().split('\t')[-1]
            if(tag[0] == 'A'):
                if(tag[-1] == 'I'):
                    arg_tag.append('A-I')
                else:
                    arg_tag.append('A-B')
                pre_tag.append('O')
            elif(tag[0] == 'P'):
                if(tag[-1] == 'I'):
                    pre_tag.append('A-I')
                else:
                    pre_tag.append('A-B')
                arg_tag.append('O')
            else:
                arg_tag.append('O')
                pre_tag.append('O')
        print(arg_tag)
        tag = ['O']*len(arg_tag)
        seg_tag = ['O']*len(arg_tag)
        if(method == '_a+p'):
            for i in range(len(arg_tag)):
                tag[i] = arg_tag[i]
                seg_tag[i] = arg_tag[i]
                if(pre_tag[i] != 'O'):
                    tag[i] = pre_tag[i]
                    seg_tag[i] = pre_tag[i]
                
        for i in range(len(tag)):
            fw.write(str(text[i]) + '\t')
            fw.write(str(tag[i]) + '\t' )
            fw.write(str(seg_tag[i]) + '\n')
        fw.write('\n')
