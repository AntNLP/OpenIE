from faulthandler import disable
import pandas as pd
import json
import spacy
dataset_type = 'train'
method = 'joint'
# method = '.debug'
# method = ''
conll_path = '/home/weidu/OPENIE/data/corups/LSOIE/lsoie_wiki_'+ dataset_type + '.conll'
fw = open('/home/weidu/OPENIE/data/corups/LSOIE/lsoie_wiki.'+ dataset_type,'w')
# gw = open('/home/weidu/OPENIE/data/corups/oie2016/oie2016' + method + dataset_type + '.gold','w')
# nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
dic = {}
mx_len = 0
tot = 0
with open(conll_path, 'r') as f:
    entries = f.read().strip().split('\n\n')
    for entry in entries:
        if(method == 'joint' or method == '.debug'):
            text = " ".join([line.strip().split('\t')[1] for line in entry.split('\n')])
            if(not (text in dic)) :
                dic[text] = {
                    'text': [line.strip().split('\t')[1] for line in entry.split('\n')],
                    'pre_tag': ['O' for line in entry.split('\n')],
                    'extractions': []
                }
            arg_tag = []
            pre_tag = []
            cnt = 0
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
                        pre_tag.append('P-I')
                        dic[text]['pre_tag'][cnt] = 'P-I'
                    else:
                        pre_tag.append('P-B')
                        dic[text]['pre_tag'][cnt] = 'P-B'
                    arg_tag.append('O')
                else:
                    arg_tag.append('O')
                    pre_tag.append('O')
                cnt+=1
            dic[text]['extractions'].append(
                {
                    'arg_tag': arg_tag, 
                    'pre_tag': pre_tag
                }
            ) 
            
        else:
            tag = [line.strip().split('\t')[-1] for line in entry.split('\n')]
            text = [line.strip().split('\t')[1] for line in entry.split('\n')]
            for i in range(len(tag)):
                fw.write(str(text[i]) + '\t')
                if(tag[i][0] == 'A'):
                    if(tag[i][-1] == 'B'):
                        tag[i] = 'A-B'
                    else:
                        tag[i] = 'A-I'
                if(tag[i][0] == 'P'):
                    if(tag[i][-1] == 'B'):
                        tag[i] = 'P-B'
                    else:
                        tag[i] = 'P-I'
                fw.write(str(tag[i]) + '\t' )
                fw.write(str('O') + '\n')
            fw.write('\n')
    if(method == '.debug'):
        print('hello')
        
        for k, v in dic.items():
            tot += 1
            if(tot == 4):
                exit()
                # train/dev/test data
            text = v['text']
            tag = v['pre_tag']
            mx_len = max(mx_len, len(text))
            for i in range(len(tag)):
                fw.write(text[i] + '\t' + 'O' + '\t' + 'O' + '\n')
            fw.write('\n')
    else:
        for k, v in dic.items():
            # train/dev/test data
            text = v['text']
            tag = v['pre_tag']
            ext = []
            for extraction in v['extractions']:
                spans = []
                tags = extraction['arg_tag']
                p_tags = extraction['pre_tag']
                for i in range(len(tags)):
                    if(p_tags[i] != 'O'):
                        tags[i] = p_tags[i]
                ext.append(tags)
            # spacy or sth
            mx_len = max(mx_len, len(text))
            for i in range(len(tag)):
                fw.write(text[i] + '\t' + tag[i])
                for j in range(len(ext)):
                    fw.write('\t' + ext[j][i])
                fw.write('\n')
            fw.write('\n')
print(mx_len)





