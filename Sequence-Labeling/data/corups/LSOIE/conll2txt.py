from faulthandler import disable
import pandas as pd
import json
import spacy
dataset_type = '_test'
# method = '_joint'
method = ''
conll_path = '/home/weidu/OPENIE/data/corups/LSOIE/lsoie_wiki' + dataset_type +'.conll'
fw = open('/home/weidu/OPENIE/data/corups/LSOIE/lsoie_wiki' + method + dataset_type + '.txt','w')
gw = open('/home/weidu/OPENIE/data/corups/LSOIE/lsoie_wiki' + method + dataset_type + '.gold','w')
nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
dic = {}
mx_len = 0

with open(conll_path, 'r') as f:
    entries = f.read().strip().split('\n\n')
    for entry in entries:
        
        
        # for i in range(len(tag)):
        #     fw.write(str(text[i]) + '\t')
        #     fw.write(str(tag[i]) + '\t' )
        #     fw.write(str(seg_tag[i]) + '\n')
        # fw.write('\n')
        if(method == '_joint'):
            text = " ".join([line.strip().split('\t')[1] for line in entry.split('\n')])
            if(not (text in dic)) :
                dic[text] = {
                    'text': [line.strip().split('\t')[1] for line in entry.split('\n')],
                    'extractions': []
                }
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
                        pre_tag.append('P-I')
                    else:
                        pre_tag.append('P-B')
                    arg_tag.append('O')
                else:
                    arg_tag.append('O')
                    pre_tag.append('O')
            dic[text]['extractions'].append(
                {
                    'arg_tag': arg_tag, 
                    'pre_tag': pre_tag
                }
            ) 
            for k, v in dic.items():
                
                texts = v['text']
                # gold data
                for extraction in v['extractions']:
                    gw.write(k)
                    span = []
                    text = []
                    tags = extraction['pre_tag']
                    for i in range(len(tags)):
                        if(tags[i] != 'O'):
                            span.append(i)
                            text.append(texts[i])
                    gw.write('\t' + str((' '.join(text), span)))
                    span = []
                    text = []
                    tags = extraction['arg_tag']
                    for i in range(len(tags)):
                        if(tags[i] == 'A-B'):
                            if(len(span) != 0):
                                gw.write('\t' + str((' '.join(text), span)))
                                span = [] 
                                text = []
                            span.append(i)
                            text.append(texts[i])
                        elif(tags[i] == 'A-I'):
                            span.append(i)
                            text.append(texts[i])
                    gw.write('\t' + str((' '.join(text), span)))
                    gw.write('\n')
                
                # train/dev/test data
                texts = v['text']
                tag = ['O'] * len(texts)
                
                for extraction in v['extractions']:
                    
                    spans = []
                    tags = extraction['arg_tag']
                    span = []
                    for i in range(len(tags)):
                        if(tags[i] == 'A-B'):
                            if(len(span) != 0):
                                spans.append(span)
                            span = [i]
                        elif(tags[i] == 'A-I'):
                            span.append(i)
                    if(len(span) != 0):
                        spans.append(span)
                    span = None
                    for span in spans:
                        flag = True
                        for i in span:
                            if(tag[i] != 'O'):
                                flag = False
                        if(flag):
                            for i in span:
                                tag[i] = 'A-I'
                            tag[span[0]] = 'A-B'

                    spans = []
                    tags = extraction['pre_tag']
                    span = None
                    for i in range(len(tags)):
                        if(tags[i] == 'P-B'):
                            if(span != None):
                                spans.append(span)
                            span = [i]
                        elif(tags[i] == 'P-I'):
                            span.append(i)
                    if(len(span) != 0):
                        spans.append(span)
                    for span in spans:
                        flag = True
                        for i in span:
                            if(tag[i] != 'O'):
                                flag = False
                        if(flag):
                            for i in span:
                                tag[i] = 'P-I'
                            tag[span[0]] = 'P-B'
                # spacy or sth
                mx_len = max(mx_len, len(texts))
                for t, tag in zip(texts, tag):
                    if(tag[0] == 'A'):
                        fw.write(t + '\t' + tag + '\t' + 'A' + '\n')
                    else:
                        fw.write(t + '\t' + tag + '\t' + 'O' + '\n')
                fw.write('\n')
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
print(mx_len)





