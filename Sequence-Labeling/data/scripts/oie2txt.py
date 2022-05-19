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
    dataset_type = 'dev'
    # method = 'np'
    method = 'gold'
    
    # conll_path = '/home/weidu/public/corpus/OIE2016/train.oie.conll'
    # fw = open('/home/weidu/OPENIE/data/oie2016.train','w')
    # conll_path = '/home/weidu/OPENIE/data/corups/oie2016/'+ dataset_type + '.oie.conll'
    # fw = open('/home/weidu/OPENIE/data/corups/oie2016/oie2016_' + method + '.' + dataset_type,'w')
    conll_path = '/home/weidu/OPENIE/data/corups/oie2016/'+ dataset_type + '.oie.conll'
    fw = open('/home/weidu/OPENIE/data/corups/oie2016/oie2016_' + method + '.' + dataset_type,'w')
    # conll_path = '/home/weidu/public/corpus/OIE2016/dev.oie.conll'
    # fw = open('/home/weidu/OPENIE/data/oie2016.dev','w')
    cnt = 0
    if(method == 'np'):
        # sent = 'QVC Network Inc. said it completed its acquisition of CVN Cos. for about $ 423 million .'
        nlp = spacy.load("en_core_web_trf")
        # doc = nlp(sent)
        # for chunk in doc.noun_chunks:
        #     print(chunk.start, chunk.end, chunk.label_)
        with open(conll_path, 'r') as f:
            entries = f.read().strip().split('\n\n')
            for entry in entries:
                cnt += 1
                text = " ".join([line.strip().split('\t')[1] for line in entry.split('\n')]);
                # seg_tag = ['O'] * len(entry.split('\n'))
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
                if( method == 'np'):
                    for ent in doc.ents:
                        for i in range(ent.start, ent.end):
                            seg_tag[i] = 'A'
                for i in range(len(tag)):
                    fw.write(str(text[i]) + '\t')
                    fw.write(str(tag[i]) + '\t' )
                    fw.write(str(seg_tag[i]) + '\n')
                fw.write('\n')
    else:
        nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    

