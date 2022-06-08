from typing import Text
import spacy
from transformers import AutoTokenizer
nlp = spacy.load("en_core_web_trf")
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
fr_name = './test'
fw_name = '../02-np/test'
type = 'dev'
fw = open(fw_name, 'w')
with open(fr_name, 'r') as f:
    entries = f.read().strip().split('\n\n')
    for entry in entries:
        text = [line.split('\t')[0] for line in entry.split('\n')]
        tokens = []
        heads = []
        for t in text:
            d = nlp(t)
            token = [token for token in d]
            heads.extend([1]+[0]*(len(token)-1))
            tokens.extend(token)
        np_tags = ['O']*len(tokens)
        doc = nlp(" ".join(text))
        for chunk in doc.noun_chunks:
            for i in range(chunk.start, chunk.end):
                np_tags[i] = 'NP'
        
        _np_tags = []
        for np_tag, head in zip(np_tags, heads):
            if head == 1:
                _np_tags.append(np_tag)
        np_tags = _np_tags
        assert(len(np_tags) == len(entry.split('\n')))
        cnt = 0
        for lines in entry.split('\n'):
            if type == 'train':
                fw.write(lines + '\t' + np_tags[cnt] + '\n')
            else:
                fw.write(lines.split('\t')[0] + '\t' + 'O' + '\t' + np_tags[cnt] + '\n')
            cnt += 1
        fw.write('\n')
