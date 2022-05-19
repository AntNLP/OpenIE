# import os
# import re
# import numpy as np
# import logging
# import torch
# from torch.utils.data import Dataset
# from typing import Tuple, List
# from transformers import BertTokenizer

# logger = logging.getLogger(__name__)

# encmodel = '/home/weidu/OPENIE/wwm-ext'
# tokenizer = BertTokenizer.from_pretrained(encmodel)
# # VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-LOC', 'B-ORG')
# VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'BO', 'IO', 'BS', 'IS', 'BR', 'IR')
# # VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-INF', 'I-INF', 'B-PAT', 'I-PAT', 'B-OPS', 
# #         'I-OPS', 'B-DSE', 'I-DSE', 'B-DRG', 'I-DRG', 'B-LAB', 'I-LAB')
# tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
# idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
# MAX_LEN = 256 - 2
# def set_encmodel(s):
#     encmodel = s
#     tokenizer = BertTokenizer.from_pretrained(encmodel)
    
# class NerDataset(Dataset):
#     def __init__(self, f_path):
#         with open(f_path, 'r', encoding='utf-8') as fr:
#             entries = fr.read().strip().split('~')
#         sents, tags_li = [], [] # list of lists
#         for entry in entries:
#             words = [line.split(' ')[0] for line in entry.strip().split('\n')]
#             tags = [line.split(' ')[-1] for line in entry.strip().split('\n')]
#             if len(words) > MAX_LEN:
#                 # 先对句号分段
#                 word, tag = [], []
#                 for char, t in zip(words, tags):
                    
#                     if char != '~':
#                         if char != '\ue236':   # 测试集中有这个字符
#                             word.append(char)
#                             tag.append(t)
#                     else:
#                         sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
#                         tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
#                         word, tag = [], []            
#                 # 最后的末尾
#                 if len(word):
#                     sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
#                     tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
#                     word, tag = [], []
#             else:
#                 sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
#                 tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
#         self.sents, self.tags_li = sents, tags_li
                

#     def __getitem__(self, idx):
#         words, tags = self.sents[idx], self.tags_li[idx]
#         x, y = [], []
#         is_heads = []
#         seg = []
#         for w, t in zip(words, tags):
#             tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
#             if(w == ''):
#                 tokens = ['<PAD>']
#             if(len(tokens) == 0):
#                 tokens = ['<PAD>']
#             if(t == ''):
#                 t = '<PAD>'
#             xx = tokenizer.convert_tokens_to_ids(tokens)
#             assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}, tokens={tokens}, xx={xx} "

#             # 中文没有英文wordpiece后分成几块的情况
#             # is_head = [1] + [0]*(len(tokens) - 1)
#             # t = [t] + ['<PAD>'] * (len(tokens) - 1)
#             is_head = [1]
#             tt = [t]
#             yy = [tag2idx[each] for each in tt]  # (T,)
#             x.extend(xx)
#             is_heads.extend(is_head)
#             y.extend(yy)
#             if(t == 'BS' or t == 'IS'):
#                 seg.extend([2])
#             elif (t == 'BO' or t == 'IO'):
#                 seg.extend([1])
#             else:
#                 seg.extend([0])

            
#             assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}, xx={(xx)}"

#         # seqlen
#         seqlen = len(y)

#         # to string
#         words = " ".join(words)
#         tags = " ".join(tags)
#         return words, x, is_heads, tags, y, seqlen, seg


#     def __len__(self):
#         return len(self.sents)


# def pad(batch):
#     '''Pads to the longest sample'''
#     f = lambda x: [sample[x] for sample in batch]
#     words = f(0)
#     is_heads = f(2)
#     tags = f(3)
#     seqlens = f(-2)
#     seg = f(-1)
#     maxlen = np.array(seqlens).max()

#     f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
#     x = f(1, maxlen)
#     y = f(-3, maxlen)
#     seg = f(-1, maxlen)


#     f = torch.LongTensor

#     return words, f(x), is_heads, tags, f(y), seqlens, f(seg)



import os
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

# bert_model = 'bert-base-cased'
# tokenizer = BertTokenizer.from_pretrained(bert_model)
# # VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-LOC', 'B-ORG')
# VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'BO', 'IO', 'BS', 'IS', 'BR', 'IR')
# # VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-INF', 'I-INF', 'B-PAT', 'I-PAT', 'B-OPS', 
# #         'I-OPS', 'B-DSE', 'I-DSE', 'B-DRG', 'I-DRG', 'B-LAB', 'I-LAB')
# VOCAB = (   '<PAD>',    '[CLS]',    '[SEP]',    'O',    'A0-B', 
#             'A1-B',     'A2-B',     'A3-B',     'A4-B', 'A5-B',
#             'A0-I',     'A1-I',     'A2-I',     'A3-I', 'A4-I', 
#             'A5-I',     'P-B',      'P-I')
# tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
# idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 256 - 2


class NerDataset(Dataset):
    def __init__(self, f_path, encmodel, VOCAB, tag2idx, idx2tag, gold_tag, seg_tag, task = 'OIE'):
        self.tokenizer = BertTokenizer.from_pretrained(encmodel)
        self.VOCAB = VOCAB
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.gold_tag = gold_tag
        self.seg_tag = seg_tag
        self.encmodel = encmodel
        self.task = task
        logging.info("Use gold tag: {}".format(self.gold_tag))
        logging.info("Use seg tag: {}".format(self.seg_tag))
        logging.info("Model for encode: {}".format(self.encmodel))
        logging.info("Task type is: {}".format(self.task))
        with open(f_path, 'r', encoding='utf-8') as fr:
            entries = fr.read().strip().split('\n\n')
        sents, tags_li, seg_tags_li, ext_tags_li,  p_tags_li = [], [], [], [], [] # list of lists
        for entry in entries:
            words = [line.split('\t')[0] for line in entry.splitlines()]
            tags =  [line.split('\t')[1] for line in entry.splitlines()]
            p_tags = ['O' for line in entry.splitlines()]
            seg_tags = [line.split('\t')[-1] for line in entry.splitlines()]
            l = len(((entry.splitlines()[0]).split('\t')))
            if(seg_tag == True):
                l-=1
            if len(words) > MAX_LEN:
                # # 先对句号分段
                # word, tag, seg_tag = [], [], []
                # for char, t, st in zip(words, tags, seg_tags):
                    
                #     if char != '~':
                #         if char != '\ue236':   # 测试集中有这个字符
                #             word.append(char)
                #             tag.append(t)
                #             seg_tag.append(st)
                #     else:
                #         sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                #         tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                #         seg_tags_li.append(['[CLS]'] + seg_tag[:MAX_LEN] + ['[SEP]'])
                #         word, tag = [], []            
                # # 最后的末尾
                # if len(word):
                #     sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                #     tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                #     seg_tags_li.append(['[CLS]'] + seg_tag[:MAX_LEN] + ['[SEP]'])
                #     word, tag, seg_tag = [], [], []
                logging.info("len(words) > MAX_LEN")
                assert(False)
            else:
                sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
                tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
                seg_tags_li.append(['[CLS]'] + seg_tags[:MAX_LEN] + ['[SEP]']) 
                p_tags_li.append(['[CLS]'] + p_tags[:MAX_LEN] + ['[SEP]'])
                ex_li = []
                for i in range(2, l):
                    ext_tags = [line.split('\t')[i] for line in entry.splitlines()]
                    ex_li.append(['[CLS]'] + ext_tags[:MAX_LEN] + ['[SEP]'])
                ext_tags_li.append(ex_li)
        
        self.sents, self.tags_li, self.seg_tags_li, self.ext_tags_li, self.p_tags_li = sents, tags_li, seg_tags_li, ext_tags_li, p_tags_li
                

    def __getitem__(self, idx):
        words, tags, seg_tags, ext_tags, p_tags = self.sents[idx], self.tags_li[idx], self.seg_tags_li[idx], self.ext_tags_li[idx], self.p_tags_li[idx]
        x, y = [], []
        is_heads = []
        seg = []
        ext = [ [] for i in range(len(ext_tags))]
        if(self.task == 'REL'):
            for i in range(len(tags)):
                if(self.task == 'REL'):
                    if(tags[i][0] == 'A'):
                        tags[i] = 'O'
        j = 0
        for w, t, st in zip(words, tags, seg_tags):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            if(len(tokens) == 0):
                tokens = ['<PAD>']
            xx = self.tokenizer.convert_tokens_to_ids(tokens)
            assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}, tokens={tokens}, xx={xx} "
            if(self.task == 'REL'):
                if(t[0] == 'A'):
                    t = 'O'
            # 中文没有英文wordpiece后分成几块的情况
            # is_head = [1] + [0]*(len(tokens) - 1)
            # tt = [t] + ['<PAD>'] * (len(tokens) - 1)
            # yy = []
            # for each in tt:
            #     if(each in tag2idx):
            #         yy.append(tag2idx[each])
            #     else:
            #         yy.append('<PAD>')
            # x.extend(xx)
            # is_heads.extend(is_head)
            # y.extend(yy)
            # if(self.task == 'SAOKE' or self.task == 'PENN' or self.task == 'WEB' or self.task == ''):
            #     if(t == 'BS' or t == 'IS'):
            #         seg.extend([2] + [0]*(len(tokens) - 1))
            #     elif (t == 'BO' or t == 'IO'):
            #         seg.extend([1] + [0]*(len(tokens) - 1))
            #     else:
            #         seg.extend([0] + [0]*(len(tokens) - 1))
            # if(self.task == 'OIE2016'):
            #     pass
            if(self.gold_tag == True):
                if(t == 'A5-B' or t == 'A5-I'):
                    seg.extend([1])
                    seg.extend([0]*(len(tokens)-1))
                elif(t == 'A4-B' or t == 'A4-I'):
                    seg.extend([1])
                    seg.extend([0]*(len(tokens)-1))
                elif(t == 'A3-B' or t == 'A3-I'):
                    seg.extend([1])
                    seg.extend([0]*(len(tokens)-1))
                elif(t == 'A2-B' or t == 'A2-I'):
                    seg.extend([1])
                    seg.extend([0]*(len(tokens)-1))
                elif(t == 'A1-B' or t == 'A1-I' or t =='O-B' or t == 'O-I'):
                    seg.extend([1])
                    seg.extend([0]*(len(tokens)-1))
                elif(t == 'P-B' or t == 'P-I'):
                    seg.extend([2]*(len(tokens)))
                elif (t == 'A0-B' or t == 'A0-I' or t == 'S-B' or t == 'S-I' or t == 'A-I' or t == 'A-B'):                  
                    seg.extend([1]*(len(tokens)))
                else:
                    seg.extend([0]*(len(tokens)))
            elif(self.seg_tag == True):
                if(st[0] == 'A'):
                    seg.extend([2]*(len(tokens)))
                elif(st[0] == 'P'):
                    seg.extend([1]*(len(tokens)))
                else:
                    seg.extend([0]*(len(tokens)))
            else:
                seg.extend([0]*(len(tokens)))
            # seg.extend([0]*(len(tokens)))
            is_head = [1] + [0]*(len(tokens) - 1)
            # tt = [t] + ['<PAD>'] * (len(tokens) - 1)
            tt = [t] * (len(tokens))
            yy = [self.tag2idx[each] for each in tt]  
            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
            for i in range(len(ext_tags)):
                ext[i].extend([ext_tags[i][j]]*len(tokens))
            j+=1

            
            assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}, xx={(xx)}"
        # seqlen
        seqlen = len(y)

        # to string
        #words = " ".join(words)
        #tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen, seg, p_tags, ext


    def __len__(self):
        return len(self.sents)


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(5)
    seg = f(6)
    p_tags = f(7)
    ext = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(4, maxlen)
    seg = f(6, maxlen)
    for ex in ext:
        for i in range(len(ex)):
            ex[i] = ex[i] + ['O'] * (maxlen - len(ex[i]))
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    



    f = torch.LongTensor
    # for i in range(len(ext)):
    #     ext[i] = f(ext[i])
    return words, f(x), is_heads, tags, f(y), seqlens, f(seg), ext, p_tags