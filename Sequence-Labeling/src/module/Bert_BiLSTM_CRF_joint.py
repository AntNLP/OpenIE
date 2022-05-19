from base64 import encode
import numpy as np
# from ctypes.wintypes import tagSIZE
from re import S
from attr import frozen
import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding
from torchsummary import summary
import module.bilstm as bilstm
# from module.crf import CRF
from module.Bert_BiLSTM_CRF import Bert_BiLSTM_CRF

from module.encoder import Encoder
from torchcrf import CRF

class Bert_BiLSTM_CRF_Joint(nn.Module):
    def __init__(self, tag_to_ix, idx2tag, encmodel, hidden_dim=768, seg_num = 7, device = 'cpu', opt = None):
        super(Bert_BiLSTM_CRF_Joint, self).__init__()
        self.device = device
        self.opt = opt
        self.hidden_dim = hidden_dim
        self.seg_num = seg_num
        self.tag_to_ix = tag_to_ix
        self.idx2tag = idx2tag
        self.tagset_size = len(tag_to_ix)

        self.start_label_id = self.tag_to_ix['[CLS]']
        self.end_label_id = self.tag_to_ix['[SEP]']
        self.features_in_hook = []
        self.features_out_hook = []
        self.features = []
        self.encmodel = encmodel

        self.BBC_1 = Bert_BiLSTM_CRF(tag_to_ix = self.tag_to_ix, encmodel = self.encmodel, seg_num = self.seg_num)
        self.BBC_2 = Bert_BiLSTM_CRF(tag_to_ix = self.tag_to_ix, encmodel = self.encmodel, seg_num = self.seg_num)


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def neg_log_likelihood(self, sentence, tags, segg, p_tags, extraction, mask = None):
        loss_1 = self.BBC_1.neg_log_likelihood(sentence, tags, segg)
        with torch.no_grad():
            _, y_hat = self.BBC_1(sentence, segg)
        #  y_hat.to(self.device)
        y_hat = y_hat.numpy().tolist()
        siz = 0
        sent_li = None
        ex_tags_li = None
        seg_li = None
        if(self.opt == 'gold'):
            for y_h, ext, sent, se, in zip(y_hat, extraction, sentence, segg):
                for i in range(len(ext)):
                    seg = [0] * (len(ext[i]))
                    # seg = torch.tensor(seg)
                    ex_tags = [self.tag_to_ix[ex] for ex in ext[i]]
                    seg = [1 if(ex == 'P-I' or ex == 'P-B') else 0 for ex in ext[i]]
                    if(sent_li == None):
                        sent_li = sent
                        sent_li.to(self.device)
                    else:
                        sent_li = torch.cat((sent_li, sent), dim = 0)

                    if(seg_li == None):
                        seg_li = torch.tensor(seg)
                    else:
                        seg_li = torch.cat((seg_li, torch.tensor(seg)), dim = 0)

                    if(ex_tags_li == None):
                        ex_tags_li = torch.tensor(ex_tags)
                    else:
                        ex_tags_li = torch.cat((ex_tags_li, torch.tensor(ex_tags)), dim = 0)
                    siz += 1
        elif(self.opt == 'soft'):
            for y_h, ext, sent, se, in zip(y_hat, extraction, sentence, segg):
                l = -1
                r = -1
                span = []
                for i in range(len(y_h)):
                    if(self.idx2tag[y_h[i]] == 'P-B'):
                        if(l != -1):
                            span.append([l, r])
                        l = i
                        r = i
                    elif(self.idx2tag[y_h[i]] == 'P-I'):
                        r = i
                    else:
                        if(l != -1):
                            span.append([l, r])
                        l = -1
                mx = len(ext)
                cnt = 0
                vis = [False] * len(y_h)
                for i in range(len(span)):
                    seg = [x for x in se]
                    ex_tags = []
                    overlap = False
                    for j in range(len(ext)):
                        for k in range(span[i][0], span[i][1]+1):
                            if(ext[j][k][0] == 'P'):
                                overlap = True
                                break
                        if(overlap):
                            for k in range(span[i][0], span[i][1]+1):
                                seg[k] = 1
                            ex_tags = [self.tag_to_ix[ex] for ex in ext[j]]
                            break
                    if(overlap == False):
                        ex_tags = [self.tag_to_ix['O'] for ex in ext[0]]
                    if(sent_li == None):
                        sent_li = sent
                        sent_li.to(self.device)
                    else:
                        sent_li = torch.cat((sent_li, sent), dim = 0)

                    if(seg_li == None):
                        seg_li = torch.tensor(seg)
                    else:
                        seg_li = torch.cat((seg_li, torch.tensor(seg)), dim = 0)

                    if(ex_tags_li == None):
                        ex_tags_li = torch.tensor(ex_tags)
                    else:
                        ex_tags_li = torch.cat((ex_tags_li, torch.tensor(ex_tags)), dim = 0)
                    cnt += 1
                    siz += 1
                while(cnt < mx):
                    seg = [0] * (len(y_h))
                    # seg = torch.tensor(seg)
                    ex_tags = [self.tag_to_ix[ex] for ex in ext[cnt]]

                    if(sent_li == None):
                        sent_li = sent
                        sent_li.to(self.device)
                    else:
                        sent_li = torch.cat((sent_li, sent), dim = 0)

                    if(seg_li == None):
                        seg_li = torch.tensor(seg)
                    else:
                        seg_li = torch.cat((seg_li, torch.tensor(seg)), dim = 0)

                    if(ex_tags_li == None):
                        ex_tags_li = torch.tensor(ex_tags)
                    else:
                        ex_tags_li = torch.cat((ex_tags_li, torch.tensor(ex_tags)), dim = 0)
                    cnt += 1
                    siz += 1
        else:
            for y_h, ext, sent, se, in zip(y_hat, extraction, sentence, segg):
                l = -1
                r = -1
                span = []
                for i in range(len(y_h)):
                    if(self.idx2tag[y_h[i]] == 'P-B'):
                        if(l != -1):
                            span.append([l, r])
                        l = i
                        r = i
                    elif(self.idx2tag[y_h[i]] == 'P-I'):
                        r = i
                    else:
                        if(l != -1):
                            span.append([l, r])
                        l = -1
                mx = len(ext)
                cnt = 0
                vis = [False] * len(y_h)
                for i in range(len(span)):
                    seg = [x for x in se]
                    flag = True
                    for j in range(span[i][0], span[i][1]+1):
                        if(vis[j] == True):
                            flag = False
                            break
                    if(flag):
                        for j in range(span[i][0], span[i][1]+1):
                            seg[j] = 1
                            vis[j] = True
                    else:
                        continue
                    if(cnt >= mx):
                        ex_tags = [self.tag_to_ix['O'] for ex in ext[0]]
                    else:
                        ex_tags = [self.tag_to_ix[ex] for ex in ext[cnt]]
                    if(sent_li == None):
                        sent_li = sent
                        sent_li.to(self.device)
                    else:
                        sent_li = torch.cat((sent_li, sent), dim = 0)

                    if(seg_li == None):
                        seg_li = torch.tensor(seg)
                    else:
                        seg_li = torch.cat((seg_li, torch.tensor(seg)), dim = 0)

                    if(ex_tags_li == None):
                        ex_tags_li = torch.tensor(ex_tags)
                    else:
                        ex_tags_li = torch.cat((ex_tags_li, torch.tensor(ex_tags)), dim = 0)
                    cnt += 1
                    siz += 1
                while(cnt < mx):
                    seg = [0] * (len(y_h))
                    # seg = torch.tensor(seg)
                    ex_tags = [self.tag_to_ix[ex] for ex in ext[cnt]]

                    if(sent_li == None):
                        sent_li = sent
                        sent_li.to(self.device)
                    else:
                        sent_li = torch.cat((sent_li, sent), dim = 0)

                    if(seg_li == None):
                        seg_li = torch.tensor(seg)
                    else:
                        seg_li = torch.cat((seg_li, torch.tensor(seg)), dim = 0)

                    if(ex_tags_li == None):
                        ex_tags_li = torch.tensor(ex_tags)
                    else:
                        ex_tags_li = torch.cat((ex_tags_li, torch.tensor(ex_tags)), dim = 0)
                    cnt += 1
                    siz += 1
        sent_li = sent_li.reshape(siz, -1)
        seg_li = seg_li.reshape(siz, -1)
        ex_tags_li = ex_tags_li.reshape(siz, -1)
        seg_li = seg_li.to(self.device)
        ex_tags_li = ex_tags_li.to(self.device)
            
        
        loss = loss_1 + self.BBC_2.neg_log_likelihood(sent_li, ex_tags_li, seg_li).to(self.device)
        
        #y_hat = [hat for hat in y_hat]
        #preds = [self.idx2tag[hat] for hat in y_hat]
        # _, y_hat = self.BBC_1.forward(sentence, seg)  # y_hat: (N, T)
        #print(preds)
        # return loss
        return loss



    def forward(self, sentence, seg, mask = None):  # dont confuse this with _forward_alg above.
        _, y_hat = self.BBC_1(sentence, seg)
        #  y_hat.to(self.device)
        y_hat = y_hat.numpy().tolist()
        Y = []
        siz = 0
        each_siz = []
        sent_li = None
        ex_tags_li = None
        seg_li = None
        for y_h,  sent in zip(y_hat, sentence):
            l = -1
            r = -1
            span = []
            for i in range(len(y_h)):
                if(self.idx2tag[y_h[i]] == 'P-B'):
                    if(l != -1):
                        span.append([l, r])
                    l = i
                    r = i
                elif(self.idx2tag[y_h[i]] == 'P-I'):
                    r = i
                else:
                    if(l != -1):
                        span.append([l, r])
                    l = -1
            
            vis = [False] * len(y_h)
            #mx_span = min(6,len(span))
            cnt = 0
            for i in range(len(span)):
                seg = [0] * (len(y_h))
                flag = True
                for j in range(span[i][0], span[i][1]+1):
                    if(vis[j] == True):
                        flag = False
                        break
                if(flag == False):
                    continue
                for j in range(span[i][0], span[i][1]+1):
                    seg[j] = 1
                # seg = torch.tensor(seg)
               

                if(sent_li == None):
                    sent_li = sent
                    sent_li.to(self.device)
                else:
                    sent_li = torch.cat((sent_li, sent), dim = 0)

                if(seg_li == None):
                    seg_li = torch.tensor(seg)
                else:
                    seg_li = torch.cat((seg_li, torch.tensor(seg)), dim = 0)
                cnt += 1
                siz += 1
            each_siz.append(cnt)
            #print(seg_li)
        if(siz == 0) :
            Y.append([]*len(y_hat))
        else:
            sent_li = sent_li.reshape(siz, -1)
            seg_li = seg_li.reshape(siz, -1)
            seg_li = seg_li.to(self.device)
            sent_li = sent_li.to(self.device)
            _, y =  self.BBC_2.forward(sent_li, seg_li)  # y_hat: (N, T)
            y = y.numpy().tolist()
            tot = 0
            for esiz in each_siz:
                tmp = []
                for k in range(esiz):
                    tmp.append(y[tot])
                    tot += 1
                Y.append(tmp)
        #y_hat = [hat for hat in y_hat]
        #preds = [self.idx2tag[hat] for hat in y_hat]
        # _, y_hat = self.BBC_1.forward(sentence, seg)  # y_hat: (N, T)
        #print(preds)
        return _, Y

