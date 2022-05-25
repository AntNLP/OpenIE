import logging

import torch
import torch.nn as nn
from utils import vocabs
from eval.oie_eval.oie_readers.goldReader import GoldReader
from modules.encoder import Encoder
from modules.decoder import Decoder

logging.basicConfig(level = logging.INFO)
class Pipeline(nn.Module):
    def __init__(self, tag2idx, idx2tag, cfg, device):
        super(Pipeline, self).__init__()
        self.device = device
        self.hidden_dim = cfg.D_MODEL
        self.seg_num = cfg.SEG_NUM
        self.tag2idx = tag2idx
        self.tagset_size = len(idx2tag)
        self.cfg = cfg
        self.encoder = Encoder( frozen = False,
                                seg_num = self.seg_num,
                                hidden_dim = self.hidden_dim,
                                tagset_size = self.tagset_size,
                                cfg = self.cfg)
        self.decoder = Decoder( hidden_dim=self.hidden_dim,
                                tagset_size = self.tagset_size,
                                cfg = self.cfg)
        
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def neg_log_likelihood(self, sentence, tags, seg):
        feats = self.encoder(sentence, seg)
        loss = self.decoder.loss(feats, tags = tags)
        return loss

    def forward(self, sentence, seg):  
        feats = self.encoder(sentence, seg)
        score, tag_seq = self.decoder(feats)
        return score, torch.tensor(tag_seq, dtype=torch.long)

class Joint(nn.Module):
    def __init__(self, tag2idx, idx2tag, cfg, device = 'cpu'):
        super(Joint, self).__init__()
        self.device = device
        self.opt = cfg.PREDICATE_FOR_LEARNING_ARGUMENT
        self.hidden_dim = cfg.D_MODEL
        self.seg_num = cfg.SEG_NUM
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.tagset_size = len(tag2idx)
        self.arg_model = Pipeline(tag2idx, idx2tag, cfg, device)
        self.pre_model = Pipeline(tag2idx, idx2tag, cfg, device)


    def neg_log_likelihood(self, sentence, tags, segg, p_tags, extraction, mask = None):
        loss_predicate = self.pre_model.neg_log_likelihood(sentence, tags, segg)
        with torch.no_grad():
            _, y_hat = self.pre_model(sentence, segg)
        y_hat = y_hat.numpy().tolist()
        siz = 0
        sent_list = torch.ones(len(sentence[0])).unsqueeze(0).to(self.device)
        extag_list = torch.ones(len(sentence[0])).unsqueeze(0).to(self.device)
        seg_list = torch.ones(len(sentence[0])).unsqueeze(0).to(self.device)

        if(self.opt == 'gold'):
            for y_h, ext, sent, se, in zip(y_hat, extraction, sentence, segg):
                for i in range(len(ext)):
                    seg = [0] * (len(ext[i]))
                    extag = [self.tag2idx[ex] for ex in ext[i]]
                    seg = [1 if(ex == 'P-I' or ex == 'P-B') else 0 for ex in ext[i]]
                    sent_list = torch.cat((sent_list, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    seg_list = torch.cat((seg_list, torch.tensor(seg).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    extag_list = torch.cat((extag_list, torch.tensor(extag).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    siz += 1
        elif(self.opt == 'soft'):
            # sent_list, sent_list, sent_list = soft_extraction()
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
                    extag = []
                    overlap = False
                    for j in range(len(ext)):
                        for k in range(span[i][0], span[i][1]+1):
                            if(ext[j][k][0] == 'P'):
                                overlap = True
                                break
                        if(overlap):
                            for k in range(span[i][0], span[i][1]+1):
                                seg[k] = 1
                            extag = [self.tag2idx[ex] for ex in ext[j]]
                            break
                    if(overlap == False):
                        extag = [self.tag2idx['O'] for ex in ext[0]]
                    sent_list = torch.cat((sent_list, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    seg_list = torch.cat((seg_list, torch.tensor(seg).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    extag_list = torch.cat((extag_list, torch.tensor(extag).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    cnt += 1
                    siz += 1
                while(cnt < mx):
                    seg = [0] * (len(y_h))
                    # seg = torch.tensor(seg)
                    extag = [self.tag2idx[ex] for ex in ext[cnt]]
                    sent_list = torch.cat((sent_list, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    seg_list = torch.cat((seg_list, torch.tensor(seg).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    extag_list = torch.cat((extag_list, torch.tensor(extag).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
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
                        extag = [self.tag2idx['O'] for ex in ext[0]]
                    else:
                        extag = [self.tag2idx[ex] for ex in ext[cnt]]
                    sent_list = torch.cat((sent_list, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    seg_list = torch.cat((seg_list, torch.tensor(seg).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    extag_list = torch.cat((extag_list, torch.tensor(extag).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    cnt += 1
                    siz += 1
                while(cnt < mx):
                    seg = [0] * (len(y_h))
                    # seg = torch.tensor(seg)
                    extag = [self.tag2idx[ex] for ex in ext[cnt]]
                    sent_list = torch.cat((sent_list, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    seg_list = torch.cat((seg_list, torch.tensor(seg).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    extag_list = torch.cat((extag_list, torch.tensor(extag).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    cnt += 1
                    siz += 1

        sent_list = sent_list[1:].long()
        seg_list = seg_list[1:].long()
        extag_list = extag_list[1:].long()
        loss = loss_predicate + self.arg_model.neg_log_likelihood(sent_list, extag_list, seg_list).to(self.device)
        return loss

    def forward(self, sentence, seg, mask = None):  
        _, y_hat = self.pre_model(sentence, seg)
        y_hat = y_hat.numpy().tolist()
        Y = []
        siz = 0
        each_siz = []
        sent_list = torch.ones(len(sentence[0])).unsqueeze(0).to(self.device)
        seg_list = torch.ones(len(sentence[0])).unsqueeze(0).to(self.device)
        for y_h, sent in zip(y_hat, sentence):
            l = -1
            r = -1
            span = []
            for i in range(len(y_h)):
                if self.idx2tag[y_h[i]] == 'P-B' :
                    if l != -1 :
                        span.append([l, r])
                    l = r = i
                elif self.idx2tag[y_h[i]] == 'P-I':
                    r = i
                else:
                    if l != -1 :
                        span.append([l, r])
                    l = -1
            vis = [False] * len(y_h)
            cnt = 0
            for i in range(len(span)):
                seg = [0] * (len(y_h))
                flag = True
                for j in range(span[i][0], span[i][1]+1):
                    if(vis[j] == True):
                        flag = False
                        break
                if flag == False:
                    continue
                for j in range(span[i][0], span[i][1]+1):
                    seg[j] = 1
                sent_list = torch.cat((sent_list, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                seg_list = torch.cat((seg_list, torch.tensor(seg).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                cnt += 1
                siz += 1
            each_siz.append(cnt)
        if(siz == 0) :
            Y.append([]*len(y_hat))
        else:
            sent_list = sent_list[1:].long()
            seg_list = seg_list[1:].long()
            _, y =  self.arg_model(sent_list, seg_list)  # y_hat: (N, T)
            y = y.numpy().tolist()
            tot = 0
            for esiz in each_siz:
                tmp = []
                for k in range(esiz):
                    tmp.append(y[tot])
                    tot += 1
                Y.append(tmp)
        return _, Y

class SeqIE():
    """a SeqIE model"""
    def __init__(self, cfg, device):
        """init a SeqIE model"""
        self.vocab = vocabs.get_vocab(cfg.DOMAIN)
        tag2idx = dict(zip(self.vocab, range(len(self.vocab))))
        idx2tag = dict(zip(range(len(self.vocab)), self.vocab))
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.device = device
        self.cfg = cfg
    def get_model(self):
        """chose a SeqIE"""
        if self.cfg.METHOD == 'joint' :
            model = Joint(self.tag2idx, self.idx2tag, self.cfg, self.device)
            return model
        if self.cfg.METHOD == 'pipeline':
            model_pre = Pipeline(self.tag2idx, self.idx2tag, self.cfg, self.device)
            model_arg = Pipeline(self.tag2idx, self.idx2tag, self.cfg, self.device)
            return model_pre, model_arg
        return None
    def get_model_info(self):
        """get model information"""
        print(self.cfg)
