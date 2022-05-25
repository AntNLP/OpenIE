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
        self.cfg = cfg
        self.arg_model = Pipeline(tag2idx, idx2tag, cfg, device)
        self.pre_model = Pipeline(tag2idx, idx2tag, cfg, device)

    def get_predicate_span(self, tags_list):
        """Function for obtaining the predicating span."""
        spans = []
        for tags in tags_list:
            span = []
            for idx, tag in enumerate(tags):
                if tag == 'P-B':
                    if len(span) != 0:
                        spans.append(span)
                    span = [idx]
                elif tag == 'P-I':
                    span.append(idx)
            if len(span) != 0:
                spans.append(span)
        return spans

    def neg_log_likelihood(self, sents, _tags_list, _segs_list, _ext_tags_list, mask = None):
        loss_predicate = self.pre_model.neg_log_likelihood(sents, _tags_list, _segs_list)
        with torch.no_grad():
            _, y_hats_list = self.pre_model(sents, _segs_list)
        y_hats_list = y_hats_list.numpy().tolist()
        siz = 0
        sents_tensor = torch.ones(len(sents[0])).unsqueeze(0).to(self.device)
        ext_tags_tensor = torch.ones(len(sents[0])).unsqueeze(0).to(self.device)
        seg_tags_tensor = torch.ones(len(sents[0])).unsqueeze(0).to(self.device)
        for y_hats, ext_tags_list, sent, in zip(y_hats_list, _ext_tags_list, sents):
            preds = [self.idx2tag[y_hat] for y_hat in y_hats]
            pre_spans = self.get_predicate_span([preds])
            gold_spans = self.get_predicate_span(ext_tags_list)
            if self.cfg.PREDICATE_FOR_LEARNING_ARGUMENT == 'gold':
                for ext_tags in ext_tags_list:
                    extag = [self.tag2idx[ex] for ex in ext_tags]
                    seg_tags = [1 if ex == 'P-I' or ex == 'P-B'  else 0 for ex in ext_tags]
                    sents_tensor = torch.cat((sents_tensor, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    seg_tags_tensor = torch.cat((seg_tags_tensor, torch.tensor(seg_tags).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    ext_tags_tensor = torch.cat((ext_tags_tensor, torch.tensor(extag).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    siz += 1
            elif self.cfg.PREDICATE_FOR_LEARNING_ARGUMENT == 'soft':
                mx = len(gold_spans)
                for gold_span, ext_tags in zip(gold_spans, ext_tags_list):
                    flag = False
                    for pre_span in pre_spans:
                        if flag:
                            break
                        for idx in pre_span:
                            if idx in gold_span:
                                flag = True
                                seg_tags = [1 if ex=='P-I' or ex=='P-B' else 0 for ex in ext_tags]
                                print(seg_tags)
                                extag = [self.tag2idx[ex] for ex in ext_tags]
                                sents_tensor = torch.cat((sents_tensor, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                                seg_tags_tensor = torch.cat((seg_tags_tensor, torch.tensor(seg_tags).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                                ext_tags_tensor = torch.cat((ext_tags_tensor, torch.tensor(extag).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                                break
                    if not flag:
                        seg_tags = [0 for ex in ext_tags]
                        extag = [self.tag2idx[ex] for ex in ext_tags]
                        sents_tensor = torch.cat((sents_tensor, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                        seg_tags_tensor = torch.cat((seg_tags_tensor, torch.tensor(seg_tags).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                        ext_tags_tensor = torch.cat((ext_tags_tensor, torch.tensor(extag).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
            else:
                mx = self.cfg.PREDICATE_LIMIT
                cnt = 0
                while len(pre_spans) <= len(ext_tags_list):
                    pre_spans.append([-1])
                for pre_span, ext_tags in zip(pre_spans, ext_tags_list):
                    extag = [self.tag2idx[ex] for ex in ext_tags]
                    seg_tags = [1 if i in pre_span else 0 for i in range(len(ext_tags))]
                    sents_tensor = torch.cat((sents_tensor, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    seg_tags_tensor = torch.cat((seg_tags_tensor, torch.tensor(seg_tags).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    ext_tags_tensor = torch.cat((ext_tags_tensor, torch.tensor(extag).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                    cnt += 1
                    if cnt >= mx:
                        break
        sents_tensor = sents_tensor[1:].long()
        ext_tags_tensor = ext_tags_tensor[1:].long()
        seg_tags_tensor = seg_tags_tensor[1:].long()
        loss = loss_predicate + self.arg_model.neg_log_likelihood(sents_tensor, ext_tags_tensor, seg_tags_tensor).to(self.device)
        return loss

    def forward(self, _sents, _segs_list, mask = None):  
        with torch.no_grad():
            _, y_hats_list = self.pre_model(_sents, _segs_list)
        y_hats_list = y_hats_list.numpy().tolist()
        Y = []
        siz = 0
        each_siz = []
        sents_tensor = torch.ones(len(_sents[0])).unsqueeze(0).to(self.device)
        seg_tags_tensor = torch.ones(len(_sents[0])).unsqueeze(0).to(self.device)
        cnt = 0
        for y_hats, sent in zip(y_hats_list, _sents):
            preds = [self.idx2tag[y_hat] for y_hat in y_hats]
            pre_spans = self.get_predicate_span([preds])
            for pre_span in pre_spans:
                seg_tags = [1 if i in pre_span else 0 for i in range(len(preds))]
                sents_tensor = torch.cat((sents_tensor, torch.tensor(sent).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                seg_tags_tensor = torch.cat((seg_tags_tensor, torch.tensor(seg_tags).unsqueeze(0).to(self.device)), dim = 0).to(self.device)
                siz += 1
            each_siz.append(len(pre_spans))
        if(siz == 0) :
            Y.append([]*len(y_hats_list))
        else:
            sents_tensor = sents_tensor[1:].long()
            seg_tags_tensor = seg_tags_tensor[1:].long()
            _, y =  self.arg_model(sents_tensor, seg_tags_tensor)  # y_hat: (N, T)
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
