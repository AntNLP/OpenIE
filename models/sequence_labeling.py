from base64 import encode
# from ctypes.wintypes import tagSIZE
from re import S
from attr import frozen
import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding
from torchsummary import summary
import modules.bilstm as bilstm
# from module.crf import CRF
from modules.encoder import Encoder
from torchcrf import CRF
from sched import scheduler
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import torch.nn as nn
import torch
from utils.ner_dataset import NerDataset, pad
from torch.utils import data
from utils import vocab
import numpy as np
import argparse
import os
from utils.oie_eval.carb import Benchmark
from utils.oie_eval.oie_readers.goldReader import GoldReader
from utils.oie_eval.matcher import Matcher
import logging
logging.basicConfig(level = logging.INFO)

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag2idx, idx2tag, cfg, device):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hidden_dim = cfg.D_MODEL
        self.seg_num = cfg.SEG_NUM
        self.tag2idx = tag2idx
        self.tagset_size = len(idx2tag)

        self.crf = CRF(num_tags=self.tagset_size,batch_first=True)

        self.start_label_id = self.tag2idx['[CLS]']
        self.end_label_id = self.tag2idx['[SEP]']
        if(cfg.ENC_MODEL == 'bert'):
            self.encmodel = 'bert-base-cased'

        self.bilstm = bilstm.BiLSTM(self.hidden_dim, self.tagset_size)

        self.encoder = Encoder(frozen = False, encoder_model = self.encmodel, seg_num = self.seg_num, hidden_dim = self.hidden_dim)
        self.features_in_hook = []
        self.features_out_hook = []
        self.features = []
        


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))



    def neg_log_likelihood(self, sentence, tags, seg, mask = None):
        embeds = self.encoder(sentence, seg)  # [8, 75, 768]
        feats = self.bilstm(embeds)  #[batch_size, max_len, 16]
        return -self.crf(emissions=feats, tags = tags, mask = None, reduction = 'mean',)



    def forward(self, sentence, seg):  # dont confuse this with _forward_alg above.

        embeds = self.encoder(sentence, seg)

        lstm_feats = self.bilstm(embeds)

        # score, tag_seq = self.crf(lstm_feats)
        score, tag_seq = None, self.crf.decode(emissions=lstm_feats)
        return score, torch.tensor(tag_seq, dtype=torch.long)

class Bert_BiLSTM_CRF_Joint(nn.Module):
    def __init__(self, tag2idx, idx2tag, cfg, device = 'cpu'):
        super(Bert_BiLSTM_CRF_Joint, self).__init__()
        self.device = device
        self.opt = cfg.OPT
        self.hidden_dim = cfg.D_MODEL
        self.seg_num = cfg.SEG_NUM
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.tagset_size = len(tag2idx)

        self.start_label_id = self.tag2idx['[CLS]']
        self.end_label_id = self.tag2idx['[SEP]']
        self.features_in_hook = []
        self.features_out_hook = []
        self.features = []
        if(cfg.ENC_MODEL == 'bert'):
            self.encmodel = 'bert-base-cased'

        self.BBC_1 = Bert_BiLSTM_CRF(tag2idx, idx2tag, cfg, device)
        self.BBC_2 = Bert_BiLSTM_CRF(tag2idx, idx2tag, cfg, device)


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
                    ex_tags = [self.tag2idx[ex] for ex in ext[i]]
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
                            ex_tags = [self.tag2idx[ex] for ex in ext[j]]
                            break
                    if(overlap == False):
                        ex_tags = [self.tag2idx['O'] for ex in ext[0]]
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
                    ex_tags = [self.tag2idx[ex] for ex in ext[cnt]]

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
                        ex_tags = [self.tag2idx['O'] for ex in ext[0]]
                    else:
                        ex_tags = [self.tag2idx[ex] for ex in ext[cnt]]
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
                    ex_tags = [self.tag2idx[ex] for ex in ext[cnt]]

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
        # y_hat = [hat for hat in y_hat]
        # preds = [self.idx2tag[hat] for hat in y_hat]
        # _, y_hat = self.BBC_1.forward(sentence, seg)  # y_hat: (N, T)
        # print(preds)
        return _, Y

class SeqIE():
    def __init__(self, cfg, device):
        VOCAB = vocab.get_vocab(cfg.DOMAIN)
        tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
        idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.device = device
        self.cfg = cfg
        pass
    def get_model(self):
        if(self.cfg.METHOD == 'joint'):
            return Bert_BiLSTM_CRF_Joint(self.tag2idx, self.idx2tag, self.cfg, self.device)
        if(self.cfg.METHOD == 'pipeline'):
            return Bert_BiLSTM_CRF(self.tag2idx, self.idx2tag, self.cfg, self.device)
        else:
            return None
    def model_train(self):
        if(self.cfg.METHOD == 'joint'):
            def train(model, iterator, optimizer, criterion, scheduler, device):
                model.train()
                for i, batch in enumerate(iterator):
                    words, x, is_heads, tags, y, seqlens, seg, ext, p_tags = batch
                    x = x.to(device)
                    y = y.to(device)
                    seg = seg.to(device)
                    _y = y # for monitoring
                    optimizer.zero_grad()
                    loss = model.neg_log_likelihood(x, y, seg, p_tags = p_tags, extraction = ext) # logits: (N, T, VOCAB), y: (N, T)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if i==0:
                        print("=====sanity check======")
                        print("x:", x.cpu().numpy()[0][:seqlens[0]])
                        print("is_heads:", is_heads[0])
                        print("y:", _y.cpu().numpy()[0][:seqlens[0]])
                        print("tags:", tags[0])
                        print("seqlen:", seqlens[0])
                        print("seg:", seg.cpu().numpy()[0][:seqlens[0]])
                        print("=======================")
                    if i%10==0: # monitoring
                        print(f"step: {i}, loss: {loss.item()}")
            def dev(model, iterator, f, device, tag2idx, idx2tag, cfg):
                model.eval()

                Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
                with torch.no_grad():
                    for i, batch in enumerate(iterator):
                        words, x, is_heads, tags, y, seqlens, seg, ext, p_tags = batch
                        x = x.to(device)
                        seg = seg.to(device)
                        _, y_hat = model(x, seg)  
                        Words.extend(words)
                        Is_heads.extend(is_heads)
                        Tags.extend(tags)
                        Y.extend(y.numpy().tolist())
                        Y_hat.extend(y_hat)
                ## gets results and save
                cnt = 0
                if not os.path.isdir(cfg.out_dir):
                    os.makedirs(cfg.out_dir)
                outfile_path = cfg.OUTPUT
                with open(outfile_path, 'w', encoding='utf-8') as fw:
                    assert(len(Y_hat) == len(Words), f"len(preds)={len(Y_hat)}, len(words)={len(Words)}")
                    for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
                        for i in range(len(y_hat)):
                            y_h = [hat for head, hat in zip(is_heads, y_hat[i]) if head == 1]
                            preds = [idx2tag[hat] for hat in y_h]
                            assert len(preds)==len(words), f"len(preds)={len(preds)}, len(words)={len(words)}, len(tags)={len(tags)}"
                            texts = words[1:-1]
                            text = " ".join(texts)
                            tags = preds[1:-1]
                            predicates = []
                            args = []
                            for i in range(len(tags)):
                                if(tags[i] == 'P-B' or tags[i] == 'P-I'):
                                    predicates.append(texts[i])
                            predicate = " ".join(predicates)
                            if(len(predicate) == 0):
                                fw.write(text + '\t' + ' ')
                            else:
                                fw.write(text + '\t' + predicate)
                            argc = 0
                            # for i in range(len(tags)):
                            #     if( (len(args) == 0 and tags[i] == 'A-B') or tags[i] == 'A-I' ):
                            #         args.append(texts[i])
                            #     else:
                            #         if(len(args) != 0):
                            #             argc += 1
                            #             fw.write('\t' + " ".join(args))
                            #         args = []
                            # if(len(args) != 0):
                            #     argc += 1
                            #     fw.write('\t' + " ".join(args))

                            for pos in ['0', '1', '2', '3']:
                                args = []
                                for i in range(len(tags)):
                                    if(len(tags[i]) >= 2 and tags[i][1] == pos):
                                        args.append(texts[i])
                                if(len(args) != 0):
                                    fw.write('\t' + " ".join(args))
                            # if(argc == 0):
                            #     fw.write('\t' + " ")
                            fw.write('\n')
                            cnt += 1
                
                if(cnt == 0):
                    return 0, 0, 0, 0
                

                gold_path = cfg.DEV_GOLD
                out = cfg.OUT
                b = Benchmark(gold_path)
                out_filename = out
                predicted = GoldReader()

                predicted.read(outfile_path)
                matchingFunc = Matcher.binary_linient_tuple_match
                logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
                error_file = cfg.ERROR_LOG
                auc, optimal_f1_point= b.compare(  predicted = predicted.oie,
                                                    matchingFunc = matchingFunc,
                                                    output_fn = out_filename,
                                                    error_file = error_file)
                precision, recall, f1 = optimal_f1_point[0], optimal_f1_point[1], optimal_f1_point[2]
                print(auc, precision, recall, f1)
                fw.close()
                print("AUC:{:.5f}, P:{:.5f}, R:{:.5f}, F1:{:.5f}".format(auc,precision, recall, f1))
                return auc, precision, recall, f1
            cfg = self.cfg
            if not os.path.exists(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # set model hp
            VOCAB = vocab.get_vocab(cfg.DOMAIN)
            tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
            idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
            model = Bert_BiLSTM_CRF_Joint(self.tag2idx, self.idx2tag, self.cfg, self.device)
            model.cuda()
            print('Initial model Done')
            train_dataset = NerDataset(cfg.TRAIN, cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag, gold_tag = cfg.train_gold_tag, seg_tag = cfg.train_seg_tag)
            dev_dataset = NerDataset(cfg.DEV, cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag, gold_tag = cfg.dev_gold_tag, seg_tag = cfg.dev_seg_tag)
            print('Load Data Done')

            train_iter  = data.DataLoader(dataset=train_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=pad
                                        )
            dev_iter    = data.DataLoader(dataset=dev_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=pad
                                        )

            epochs = cfg.N_EPOCH
            optimizer = optim.Adam(model.parameters(), lr = float(cfg.LR))
            criterion = nn.CrossEntropyLoss(ignore_index=0) 
            total_steps = len(train_iter)*epochs
            print("train steps:" + str(total_steps))
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps/epochs/10, num_training_steps=total_steps)
            print('Start Train...,')
            
            best_f1 = -1
            best_epoch = 0
            if not os.path.isdir(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)
            fw_log = open(cfg.LOG, 'w')
            for epoch in range(0, cfg.N_EPOCH+1):  # 每个epoch对dev集进行测试
                train(model, train_iter, optimizer, criterion, scheduler, device)
                print(f"=========train at epoch={epoch}=========")
                fname = cfg.ckpt_dir
                if(epoch >= 190):
                    print(f"=========dev at epoch={epoch}=========")
                    auc, precision, recall, f1 = dev(model, dev_iter, fname, device,tag2idx, idx2tag, cfg)
                    # save current model
                    torch.save({
                        'epoch': epoch,
                        'best': (best_f1, best_epoch),
                        'model': model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict(),
                    }, cfg.LAST)

                    if(float(f1) > float(best_f1)):
                        best_f1 = f1
                        best_epoch = epoch
                        torch.save({
                        'epoch': epoch,
                        'best': (best_f1, best_epoch),
                        'model': model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict(),
                    }, cfg.BEST)

                    fw_log.write(
                            'n_epcch: ' + str(epoch) + ' ' +
                            'auc: ' + str(auc) + ' '
                            'precision: ' + str(precision) + ' ' + 
                            'recall: ' + str(recall) + ' ' + 
                            'f1: ' + str(f1) + ' ' +
                            'best_f1: ' + str(best_f1) + ' ' +
                            'best_epoch: ' + str(best_epoch) +
                            '\n'
                    ) 
                    print(  'n_epcch: ' + str(epoch) + ' ' +
                            'auc: ' + str(auc) + ' '
                            'precision: ' + str(precision) + ' ' + 
                            'recall: ' + str(recall) + ' ' + 
                            'f1: ' + str(f1) + ' ' +
                            'best_f1: ' + str(best_f1) + ' ' +
                            'best_epoch: ' + str(best_epoch) +
                            '\n')
            fw_log.close()
        elif(self.cfg.METHOD == 'pipeline'):
            def train_pre(model, iterator, optimizer, criterion, scheduler,  tag2idx, idx2tag, device, cfg):
                model.train()
                for i, batch in enumerate(iterator):
                    words, x, is_heads, tags, y, seqlens, seg, ext, p_tags = batch
                    x = x.to(device)
                    y = y.to(device)
                    seg = seg.to(device)
                    _y = y # for monitoring
                    optimizer.zero_grad()
                    loss = model.neg_log_likelihood(x, y, seg) # logits: (N, T, VOCAB), y: (N, T)
                    # _, y_hat = model(x, seg)
                    # x = x.to(device)
                    # seg = seg.to(device)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if i==0:
                        print("=====sanity check======")
                        #print("words:", words[0])
                        print("x:", x.cpu().numpy()[0][:seqlens[0]])
                        # print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
                        print("is_heads:", is_heads[0])
                        print("y:", _y.cpu().numpy()[0][:seqlens[0]])
                        print("tags:", tags[0])
                        print("seqlen:", seqlens[0])
                        print("seg:", seg.cpu().numpy()[0][:seqlens[0]])
                        print("=======================")
                    if i%10==0: # monitoring
                        print(f"step: {i}, loss: {loss.item()}")
            def create_tmp_data(model, type, iterator, optimizer, criterion, scheduler,  tag2idx, idx2tag, device, cfg):
                model.eval()
                f = './tmp.' + cfg.OPT + '.' + type
                fw = open(f,'w')
                if(type == 'train'):
                    for i, batch in enumerate(iterator):
                        words, x, is_heads, tags, y, seqlens, seg, ext, p_tags = batch
                        x = x.to(device)
                        y = y.to(device)
                        seg = seg.to(device)
                        _y = y # for monitoring
                        _, y_hat = model(x, seg)
                        x = x.to(device)
                        seg = seg.to(device)
                        for k in range(len(words)):
                            # 预测关系
                            y_h = [hat for head, hat in zip(is_heads[k], y_hat[k].cpu().numpy().tolist()) if head == 1]
                            preds = [idx2tag[hat] for hat in y_h]
                            assert(len(preds) == len(words[k]))
                            # 关系解码
                            span = []
                            ture_span = []
                            for ex in ext[k]:
                                l = -1
                                r = -1
                                for i in range(len(ex)):
                                    if(ex[i] == 'P-B'):
                                        l = r = i
                                    elif(ex[i] == 'P-I'):
                                        r = i
                                    else:
                                        if(l != -1):
                                            ture_span.append([l, r+1])
                                            break
                            if(cfg.OPT == 'soft'):
                                l = -1
                                r = -1
                                for i in range(len(preds)):
                                    if(preds[i] == 'P-B'):
                                        if(l != -1):
                                            # 方便range
                                            span.append([l, r+1])
                                        l = r = i
                                    elif(preds[i] == 'P-I'):
                                        r = i
                                    else:
                                        if(l != -1):
                                            # 方便range
                                            span.append([l, r+1])
                                        l = -1

                                sspan = []
                                for tsp in ture_span:
                                    for sp in span:
                                        #三种相交情况
                                        if( (sp[0]<=tsp[0] and sp[1]>=tsp[0]) or (sp[0]<=tsp[1] and sp[1]>=tsp[1]) or (sp[0]>=tsp[0] and sp[1]<=tsp[1]) ):
                                            sspan.append([tsp[0], tsp[1]])
                                span = sspan
                            elif(cfg.OPT == 'gold'):
                                span = [sp for sp in ture_span]
                            else:
                                l = -1
                                r = -1
                                for i in range(len(preds)):
                                    if(preds[i] == 'P-B'):
                                        if(l != -1):
                                            # 方便range
                                            span.append([l, r+1])
                                        l = r = i
                                    elif(preds[i] == 'P-I'):
                                        r = i
                                    else:
                                        if(l != -1):
                                            # 方便range
                                            span.append([l, r+1])
                                        l = -1
                            pre_tag = []
                            cnt = 0
                            for sp in span:
                                cnt += 1
                                pre_tag.append(['O']*(sp[0]) + ['P-B'] + ['P-I']*(sp[1] - sp[0] - 1) + ['O']*(len(words[k]) - sp[1]))
                                if(cnt == len(ext[k])):
                                    break
                            while(len(pre_tag) < len(ext[k])):
                                pre_tag.append(['O']*len(words[k]))
                            # print(words[k])
                            # print(ext[k])
                            # print(pre_tag)
                            for ex, pt in zip(ext[k], pre_tag):
                                ex = [e for head, e in zip(is_heads[k], ex) if head == 1]
                                for w, p, s in zip(words[k][1:-1], ex[1:-1], pt[1:-1]):
                                    fw.write(w + '\t' + p + '\t' + s  + '\n')
                                fw.write('\n')
                if(type == 'dev'):
                    for i, batch in enumerate(iterator):
                        words, x, is_heads, tags, y, seqlens, seg, ext, p_tags = batch
                        x = x.to(device)
                        y = y.to(device)
                        seg = seg.to(device)
                        _y = y # for monitoring
                        _, y_hat = model(x, seg)
                        x = x.to(device)
                        seg = seg.to(device)
                        for k in range(len(words)):
                            # 预测关系
                            y_h = [hat for head, hat in zip(is_heads[k], y_hat[k].cpu().numpy().tolist()) if head == 1]
                            preds = [idx2tag[hat] for hat in y_h]
                            # 关系解码
                            span = []
                            l = -1
                            r = -1
                            for i in range(len(preds)):
                                if(preds[i] == 'P-B'):
                                    if(l != -1):
                                        # 方便range
                                        span.append([l, r+1])
                                    l = r = i
                                elif(preds[i] == 'P-I'):
                                    r = i
                                else:
                                    if(l != -1):
                                        # 方便range
                                        span.append([l, r+1])
                                    l = -1
                            pre_tag = []
                            cnt = 0
                            for sp in span:
                                pre_tag.append(['O']*(sp[0]) + ['P-B'] + ['P-I']*(sp[1] - sp[0] - 1) + ['O']*(len(words[k]) - sp[1]))
                                cnt += 1
                                if(cnt >= 3):
                                    break
                            for pt in pre_tag:
                                assert(len(words[k]) == len(pt))
                                for w,  s in zip(words[k][1:-1],  pt[1:-1]):
                                    fw.write(w + '\t' + 'O' + '\t' + s  + '\n')
                                fw.write('\n')
            def train_arg(model, iterator, optimizer, criterion, scheduler,  tag2idx, idx2tag, device, cfg):
                model.train()
                for i, batch in enumerate(iterator):
                    words, x, is_heads, tags, y, seqlens, seg, ext, p_tags = batch
                    x = x.to(device)
                    y = y.to(device)
                    seg = seg.to(device)
                    _y = y # for monitoring
                    optimizer.zero_grad()
                    loss = model.neg_log_likelihood(x, y, seg) # logits: (N, T, VOCAB), y: (N, T)
                    x = x.to(device)
                    seg = seg.to(device)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if i==0:
                        print("=====sanity check======")
                        #print("words:", words[0])
                        print("x:", x.cpu().numpy()[0][:seqlens[0]])
                        # print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
                        print("is_heads:", is_heads[0])
                        print("y:", _y.cpu().numpy()[0][:seqlens[0]])
                        print("tags:", tags[0])
                        print("seqlen:", seqlens[0])
                        print("seg:", seg.cpu().numpy()[0][:seqlens[0]])
                        print("=======================")
                    if i%10==0: # monitoring
                        print(f"step: {i}, loss: {loss.item()}")

                # create pred data

            def dev(model, iterator, f, device, tag2idx, idx2tag, cfg ):
                model.eval()

                Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
                with torch.no_grad():
                    for i, batch in enumerate(iterator):
                        words, x, is_heads, tags, y, seqlens, seg, ext, p_tags = batch
                        x = x.to(device)
                        seg = seg.to(device)
                        _, y_hat = model(x, seg) 
                        Words.extend(words)
                        Is_heads.extend(is_heads)
                        Tags.extend(tags)
                        Y.extend(y.numpy().tolist())
                        Y_hat.extend(y_hat.cpu().numpy().tolist())
                ## gets results and save
                outfile_path = cfg.OUTPUT
                d = []
                with open(outfile_path, 'w', encoding='utf-8') as fout:
                    for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
                        y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
                        preds = [idx2tag[hat] for hat in y_hat]
                        assert len(preds)==len(words)==len(tags), f"len(preds)={len(preds)}, len(words)={len(words)}, len(tags)={len(tags)}"
                        texts = words[1:-1]
                        text = " ".join(texts)
                        tags = preds[1:-1]
                        predicates = []
                        args = []
                        for i in range(len(tags)):
                            if(tags[i] == 'P-B' or tags[i] == 'P-I'):
                                predicates.append(texts[i])
                        predicate = " ".join(predicates)
                        if(len(predicate) == 0):
                            predicate = ' '
                        fout.write(text + '\t' + predicate)
                        # for i in range(len(tags)):
                        #     if( (len(args) == 0 and tags[i] == 'A-B') or tags[i] == 'A-I' ):
                        #         args.append(texts[i])
                        #     else:
                        #         if(len(args) != 0):
                        #             fout.write('\t' + " ".join(args))
                        #         args = []
                        # if(len(args) != 0):
                        #     fout.write('\t' + " ".join(args))
                        for pos in ['0', '1', '2', '3']:
                            args = []
                            for i in range(len(tags)):
                                if(len(tags[i]) >= 2 and tags[i][1] == pos):
                                    args.append(texts[i])
                            if(len(args) != 0):
                                fout.write('\t' + " ".join(args))
                        fout.write('\n')
                    fout.close()
                gold_path = cfg.DEV_GOLD
                out = cfg.OUT
                b = Benchmark(gold_path)
                out_filename = out
                predicted = GoldReader()
                
                predicted.read(outfile_path)
                matchingFunc = Matcher.binary_linient_tuple_match
                logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
                error_file = cfg.ckpt_dir + '/log'
                auc, optimal_f1_point = b.compare(  predicted = predicted.oie,
                                                    matchingFunc = matchingFunc,
                                                    output_fn = out_filename,
                                                    error_file = error_file)
                ## calc metric
                precision, recall, f1 = optimal_f1_point[0], optimal_f1_point[1], optimal_f1_point[2]
                print(auc, precision, recall, f1)
                return auc, precision, recall, f1
            
            # Configuration file processing
            cfg = self.cfg
            if not os.path.exists(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)
            if not os.path.exists(cfg.out_dir):
                os.makedirs(cfg.out_dir)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # set model hp
            VOCAB = vocab.get_vocab(cfg.DOMAIN)
            tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
            idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

            train_dataset = NerDataset(cfg.TRAIN, cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag, gold_tag = cfg.train_gold_tag, seg_tag = cfg.train_seg_tag)
            
            dev_dataset = NerDataset(cfg.DEV, cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag, gold_tag = cfg.dev_gold_tag, seg_tag = cfg.dev_seg_tag)
            print('Load Data Done')

            train_iter  = data.DataLoader(dataset=train_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=pad
                                        )
            dev_iter    = data.DataLoader(dataset=dev_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=pad
                                        )

            epochs = cfg.N_EPOCH
            model_pre = Bert_BiLSTM_CRF(tag2idx, idx2tag, cfg, device)
            optimizer = optim.Adam(model_pre.parameters(), lr = float(cfg.LR))
            
            model_pre.cuda()
            criterion = nn.CrossEntropyLoss(ignore_index=0) 
            print('Initial model Done')
            total_steps = len(train_iter)*epochs
            print("train steps:" + str(total_steps))
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps/epochs/10, num_training_steps=total_steps)
            print('Start pre_train...,')
            fw_log = open(cfg.LOG, 'w')
            if(cfg.CKP == False):
                for epoch in range(0, cfg.N_EPOCH+1): 
                    train_pre(model_pre, train_iter, optimizer, criterion, scheduler, tag2idx, idx2tag, device, cfg)
                    if(epoch % 50 == 0):
                        torch.save({
                            'epoch': epoch,
                            'model_pre': model_pre.state_dict(),
                            'optim_pre': optimizer.state_dict(),
                            'sched_pre': scheduler.state_dict(),
                        }, cfg.PRE)
            ckpt = torch.load('/home/weidu/data/OPENIE/ckpts/CaRB/pipeline/default/pre.pt')
            model_pre.load_state_dict(ckpt['model_pre'])
            optimizer.load_state_dict(ckpt['optim_pre'])
            print('pre_train Ended')


            print('Start create_tmp_data...')
            create_tmp_data(model_pre, 'train', train_iter, optimizer, criterion, scheduler, tag2idx, idx2tag, device, cfg)
            create_tmp_data(model_pre, 'dev', dev_iter, optimizer, criterion, scheduler, tag2idx, idx2tag, device, cfg)
            print('create_tmp_data Ended')
            print('Start arg_train...')
            model_arg = Bert_BiLSTM_CRF(tag2idx, idx2tag, cfg, device)
            model_arg.cuda()
            optimizer = optim.Adam(model_arg.parameters(), lr = float(cfg.LR))
            criterion = nn.CrossEntropyLoss(ignore_index=0) 
            total_steps = len(train_iter)*epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps/epochs/10, num_training_steps=total_steps)
            if not os.path.isdir(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)

            arg_train_dataset = NerDataset('./tmp.' + cfg.OPT + '.' + 'train', cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag, gold_tag = cfg.train_gold_tag, seg_tag = True)
            arg_train_iter  = data.DataLoader(dataset=arg_train_dataset,
                                            batch_size=cfg.N_BATCH,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=pad
                                            )

            dev_dataset = NerDataset('./tmp.' + cfg.OPT + '.' + 'dev', cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag, gold_tag = cfg.train_gold_tag, seg_tag = True)
            dev_iter  = data.DataLoader(dataset=dev_dataset,
                                            batch_size=cfg.N_BATCH,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=pad
                                            )
            best_f1 = 0
            best_epoch = 0
            for epoch in range(0, cfg.N_EPOCH*2+1):  # 每个epoch对dev集进行测试
                
                train_arg(model_arg, arg_train_iter, optimizer, criterion, scheduler, tag2idx, idx2tag, device, cfg)
                print(f"=========train eval at epoch={epoch}=========")
                fname = cfg.ckpt_dir
                if(epoch % 20 == 0):
                    print(f"=========dev eval at epoch={epoch}=========")
                    auc, precision, recall, f1 = dev(model_arg, dev_iter, fname, device, tag2idx, idx2tag, cfg)
                            # save current model
                    torch.save({
                        'epoch_arg': epoch,
                        'best_arg': (best_f1, best_epoch),
                        'model_arg': model_arg.state_dict(),
                        'optim_arg': optimizer.state_dict(),
                        'sched_arg': scheduler.state_dict(),
                    }, cfg.LAST)

                    if(float(f1) > float(best_f1)):
                        best_f1 = f1
                        best_epoch = epoch
                        torch.save({
                        'epoch_arg': epoch,
                        'best_arg': (best_f1, best_epoch),
                        'model_arg': model_arg.state_dict(),
                        'optim_arg': optimizer.state_dict(),
                        'sched_arg': scheduler.state_dict(),
                    }, cfg.BEST)
                    fw_log.write(
                            'n_epcch_arg: ' + str(epoch) + ' ' +
                            'precision_arg: ' + str(precision) + ' ' + 
                            'recall_arg: ' + str(recall) + ' ' + 
                            'f1_arg: ' + str(f1) + ' ' + 
                            'best_f1_arg: ' + str(best_f1) + ' ' +
                            'best_epoch_arg: ' + str(best_epoch) +
                            '\n'
                    ) 
                    print("weights were saved to" + cfg.LAST)
            os.remove('./tmp.' + cfg.OPT + '.' + 'train')
            os.remove('./tmp.' + cfg.OPT + '.' + 'dev')
            print('arg_train Ended')

