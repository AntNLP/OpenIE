
from sched import scheduler
from antu.io.configurators import IniConfigurator
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import torch.nn as nn
import torch
from module.Bert_BiLSTM_CRF import Bert_BiLSTM_CRF
from utils.oie_eval.oie_readers.goldReader import GoldReader
from utils.ner_dataset import NerDataset, pad
from torch.utils import data
from utils import vocab
import numpy as np
import argparse
import os
from utils.oie_eval.carb import Benchmark
from utils.oie_eval.oie_readers.seqReader import SeqReader
from utils.oie_eval.matcher import Matcher

import logging
from tqdm import tqdm
logging.basicConfig(level = logging.INFO)
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


    # gold_path = cfg.DEV_GOLD
    # out = cfg.ckpt_dir + '/dev.dat'
    # b = Benchmark1(gold_path)
    # out_filename = out
    # predicted = SeqReader1()
    
    # predicted.read(outfile_path)
    # matchingFunc = Matcher1.lexicalMatch
    # logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
    # error_file = cfg.ckpt_dir + '/log'
    # precision, recall, f1, threshold= b.compare(  predicted = predicted.oie,
    #                                     matchingFunc = matchingFunc,
    #                                     output_fn = out_filename,
    #                                     error_file = error_file)
    # ## calc metric
    # print(precision, recall, f1, threshold)
    # return precision, recall, f1, threshold

def main():
    # Configuration file processing
    parser = argparse.ArgumentParser(description="Usage for OPENIE.")
    parser.add_argument('--CFG', type=str, help="Path to config file.")
    parser.add_argument('--DEBUG', action='store_true', help="DEBUG mode.")
    args, extra_args = parser.parse_known_args()
    cfg = IniConfigurator(args.CFG, extra_args)
    if not os.path.exists(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = Bert_BiLSTM_CRF(tag2idx, 'hfl/chinese-bert-wwm-ext')

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
    model_pre = Bert_BiLSTM_CRF(tag2idx, encmodel = cfg.MODEL_TYPE, seg_num = cfg.SEG_NUM)
    optimizer = optim.Adam(model_pre.parameters(), lr = float(cfg.LR))
    # ckpt = torch.load('/home/weidu/data/OPENIE/ckpts/CaRB/pipeline/default/pre.pt')
    model_pre.load_state_dict(ckpt['model_pre'])
    optimizer.load_state_dict(ckpt['optim_pre'])
    model_pre.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    print('Initial model Done')
    total_steps = len(train_iter)*epochs
    print("train steps:" + str(total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps/epochs/10, num_training_steps=total_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, 1, num_training_steps=total_steps)
    print('Start pre_train...,')
    fw_log = open(cfg.LOG, 'w')
    for epoch in range(0, cfg.N_EPOCH+1): 
        train_pre(model_pre, train_iter, optimizer, criterion, scheduler, tag2idx, idx2tag, device, cfg)
        if(epoch % 50 == 0):
            torch.save({
                'epoch': epoch,
                'model_pre': model_pre.state_dict(),
                'optim_pre': optimizer.state_dict(),
                'sched_pre': scheduler.state_dict(),
            }, cfg.PRE)
    print('pre_train Ended')
    print('Start create_tmp_data...')
    create_tmp_data(model_pre, 'train', train_iter, optimizer, criterion, scheduler, tag2idx, idx2tag, device, cfg)
    create_tmp_data(model_pre, 'dev', dev_iter, optimizer, criterion, scheduler, tag2idx, idx2tag, device, cfg)
    print('create_tmp_data Ended')
    print('Start arg_train...')
    model_arg = Bert_BiLSTM_CRF(tag2idx, encmodel = cfg.MODEL_TYPE, seg_num = cfg.SEG_NUM)
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
        print(f"=========dev eval at epoch={epoch}=========")
        fname = cfg.ckpt_dir
        if(epoch % 20 == 0):
            
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
                    'f1_arg: ' + str(f1) + '\n'
                    'best_f1_arg: ' + str(best_f1) + ' ' +
                    'best_epoch_arg: ' + str(best_epoch) +
                    '\n'
            ) 
            print("weights were saved to" + cfg.LAST)
    os.remove('./tmp.' + cfg.OPT + '.' + 'train')
    os.remove('./tmp.' + cfg.OPT + '.' + 'dev')
    print('arg_train Ended')

if __name__ == "__main__":
    main()
