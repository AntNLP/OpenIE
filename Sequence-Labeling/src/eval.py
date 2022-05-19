

from sched import scheduler
from antu.io.configurators import IniConfigurator
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import torch.nn as nn
import torch
from module.Bert_BiLSTM_CRF import Bert_BiLSTM_CRF
from module.Bert_BiLSTM_CRF_joint import Bert_BiLSTM_CRF_Joint
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

def eval(model, iterator, f, device, tag2idx, idx2tag, cfg ):
    model.eval()
    if(cfg.METHOD == 'pipeline'):
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
        with open(outfile_path, 'w', encoding='utf-8') as fw:
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
                fw.write(text + '\t' + predicate)
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
                        fw.write('\t' + " ".join(args))
                fw.write('\n')
            fw.close()
        gold_path = cfg.TEST_GOLD
        out = cfg.OUT
        b = Benchmark(gold_path)
        out_filename = out
        predicted = GoldReader()
        
        predicted.read(outfile_path)
        matchingFunc = Matcher.binary_linient_tuple_match
        logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
        error_file = cfg.ckpt_dir + 'log'
        auc, optimal_f1_point = b.compare(  predicted = predicted.oie,
                                            matchingFunc = matchingFunc,
                                            output_fn = out_filename,
                                            error_file = error_file)
        ## calc metric
        precision, recall, f1 = optimal_f1_point[0], optimal_f1_point[1], optimal_f1_point[2]
        print(auc, precision, recall, f1)
        return auc, precision, recall, f1
    elif(cfg.METHOD == 'joint'):
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
        outfile_path = cfg.OUTPUT
        with open(outfile_path, 'w', encoding='utf-8') as fw:
            for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
                for i in range(len(y_hat)):
                    # y_hat[i] = [hat for hat in y_hat[i]]
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
                    # if(argc == 0):
                    #     fw.write('\t' + " ")
                    for pos in ['0', '1', '2', '3']:
                        args = []
                        for i in range(len(tags)):
                            if(len(tags[i]) >= 2 and tags[i][1] == pos):
                                args.append(texts[i])
                        if(len(args) != 0):
                            fw.write('\t' + " ".join(args))
                    fw.write('\n')
                    cnt += 1
        gold_path = cfg.TEST_GOLD
        out = cfg.OUT
        b = Benchmark(gold_path)
        out_filename = out
        predicted = GoldReader()
        predicted.read(outfile_path)
        # matchingFunc = Matcher.binary_linient_tuple_match
        matchingFunc = Matcher.lexicalMatch
        logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
        error_file = './log'
        auc, optimal_f1_point= b.compare(  predicted = predicted.oie,
                                            matchingFunc = matchingFunc,
                                            output_fn = out_filename,
                                            error_file = error_file)
        precision, recall, f1 = optimal_f1_point[0], optimal_f1_point[1], optimal_f1_point[2]
        fw.close()
        print("AUC:{:.5f}, P:{:.5f}, R:{:.5f}, F1:{:.5f}".format(auc,precision, recall, f1))
        return auc, precision, recall, f1
def create_tmp_data(model, type, iterator, optimizer, criterion,  tag2idx, idx2tag, device, cfg):
    model.eval()
    f = './tmp.' + type
    fw = open(f,'w')
    if(type == 'test'):
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
                tag = tags[k]
                preds = [idx2tag[hat] for hat in y_h]
                assert(len(preds) == len(tags[k]))
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
                for sp in span:
                    pre_tag.append(['O']*(sp[0]) + ['P-B'] + ['P-I']*(sp[1] - sp[0] - 1) + ['O']*(len(words[k]) - sp[1]))
                for pt, ta in zip(pre_tag, tag):
                    assert(len(words[k]) == len(pt))
                    for w, t,  s in zip(words[k][1:-1], tag[1:-1], pt[1:-1]):
                        fw.write(w + '\t' + t + '\t' + s  + '\n')
                    fw.write('\n')
    
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

    if(cfg.METHOD == 'pipeline'):
        model_pre = Bert_BiLSTM_CRF(tag2idx, encmodel = cfg.MODEL_TYPE, seg_num = cfg.SEG_NUM)
        optimizer = optim.Adam(model_pre.parameters(), lr = float(cfg.LR))
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        ckpt = torch.load(cfg.PRE)
        model_pre.load_state_dict(ckpt['model_pre'])
        optimizer.load_state_dict(ckpt['optim_pre'])

        model_pre.cuda()
        test_dataset = NerDataset(cfg.TEST, cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag, gold_tag = cfg.dev_gold_tag, seg_tag = cfg.dev_seg_tag)
        test_iter    = data.DataLoader( dataset=test_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=pad
                                        )
        create_tmp_data(model_pre, 'test', test_iter, optimizer, criterion, tag2idx, idx2tag, device, cfg)
        
        model_arg = Bert_BiLSTM_CRF(tag2idx, encmodel = cfg.MODEL_TYPE, seg_num = cfg.SEG_NUM)
        optimizer = optim.Adam(model_arg.parameters(), lr = float(cfg.LR))
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        ckpt = torch.load(cfg.BEST)
        model_arg.load_state_dict(ckpt['model_arg'])
        optimizer.load_state_dict(ckpt['optim_arg'])
        model_arg.cuda()
        test_iter  = data.DataLoader(dataset=test_dataset,
                                 batch_size=cfg.N_BATCH,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad
                                 )
        test_dataset = NerDataset('./tmp.test', cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag, gold_tag = cfg.train_gold_tag, seg_tag = True)
        test_iter  = data.DataLoader(dataset=test_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=pad
                                    )
        fname = cfg.ckpt_dir
        auc, precision, recall, f1 = eval(model_arg, test_iter, fname, device, tag2idx, idx2tag, cfg)
        print("auc:{:.5f}, precision:{:.5f}, recall:{:.5f}, f1:{:.5f},".format(auc, precision, recall, f1))
        fw = open(fname + 'res', 'w')

        fw.write("auc:{:.5f}, precision:{:.5f}, recall:{:.5f}, f1:{:.5f},".format(auc, precision, recall, f1))

    elif(cfg.METHOD == 'joint'):
        model_arg = Bert_BiLSTM_CRF_Joint(tag2idx, idx2tag, encmodel = cfg.MODEL_TYPE, seg_num = cfg.SEG_NUM, device=device)
        ckpt = torch.load(cfg.BEST)
        optimizer = optim.Adam(model_arg.parameters(), lr = float(cfg.LR))
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        model_arg.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        model_arg.cuda()
        test_dataset = NerDataset(cfg.TEST, cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag, gold_tag = cfg.train_gold_tag, seg_tag = True)
        test_iter  = data.DataLoader(dataset=test_dataset,
                                    batch_size=cfg.N_BATCH,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=pad
                                    )
        fname = cfg.ckpt_dir    
        auc, precision, recall, f1 = eval(model_arg, test_iter, fname, device, tag2idx, idx2tag, cfg)
        print("auc:{:.5f}, precision:{:.5f}, recall:{:.5f}, f1:{:.5f},".format(auc, precision, recall, f1))
        fw = open(fname + 'res', 'w')

        fw.write("auc:{:.5f}, precision:{:.5f}, recall:{:.5f}, f1:{:.5f},".format(auc, precision, recall, f1))

    

if __name__ == "__main__":
    main()
