
from sched import scheduler
from antu.io.configurators import IniConfigurator
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import torch.nn as nn
import torch
from module.Bert_BiLSTM_CRF_joint import Bert_BiLSTM_CRF_Joint
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

        # logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        # y = y.view(-1)  # (N*T,)
        # writer.add_scalar('data/loss', loss.item(), )

        # loss = criterion(logits, y)
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

def eval(model, iterator, f, device, tag2idx, idx2tag, cfg):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens, seg, ext, p_tags = batch
            x = x.to(device)
            seg = seg.to(device)
            _, y_hat = model(x, seg)  # y_hat: (N, T)
            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            # print(y.numpy().tolist())
            Y_hat.extend(y_hat)
    ## gets results and save
    cnt = 0
    outfile_path = cfg.OUTPUT
    with open(outfile_path, 'w', encoding='utf-8') as fw:
        assert(len(Y_hat) == len(Words), f"len(preds)={len(Y_hat)}, len(words)={len(Words)}")
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
    error_file = './log'
    auc, optimal_f1_point= b.compare(  predicted = predicted.oie,
                                        matchingFunc = matchingFunc,
                                        output_fn = out_filename,
                                        error_file = error_file)
    precision, recall, f1 = optimal_f1_point[0], optimal_f1_point[1], optimal_f1_point[2]
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
    ## calc metric
    print(auc, precision, recall, f1)
    fw.close()
    print("AUC:{:.5f}, P:{:.5f}, R:{:.5f}, F1:{:.5f}".format(auc,precision, recall, f1))
    return auc, precision, recall, f1
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
    model = Bert_BiLSTM_CRF_Joint(tag2idx, idx2tag, encmodel = cfg.MODEL_TYPE, seg_num = cfg.SEG_NUM, device=device, opt=cfg.OPT)
    # print(model)
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
    for epoch in range(0, cfg.N_EPOCH):  # 每个epoch对dev集进行测试
        train(model, train_iter, optimizer, criterion, scheduler, device)
        print(f"=========train at epoch={epoch}=========")
        fname = cfg.ckpt_dir
        if(epoch % 20 == 0 and epoch >= 50):
            print(f"=========dev at epoch={epoch}=========")
            auc, precision, recall, f1 = eval(model, dev_iter, fname, device,tag2idx, idx2tag, cfg)
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
                    'f1: ' + str(f1) + 
                    'best_f1: ' + str(best_f1) + ' ' +
                    'best_epoch: ' + str(best_epoch) +
                    '\n'
            ) 
            print(  'n_epcch: ' + str(epoch) + ' ' +
                    'auc: ' + str(auc) + ' '
                    'precision: ' + str(precision) + ' ' + 
                    'recall: ' + str(recall) + ' ' + 
                    'f1: ' + str(f1) + 
                    'best_f1: ' + str(best_f1) + ' ' +
                    'best_epoch: ' + str(best_epoch) +
                    '\n')
    fw_log.close()
if __name__ == "__main__":
    main()
