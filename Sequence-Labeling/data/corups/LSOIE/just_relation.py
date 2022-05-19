
from sched import scheduler
from antu.io.configurators import IniConfigurator
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import torch.nn as nn
import torch
from module.Bert_BiLSTM_CRF import Bert_BiLSTM_CRF
from utils.ner_dataset import NerDataset, pad
from torch.utils import data
from utils import vocab
import numpy as np
import argparse
import os
from utils.oie_eval1.benchmark import Benchmark
from utils.oie_eval1.oie_readers.seqReader import SeqReader
from utils.oie_eval1.matcher import Matcher
import logging
logging.basicConfig(level = logging.INFO)
def train(model, iterator, optimizer, criterion, scheduler, device):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens, seg = batch
        x = x.to(device)
        y = y.to(device)
        seg = seg.to(device)
        _y = y # for monitoring
        optimizer.zero_grad()
        
        loss = model.neg_log_likelihood(x, y, seg) # logits: (N, T, VOCAB), y: (N, T)

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

def dev(model, iterator, f, device, tag2idx, idx2tag, task = 'oie2026'):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens, seg = batch
            x = x.to(device)
            # y = y.to(device)
            seg = seg.to(device)
            _, y_hat = model(x, seg)  # y_hat: (N, T)
            # print(len(words))
            # print(len(tags))
            # print(len(is_heads))
            # print(len(y_hat))
            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
    ## gets results and save
    with open("temp", 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words)==len(tags), f"len(preds)={len(preds)}, len(words)={len(words)}, len(tags)={len(tags)}"
            for w, t, p in zip(words[1:-1], tags[1:-1], preds[1:-1]):
                fout.write(f"{w}\t{t}\t{p}\n")
            fout.write("\n")
    ## calc metric
    y_true =  np.array([tag2idx[line.split('\t')[1]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split('\t')[2]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])

    # num_proposed = len(y_pred[y_pred>3])
    # num_correct = (np.logical_and(y_true==y_pred, y_true>3)).astype(np.int).sum()
    # num_gold = len(y_true[y_true>3])
    # gold只有rel
    r = 3
    num_proposed = len(y_pred[y_pred>r])
    num_correct = (np.logical_and(y_true==y_pred, y_true>r)).astype(np.int).sum()
    num_gold = len(y_true[y_true>r])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    # final = f + ".P%.4f_R%.4f_F%.4f" %(precision, recall, f1)
    # with open(final, 'w', encoding='utf-8') as fout:
    #     result = open("temp6", "r", encoding='utf-8').read()
    #     fout.write(f"{result}\n")

    #     fout.write(f"precision={precision}\n")
    #     fout.write(f"recall={recall}\n")
    #     fout.write(f"f1={f1}\n")

    os.remove("temp")

    print("precision=%.4f"%precision)
    print("recall=%.4f"%recall)
    print("f1=%.4f"%f1)
    return precision, recall, f1
# def dev(model, iterator, f, device, tag2idx, idx2tag, cfg ):
#     model.eval()

#     Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
#     with torch.no_grad():
#         for i, batch in enumerate(iterator):
#             words, x, is_heads, tags, y, seqlens, seg = batch
#             x = x.to(device)
#             # y = y.to(device)
#             seg = seg.to(device)
#             _, y_hat = model(x, seg)  # y_hat: (N, T)
#             # print(len(words))
#             # print(len(tags))
#             # print(len(is_heads))
#             # print(len(y_hat))
#             Words.extend(words)
#             Is_heads.extend(is_heads)
#             Tags.extend(tags)
#             Y.extend(y.numpy().tolist())
#             Y_hat.extend(y_hat.cpu().numpy().tolist())
#     ## gets results and save
#     outfile_path = cfg.TASK + '_out.dev'
#     d = []
#     with open(outfile_path, 'w', encoding='utf-8') as fout:
#         for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
#             y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
#             preds = [idx2tag[hat] for hat in y_hat]
#             assert len(preds)==len(words)==len(tags), f"len(preds)={len(preds)}, len(words)={len(words)}, len(tags)={len(tags)}"
#             s = ''
#             for w, t, p in zip(words[1:-1], tags[1:-1], preds[1:-1]):
#                 # fout.write(f"{w}\t{t}\t{p}\n")
#                 s += w
#             if(s not in d):
#                 for w, t, p in zip(words[1:-1], tags[1:-1], preds[1:-1]):
#                     fout.write(f"{w}\t{p}\n")
#                 fout.write('\n')
#                 d.append(s)
            
#         fout.close()
#     # outfile_path = '/home/weidu/OPENIE/data/corups/oie2016/oie2016_np_dev.txt'
#     gold_path = cfg.gold_path + '/dev.oie'
#     out = './dev.dat'
#     b = Benchmark(gold_path)
#     out_filename = out
#     predicted = SeqReader()
#     predicted.read(outfile_path)
#     matchingFunc = Matcher.lexicalMatch
#     logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
#     error_file = './log'
#     precision, recall, f1, threshold = b.compare(  predicted = predicted.oie,
#                                         matchingFunc = matchingFunc,
#                                         output_fn = out_filename,
#                                         error_file = error_file)
#     ## calc metric
#     return precision, recall, f1
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
    model = Bert_BiLSTM_CRF(tag2idx, encmodel = cfg.MODEL_TYPE, seg_num = cfg.SEG_NUM)
    print(model)
    model.cuda()
    print('Initial model Done')
    logging.info("Create train dataset.")
    train_dataset = NerDataset(cfg.TRAIN, cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag, gold_tag = False, seg_tag = False, task = 'REL')
    logging.info("Create dev dataset.")
    dev_dataset = NerDataset(cfg.DEV, cfg.MODEL_TYPE, VOCAB, tag2idx, idx2tag,gold_tag = False, seg_tag = False, task = 'REL')
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
    fw_log = open(cfg.LOG, 'w')
    best_f1 = 0
    best_epoch = 0
    for epoch in range(0, cfg.N_EPOCH):  # 每个epoch对dev集进行测试
        train(model, train_iter, optimizer, criterion, scheduler, device)

        print(f"=========SAOKE eval at epoch={epoch}=========")
        fname = cfg.ckpt_dir
        precision, recall, f1 = dev(model, dev_iter, fname, device,tag2idx, idx2tag, cfg)
        if not os.path.isdir(cfg.ckpt_dir):
            os.makedirs(cfg.ckpt_dir)
        # save each check point
        torch.save({
            'epoch': epoch,
            'best': (best_f1, best_epoch),
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'sched': scheduler.state_dict(),
        }, cfg.ckpt_dir + str(epoch) + '.pt')
        
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
                'precision: ' + str(precision) + ' ' + 
                'recall: ' + str(recall) + ' ' + 
                'f1: ' + str(f1) + '\n'
                'best_f1: ' + str(best_f1) + ' ' +
                'best_epoch: ' + str(best_epoch) +
                '\n'
        ) 
        print("weights were saved to" + cfg.LAST)

    # print(f"=========SAOKE test=========")
    # checkpoint_PATH = '/home/weidu/data/OPENIE/OIE/CN/SAOKE/+grad/SAOKE/' + str(38) +'.pt'

    # model_CKPT = torch.load(checkpoint_PATH)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = Bert_BiLSTM_CRF(tag2idx, 'hfl/chinese-bert-wwm-ext')

    # model.load_state_dict(model_CKPT)
    # model.cuda()
    # if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
    # fname = os.path.join(hp.logdir, str(1))
    # precision, recall, f1 = eval(model, eval_SAOKE_iter, fname, device)
    # fw_SAOKE.write(
    #     'precision: ' + str(precision) + ' ' + 
    #     'recall: ' + str(recall) + ' ' + 
    #     'f1: ' + str(f1) + 
    #     '\n'
    # )
    print(best_f1, best_epoch)
    fw_log.close()

if __name__ == "__main__":
    main()
