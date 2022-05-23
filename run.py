# pylint: disable=import-error
import logging
import os
from models.seqie import SeqIE
from utils.antu.io.configurators import IniConfigurator
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from torchcrf import CRF
import torch
from utils.ner_dataset import NerDataset, pad
from torch.utils import data
from utils import vocabs
import argparse
from utils.oie_eval.carb import Benchmark
from utils.oie_eval.oie_readers.goldReader import GoldReader
from utils.oie_eval.matcher import Matcher
logging.basicConfig(level = logging.INFO)
def create_train_tmp_data(_model, fname, iterator, tag2idx, idx2tag, device, cfg):
    """Function for create tmp train data in pipeline method."""
    model = _model
    model = model.eval()
    model.to(device)
    fw = open(fname,'w')
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens, seg, ext, p_tags, att_mask = batch
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
            if(cfg.PREDICATE_FOR_LEARNING_ARGUMENT == 'soft'):
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
            elif(cfg.PREDICATE_FOR_LEARNING_ARGUMENT == 'gold'):
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
            for ex, pt in zip(ext[k], pre_tag):
                ex = [e for head, e in zip(is_heads[k], ex) if head == 1]
                for w, p, s in zip(words[k][1:-1], ex[1:-1], pt[1:-1]):
                    fw.write(w + '\t' + p + '\t' + s  + '\n')
                fw.write('\n')
    fw.close()
def create_test_tmp_data(_model, fname, iterator, tag2idx, idx2tag, device, cfg):
    """Function for create tmp test data in pipeline method."""
    model = _model
    model = model.eval()
    model.to(device)
    fw = open(fname,'w')
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens, seg, ext, p_tags, att_mask = batch
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
    fw.close()
def test(_model, cfg, _iter, error_file, outfile_path, gold_path, out_filename):
    """Function for model developing and testing."""
    model = _model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    print("Starting Loading Data...")
    vocab = vocabs.get_vocab(cfg.DOMAIN)
    tag2idx = dict(zip(vocab, range(len(vocab))))
    idx2tag = dict(zip(range(len(vocab)), vocab))
    print("Loading Data Ended")
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():

        for i, batch in enumerate(_iter):
            words, inputs, is_heads, tags, y, seqlens, seg, ext, p_tags, att_mask = batch
            inputs = inputs.to(device)
            seg = seg.to(device)
            _, y_hat = model(inputs, seg)
            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            if(cfg.METHOD == 'joint'):
                Y_hat.extend(y_hat)
            elif(cfg.METHOD == 'pipeline'):
                Y_hat.extend(y_hat.cpu().numpy().tolist())
    ## gets results and save
    cnt = 0
    with open(outfile_path, 'w', encoding='utf-8') as output:
        if cfg.METHOD == 'joint':
            for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
                for _y_hat in y_hat:
                    y_h = [hat for head, hat in zip(is_heads, _y_hat) if head == 1]
                    
                    preds = [idx2tag[hat] for hat in y_h]
                    texts = words[1:-1]
                    tags = preds[1:-1]
                    predicates = []
                    args = []
                    for tag, text in zip(tags, texts):
                        if tag in ['P-B', 'P-I']:
                            predicates.append(text)
                    predicate = " ".join(predicates)
                    if len(predicate) == 0 :
                        output.write(" ".join(texts) + '\t' + ' ')
                    else:
                        output.write(" ".join(texts) + '\t' + predicate)
                    for pos in ['0', '1', '2', '3']:
                        args = []
                        for tag, text in zip(tags, texts):
                            if(len(tag) >= 2 and tag[1] == pos):
                                args.append(text)
                        if(len(args) != 0):
                            output.write('\t' + " ".join(args))
                    output.write('\n')
                    cnt += 1
        elif(cfg.METHOD == 'pipeline'):
            for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
                y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
                preds = [idx2tag[hat] for hat in y_hat]
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
                output.write(text + '\t' + predicate)
                for pos in ['0', '1', '2', '3']:
                    cnt += 1
                    args = []
                    for i in range(len(tags)):
                        if(len(tags[i]) >= 2 and tags[i][1] == pos):
                            args.append(texts[i])
                    if(len(args) != 0):
                        output.write('\t' + " ".join(args))
                output.write('\n')
        output.close()
    if(cnt == 0):
        auc,precision, recall, f_1 =  0, 0, 0, 0
    else:
        auc, precision, recall, f_1 = compare(error_file, outfile_path, gold_path, out_filename)
    print("AUC:{:.5f}, P:{:.5f}, R:{:.5f}, F1:{:.5f}".format(auc,precision, recall, f_1))
    return auc, precision, recall, f_1
def compare(error_file, outfile_path, gold_path, out_filename):
    """"compare the output with gold data"""
    benchmark = Benchmark(gold_path)
    predicted = GoldReader()
    predicted.read(outfile_path)
    matching_func = Matcher.binary_linient_tuple_match
    logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
    auc, optimal_f1_point= benchmark.compare(  predicted = predicted.oie,
                                        matchingFunc = matching_func,
                                        output_fn = out_filename,
                                        error_file = error_file)
    precision, recall, f_1 = optimal_f1_point[0], optimal_f1_point[1], optimal_f1_point[2]
    return auc, precision, recall, f_1
def train(cfg, _iter, device, optimizer, scheduler, model):
    """Function for training in single epoch"""
    model.train()
    iter = _iter
    for i, batch in enumerate(iter):
        words, x, is_heads, tags, y, seqlens, seg, ext, p_tags, att_mask = batch
        x = x.to(device)
        y = y.to(device)
        seg = seg.to(device)
        _y = y # for monitoring
        optimizer.zero_grad()
        if cfg.METHOD == 'joint':
            loss = model.neg_log_likelihood(x, y, seg, p_tags, ext) # logits: (N, T, VOCAB), y: (N, T)
        elif cfg.METHOD == 'pipeline':
            loss = model.neg_log_likelihood(x, y, seg) # logits: (N, T, VOCAB), y: (N, T)
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

def train_joint(_model, cfg): # model -> output
    """Function for joint model training."""
    print("Starting Loading Data...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = vocabs.get_vocab(cfg.DOMAIN)
    tag2idx = dict(zip(vocab, range(len(vocab))))
    idx2tag = dict(zip(range(len(vocab)), vocab))
    train_dataset = NerDataset(cfg.TRAIN, cfg, cfg.TRAIN_GOLD_TAG, cfg.TRAIN_SEG_TAG)
    train_iter  = data.DataLoader(dataset=train_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=pad
                                        )
    test_dataset = NerDataset(cfg.DEV, cfg, cfg.DEV_GOLD_TAG, cfg.DEV_SEG_TAG)
    test_iter    = data.DataLoader(dataset=test_dataset,
                                            batch_size=cfg.N_BATCH,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=pad
                                            )
    print("Loading Data Ended")

    epochs = cfg.N_EPOCH
    model = _model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = float(cfg.LR))
    total_steps = len(test_iter)*epochs
    if not os.path.isdir(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)
    fw_log = open(cfg.LOG, 'w')
    print("train steps:" + str(total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter)*cfg.WARM_UP_STEPS, num_training_steps=total_steps)
    print('Start Train...,')
    best_f1 = -1
    best_epoch = 0
    for epoch in range(0, cfg.N_EPOCH):
        print(f"=========train at epoch={epoch}=========")
        train(cfg, train_iter, device, optimizer, scheduler, model)
        print(f"=========dev at epoch={epoch}=========")
        auc, precision, recall, f1 = test(model, cfg, test_iter,cfg.ERROR_LOG, cfg.OUTPUT, cfg.DEV_GOLD, cfg.OUT)
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
                'auc: ' + str(auc) + ' ' +
                'precision: ' + str(precision) + ' ' +
                'recall: ' + str(recall) + ' ' +
                'f1: ' + str(f1) + ' ' +
                'best_f1: ' + str(best_f1) + ' ' +
                'best_epoch: ' + str(best_epoch) +
                '\n'
        ) 
        print(  'n_epcch: ' + str(epoch) + ' ' +
                'auc: ' + str(auc) + ' ' +
                'precision: ' + str(precision) + ' ' +
                'recall: ' + str(recall) + ' ' +
                'f1: ' + str(f1) + ' ' +
                'best_f1: ' + str(best_f1) + ' ' +
                'best_epoch: ' + str(best_epoch) +
                '\n')
    
    ckpt = torch.load(cfg.BEST)
    model.load_state_dict(ckpt['model'])
    optimizer = optim.Adam(model.parameters(), lr = float(cfg.LR))
    optimizer.load_state_dict(ckpt['optim'])
    test_dataset = NerDataset(cfg.TEST, cfg, False, False)
    test_iter    = data.DataLoader(dataset=test_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=pad
                                    )
    print('Start Test...')
    auc, precision, recall, f1 = test(model, cfg, test_iter,cfg.ERROR_LOG, cfg.OUTPUT, cfg.TEST_GOLD, cfg.OUT)
    print('Test Ended')
    # test for test dataset
    fw_log.close()
def train_pipeline(_model_pre, _model_arg, cfg): # model1 -> output1 -> model2 -> output2
    """Function for training pipeline model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = vocabs.get_vocab(cfg.DOMAIN)
    epochs = cfg.N_EPOCH
    tag2idx = dict(zip(vocab, range(len(vocab))))
    idx2tag = dict(zip(range(len(vocab)), vocab))
    print("Starting Loading Data...")
    train_dataset = NerDataset(cfg.TRAIN, cfg, cfg.TRAIN_GOLD_TAG, cfg.TRAIN_SEG_TAG)
    train_iter  = data.DataLoader(dataset=train_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=pad
                                        )
    test_dataset = NerDataset(cfg.DEV, cfg, cfg.DEV_GOLD_TAG, cfg.DEV_SEG_TAG)
    test_iter  = data.DataLoader(dataset=test_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=pad
                                        )
    print("Loading Data Ended")

    model_pre = _model_pre.to(device)
    total_steps = len(train_iter)*epochs
    optimizer = optim.Adam(model_pre.parameters(), lr = float(cfg.LR))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter)*cfg.WARM_UP_STEPS, num_training_steps=total_steps)
    
    if not os.path.isdir(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)
    print("train steps:" + str(total_steps))
    print('Start pre_train...,')
    fw_log = open(cfg.LOG, 'w')
    if cfg.CHECK_POINT == False:
        for epoch in range(0, cfg.N_EPOCH+1): 
            train(cfg, train_iter, device, optimizer, scheduler, model_pre)
            if(epoch % 50 == 0):
                torch.save({
                    'epoch': epoch,
                    'model_pre': model_pre.state_dict(),
                    'optim_pre': optimizer.state_dict(),
                    'sched_pre': scheduler.state_dict(),
                }, cfg.PRE)
    print('pre_train Ended')
    ckpt = torch.load(cfg.PRE)
    
    model_pre.load_state_dict(ckpt['model_pre'])
    optimizer = optim.Adam(model_pre.parameters(), lr = float(cfg.LR))
    optimizer.load_state_dict(ckpt['optim_pre'])
    print('Start create_tmp_data...')
    create_train_tmp_data(model_pre, cfg.TRAIN_TMP, train_iter, tag2idx, idx2tag, device, cfg)
    create_test_tmp_data(model_pre, cfg.DEV_TMP, test_iter,  tag2idx, idx2tag, device, cfg)
    print('create_tmp_data Ended')

    print("Loading tmp Data Starting...")
    train_dataset = NerDataset(cfg.TRAIN_TMP, cfg, cfg.TRAIN_GOLD_TAG, True)
    train_iter  = data.DataLoader(dataset=train_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=pad
                                        )
    test_dataset = NerDataset(cfg.DEV_TMP, cfg, cfg.DEV_GOLD_TAG, True)
    test_iter  = data.DataLoader(dataset=test_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=pad
                                        )
    print("Loading tmp Data Ended")
    best_f1 = -1
    best_epoch = 0
    model_arg = _model_arg.to(device)
    total_steps = len(train_iter)*epochs
    optimizer = optim.Adam(model_arg.parameters(), lr = float(cfg.LR))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter)*cfg.WARM_UP_STEPS, num_training_steps=total_steps)
    for epoch in range(0, cfg.N_EPOCH):
        print(f"=========train at epoch={epoch}=========")
        train(cfg, train_iter, device, optimizer, scheduler, model_arg)
        print(f"=========dev at epoch={epoch}=========")
    auc, precision, recall, f1 = test(model_arg, cfg, test_iter, cfg.ERROR_LOG, cfg.OUTPUT, cfg.DEV_GOLD, cfg.OUT)
    # save current model
    torch.save({
        'epoch': epoch,
        'best': (best_f1, best_epoch),
        'model': model_arg.state_dict(),
        'optim': optimizer.state_dict(),
        'sched': scheduler.state_dict(),
    }, cfg.LAST)

    if(float(f1) > float(best_f1)):
        best_f1 = f1
        best_epoch = epoch
        torch.save({
        'epoch': epoch,
        'best': (best_f1, best_epoch),
        'model': model_arg.state_dict(),
        'optim': optimizer.state_dict(),
        'sched': scheduler.state_dict(),
    }, cfg.BEST)

    fw_log.write(
            'n_epcch: ' + str(epoch) + ' ' +
            'auc: ' + str(auc) + ' ' +
            'precision: ' + str(precision) + ' ' +
            'recall: ' + str(recall) + ' ' +
            'f1: ' + str(f1) + ' ' +
            'best_f1: ' + str(best_f1) + ' ' +
            'best_epoch: ' + str(best_epoch) +
            '\n'
    ) 
    print(  'n_epcch: ' + str(epoch) + ' ' +
            'auc: ' + str(auc) + ' ' +
            'precision: ' + str(precision) + ' ' +
            'recall: ' + str(recall) + ' ' +
            'f1: ' + str(f1) + ' ' +
            'best_f1: ' + str(best_f1) + ' ' +
            'best_epoch: ' + str(best_epoch) +
            '\n')
    ckpt = torch.load(cfg.BEST)
    model = model_arg
    optimizer = optim.Adam(model.parameters(), lr = float(cfg.LR))
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optim'])
    print('Start Test...')
    
    test_dataset = NerDataset(cfg.TEST, cfg, False, True)
    test_iter  = data.DataLoader(dataset=test_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=pad
                                        )
    create_test_tmp_data(model_pre, cfg.TEST_TMP, test_iter,  tag2idx, idx2tag, device, cfg)
    test_dataset = NerDataset(cfg.TEST_TMP, cfg, False, True)
    test_iter  = data.DataLoader(dataset=test_dataset,
                                    batch_size=cfg.N_BATCH,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=pad
                                    )
    auc, precision, recall, f1 = test(model_arg, cfg, test_iter, cfg.ERROR_LOG, cfg.OUTPUT, cfg.TEST_GOLD, cfg.OUT)
    print('Test Ended')
    # test for test dataset
    fw_log.close()
def main():
    """Function strating running."""
    parser = argparse.ArgumentParser(description="Usage for OPENIE.")
    parser.add_argument('--CFG', type=str, help="Path to config file.")
    parser.add_argument('--DEBUG', action='store_true', help="DEBUG mode.")
    args, extra_args = parser.parse_known_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = IniConfigurator(args.CFG, extra_args)
    seqie = SeqIE(cfg, device)
    if cfg.METHOD == 'joint':
        train_joint(seqie.get_model(), cfg)
    elif cfg.METHOD == 'pipeline':
        model_pre, model_arg = seqie.get_model()
        train_pipeline(model_pre, model_arg, cfg)
if __name__ == "__main__":
    main()
