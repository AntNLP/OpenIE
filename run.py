# pylint: disable=import-error
import argparse
import logging
import os
import time

from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import torch
from torch.utils import data
from tqdm import tqdm

from utils.antu.io.configurators import IniConfigurator
from utils.dataset import NerDataset, pad
from eval.evaluation import eval
from models.seqie import SeqIE
from utils.tagset import TagSet

logging.basicConfig(level = logging.INFO)
def get_predicate_span(pre_tags_list, gold_tags_list, pfla, tagset):
    """Function for obtaining the predicating span.
    
    Args:
        pre_tags_list:  A two-dimension list of predicating tags.
        gold_tags_list: A two-dimension list of ground truth tags.
        pfla:           The mode for learning predicating arguments.
        tagset:         A Tagset class.
    """
    def tag2span(tags_list):
        spans = []
        for tags in tags_list:
            l = r = -1
            for idx, tag in enumerate(tags):
                if tag == tagset.predicate_tag_B :
                    l = r = idx
                elif tag == tagset.predicate_tag_I :
                    r = idx
                elif l != -1 and [l,r] not in spans:
                    spans.append([l, r])
                    l = r = -1
        return spans
    spans = []
    gold_span_list = tag2span(gold_tags_list)
    pre_span_list = tag2span(pre_tags_list)
    if pfla == 'soft':
        for gold_span in gold_span_list:
            for pre_span in pre_span_list:
                #三种相交情况
                l_pre,  r_pre   =  pre_span[0],     pre_span[1]
                l_gold, r_gold  =  gold_span[0],    gold_span[1]
                if  (l_pre <= l_gold <= r_pre) or \
                    (l_pre <= r_gold <= r_pre) or \
                    (l_pre>=l_gold and r_pre<=r_gold) :
                    spans.append([l_gold, r_gold])
    elif pfla == 'gold':
        spans = [sp for sp in gold_span_list]
    else:
        spans = [sp for sp in pre_span_list]
    return spans

def get_predicate_tags(spans, tags_size, limit, tagset):
    pre_tags_list = []
    cnt = 0
    for (l, r) in spans:
        cnt += 1
        pre_tags = []
        for idx in range(tags_size):
            if l == idx:
                pre_tags.append(tagset.predicate_tag_B)
            elif l<idx<=r :
                pre_tags.append(tagset.predicate_tag_I)
            else:
                pre_tags.append('O')
        pre_tags_list.append(pre_tags)
        if cnt == limit:
            break
    while len(pre_tags_list) < limit :
        pre_tags_list.append(['O']*tags_size)
    return pre_tags_list

def create_train_tmp_data(_model, fname, iterator, tagset, device, cfg):
    """Function for create tmp train data in pipeline method."""
    model = _model
    model = model.eval()
    model.to(device)
    fw = open(fname,'w')
    idx2tag = tagset.get_idx2tag()
    for i, batch in enumerate(iterator):
        words_list, x, is_heads_list, tags, y, seqlens, segs_list, exts_list, p_tags, att_masks = batch
        x = x.to(device)
        y = y.to(device)
        segs_list = segs_list.to(device)
        att_masks = att_masks.to(device)
        _y = y 
        _, y_hats_list = model(x, segs_list, att_masks)
        x = x.to(device)
        
        for idx, (words, y_hats, is_heads, exts, segs) in enumerate(zip(words_list, y_hats_list, is_heads_list, exts_list, segs_list)):
            y_h = [hat for head, hat in zip(is_heads, y_hats.cpu().numpy().tolist()) if head == 1]
            preds = [idx2tag[hat] for hat in y_h]
            ext_tags_list = []
            for ext in exts:
                ext_tags = [idx2tag[e] for head, e in zip(is_heads, ext) if head == 1]
                ext_tags_list.append(ext_tags)
            assert(len(preds) == len(words))
            spans = get_predicate_span([preds], ext_tags_list, cfg.PREDICATE_FOR_LEARNING_ARGUMENT, tagset)
            pre_tags_list = get_predicate_tags(spans, len(preds), len(exts), tagset)
            for ext_tags, pre_tags in zip(ext_tags_list, pre_tags_list):
                for w, p, s in zip(words[1:-1], ext_tags[1:-1], pre_tags[1:-1]):
                    fw.write(w + '\t' + p + '\t' + s  + '\n')
                fw.write('\n')
    fw.close()

def create_test_tmp_data(_model, fname, iterator, tagset, device, cfg):
    """Function for create tmp test data in pipeline method."""
    model = _model
    model = model.eval()
    model.to(device)
    fw = open(fname,'w')
    idx2tag = tagset.get_idx2tag()
    for i, batch in enumerate(iterator):
        words_list, x, is_heads_list, tags, y, seqlens, seg, exts_list, p_tags, att_masks = batch
        x = x.to(device)
        y = y.to(device)
        seg = seg.to(device)
        att_masks = att_masks.to(device)
        _y = y 
        _, y_hats_list = model(x, seg, att_masks)
        x = x.to(device)
        seg = seg.to(device)
        for idx, (words, y_hats, is_heads, exts) in enumerate(zip(words_list, y_hats_list, is_heads_list, exts_list)):
            y_h = [hat for head, hat in zip(is_heads, y_hats.cpu().numpy().tolist()) if head == 1]
            preds = [idx2tag[hat] for hat in y_h]
            assert(len(preds) == len(words))
            ext_tags_list = []
            for ext in exts:
                ext_tags = [idx2tag[e] for head, e in zip(is_heads, ext) if head == 1]
                ext_tags_list.append(ext_tags)
            assert(len(preds) == len(words))
            spans = get_predicate_span([preds], ext_tags_list, None, tagset)
            pre_tags_list = get_predicate_tags(spans, len(preds), min(cfg.PREDICATE_LIMIT, len(spans)), tagset)
            for pre_tags in pre_tags_list:
                for w, s in zip(words[1:-1],  pre_tags[1:-1]):
                    fw.write(w + '\t' + 'O' + '\t' + s  + '\n')
                fw.write('\n')
    fw.close()

def write_output(preds, texts, output):
    """Function for outputting the results into file in form of column file.

    Args:
        preds:  A list of tags the model predicated ([CLS] and [SEP] are the 
                first one and the last one respectively).
        testxs: A list of the words in sentences ([CLS] and [SEP] are the 
                first one and the last one respectively).
        output: The output file path.
    """
    cnt = 0
    tags = preds[1:-1]
    predicates = []
    args = []
    for tag, text in zip(tags, texts):
        if tag in ['P-B', 'P-I']:
            predicates.append(text)
    predicate = " ".join(predicates)
    if len(predicate) == 0:
        predicate = ' '
    output.write(" ".join(texts) + '\t' + predicate)
    for pos in ['0', '1', '2', '3']:
        args = []
        for tag, text in zip(tags, texts):
            if(len(tag) >= 2 and tag[1] == pos):
                args.append(text)
        if len(args) != 0:
            output.write('\t' + " ".join(args))
    output.write('\n')
    cnt += 1
    return cnt

def test(_model, cfg, _iter, test_type):
    """Function for model developing and testing.

    Args:
        _model:     A model to be tested.
        cfg:        A config file for setting model parameters.
        _iter:      A dataset loader.
        test_type:  To choose different running mode('dev' or 'test').

    Returns:
        4 float variates for the test results including accuracy, precision, recall, f1.
        example:

        0.315 0.609 0.391 0.476
    """
    model = _model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    if test_type == 'dev':
        outfile_path = cfg.DEV_OUTPUT
    elif test_type == 'test':
        outfile_path = cfg.TEST_OUTPUT
    print("Starting Loading Data...")
    tagset = TagSet(cfg)
    tag2idx = tagset.get_tag2idx()
    idx2tag = tagset.get_idx2tag()
    print("Loading Data Ended")
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        # iterator = tqdm(enumerate(_iter), desc='steps', total=len(_iter))
        # for i, batch in iterator:
        for i, batch in enumerate(_iter):
            words, inputs, is_heads, tags, y, seqlens, seg, ext, p_tags, att_masks = batch
            inputs = inputs.to(device)
            seg = seg.to(device)
            att_masks = att_masks.to(device)
            _, y_hat = model(inputs, seg, att_masks)
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
                    cnt += write_output(preds, texts, output)
        elif cfg.METHOD == 'pipeline':
            for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
                y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
                preds = [idx2tag[hat] for hat in y_hat]
                texts = words[1:-1]
                cnt += write_output(preds, texts, output)
        output.close()
    if cnt == 0:
        auc,precision, recall, f1 =  0, 0, 0, 0
        print("AUC:{:.5f}, P:{:.5f}, R:{:.5f}, F1:{:.5f}".format(auc,precision, recall, f1))
    else:
        auc, precision, recall, f1 = eval(cfg, test_type)
    return auc, precision, recall, f1

def train(cfg, _iter, test_iter, device, optimizer, scheduler, model):
    """Function for training in single epoch"""
    t_dataset = NerDataset(cfg.TEST, cfg, False, False)
    t_iter    = data.DataLoader(dataset=t_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=pad
                                    )
    model.train()
    iter = _iter
    iterator = tqdm(enumerate(iter), desc='steps', total=len(iter))
    for i, batch in iterator:
        model.train()
        st = time.perf_counter()
        words, x, is_heads, tags, y, seqlens, seg, ext, p_tags, att_masks = batch
        x = x.to(device)
        y = y.to(device)
        att_masks = att_masks.to(device)
        seg = seg.to(device)
        _y = y # for monitoring
        optimizer.zero_grad()
        if cfg.METHOD == 'joint':
            loss = model.neg_log_likelihood(x, y, seg, ext, att_masks) # logits: (N, T, VOCAB), y: (N, T)
        elif cfg.METHOD == 'pipeline':
            loss = model.neg_log_likelihood(x, y, seg, att_masks) # logits: (N, T, VOCAB), y: (N, T)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ed = time.perf_counter()
        #print(ed-st)
        if i==0:
            print("=====sanity check======")
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("seg:", seg.cpu().numpy()[0][:seqlens[0]])
            print("=======================")
        if i%1000==0 and i != 0: # monitoring
            model.eval()
            print(f"step: {i}, loss: {loss.item()}")
            print('dev:')
            auc, precision, recall, f1 = test(model, cfg, test_iter, 'dev')
            # print(auc, precision, recall, f1)
            print('test:')
            auc, precision, recall, f1 = test(model, cfg, t_iter, 'test')
            # print(auc, precision, recall, f1)

def model_save(cfg, fw_log, best, res, model_setting):
    """function for saving model information"""
    best_f1, best_epoch = (ele for ele in best)
    auc, precision, recall, f1 = (ele for ele in res)
    model, optimizer, scheduler, epoch = (ele for ele in model_setting)
    if epoch < cfg.CKPT_LIMIT:
        return best_f1, best_epoch
    torch.save({
            'epoch': epoch,
            'best': (best_f1, best_epoch),
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'sched': scheduler.state_dict(),
        }, cfg.LAST)
    if float(f1) > float(best_f1):
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
            '\n')
    print(  'n_epcch: ' + str(epoch) + ' ' +
            'auc: ' + str(auc) + ' ' +
            'precision: ' + str(precision) + ' ' +
            'recall: ' + str(recall) + ' ' +
            'f1: ' + str(f1) + ' ' +
            'best_f1: ' + str(best_f1) + ' ' +
            'best_epoch: ' + str(best_epoch) +
            '\n')
    return best_f1, best_epoch

def train_joint(_model, cfg): # model -> output
    """Function for joint model training."""
    print("Starting Loading Data...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    total_steps = len(train_iter)*epochs
    if not os.path.isdir(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)
    fw_log = open(cfg.LOG, 'w')
    print("train steps:" + str(total_steps))
    scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps = len(train_iter)*cfg.WARM_UP_STEPS,
                num_training_steps = total_steps)
    print('Start Train...,')
    best_f1 = -1
    best_epoch = 0
    for epoch in tqdm(range(1, cfg.N_EPOCH + 1), desc='epochs'):
        print(f"=========train at epoch={epoch}=========")
        train(cfg, train_iter, test_iter, device, optimizer, scheduler, model)
        print(f"=========dev at epoch={epoch}=========")
        auc, precision, recall, f1 = test(model, cfg, test_iter, 'dev')
        res = [auc, precision, recall, f1]
        model_setting = [model, optimizer, scheduler, epoch]
        best_f1, best_epoch = model_save(cfg, fw_log, [best_f1, best_epoch], res, model_setting)
    # save current model
    ckpt = torch.load(cfg.BEST)
    #ckpt = torch.load(cfg.LAST)
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
    auc, precision, recall, f1 = test(model, cfg, test_iter, 'test')
    print('Test Ended')
    fw_log.close()

def train_pipeline(_model_pre, _model_arg, cfg): # model1 -> output1 -> model2 -> output2
    """Function for training pipeline model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tagset = TagSet(cfg)
    epochs = cfg.N_EPOCH
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
    scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps = len(train_iter)*cfg.WARM_UP_STEPS,
                num_training_steps=total_steps)
    if not os.path.isdir(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)
    print("train steps:" + str(total_steps))
    print('Start pre_train...,')
    fw_log = open(cfg.LOG, 'w')
    if not cfg.CHECK_POINT:
        for epoch in range(0, cfg.N_EPOCH+1):
            train(cfg, train_iter, test_iter, device, optimizer, scheduler, model_pre)
            if epoch % 50 == 0:
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
    create_train_tmp_data(model_pre, cfg.TRAIN_TMP, train_iter, tagset, device, cfg)
    create_test_tmp_data(model_pre, cfg.DEV_TMP, test_iter,  tagset, device, cfg)
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
    scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps = len(train_iter)*cfg.WARM_UP_STEPS,
                num_training_steps = total_steps)
    for epoch in range(0, cfg.N_EPOCH):
        print(f"=========train at epoch={epoch}=========")
        train(cfg, train_iter, test_iter, device, optimizer, scheduler, model_arg)
        print(f"=========dev at epoch={epoch}=========")
        # save current model
        auc, precision, recall, f1 = test(model_arg, cfg, test_iter, 'dev')
        res = [auc, precision, recall, f1]
        model_setting = [model_arg, optimizer, scheduler, epoch]
        best_f1, best_epoch = model_save(cfg, fw_log, [best_f1, best_epoch], res, model_setting)
    ckpt = torch.load(cfg.BEST)
    model = model_arg
    model.load_state_dict(ckpt['model'])
    optimizer = optim.Adam(model.parameters(), lr = float(cfg.LR))
    optimizer.load_state_dict(ckpt['optim'])
    print('Start Test...')
    
    test_dataset = NerDataset(cfg.TEST, cfg, False, True)
    test_iter  = data.DataLoader(dataset=test_dataset,
                                        batch_size=cfg.N_BATCH,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=pad
                                        )
    create_test_tmp_data(model_pre, cfg.TEST_TMP, test_iter,  tagset, device, cfg)
    test_dataset = NerDataset(cfg.TEST_TMP, cfg, False, True)
    test_iter  = data.DataLoader(dataset=test_dataset,
                                    batch_size=cfg.N_BATCH,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=pad
                                    )
    auc, precision, recall, f1 = auc, precision, recall, f1 = test(model, cfg, test_iter, 'test')
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
    torch.manual_seed(cfg.SEED)
    if cfg.METHOD == 'joint':
        train_joint(seqie.get_model(), cfg)
    elif cfg.METHOD == 'pipeline':
        model_pre, model_arg = seqie.get_model()
        train_pipeline(model_pre, model_arg, cfg)
if __name__ == "__main__":
    main()
