from models.sequence_labeling import SeqIE
from sched import scheduler
from utils.antu.io.configurators import IniConfigurator
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
# parser = argparse.ArgumentParser(description="Usage for OPENIE.")
# parser.add_argument('--CFG', type=str, help="Path to config file.")
# parser.add_argument('--DEBUG', action='store_true', help="DEBUG mode.")
# args, extra_args = parser.parse_known_args()
# cfg = IniConfigurator(args.CFG, extra_args)
# if not os.path.exists(cfg.ckpt_dir):
#     os.makedirs(cfg.ckpt_dir)
# 
# VOCAB = vocab.get_vocab(cfg.DOMAIN)
# tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
# idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
# model = SeqIE().get_model(tag2idx, idx2tag, cfg, device)
def main():
    parser = argparse.ArgumentParser(description="Usage for OPENIE.")
    parser.add_argument('--CFG', type=str, help="Path to config file.")
    parser.add_argument('--DEBUG', action='store_true', help="DEBUG mode.")
    args, extra_args = parser.parse_known_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = IniConfigurator(args.CFG, extra_args)
    model = SeqIE(cfg, device)
    model.model_train()
if __name__ == "__main__":
    main()