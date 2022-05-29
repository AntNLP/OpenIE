import argparse
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# sys.path.append('../')

import torch.nn as nn
import logging
import torch
from utils import vocabs
from utils.antu.io.configurators import IniConfigurator
from .oie_eval.oie_readers.goldReader import GoldReader
from .oie_eval.carb import Benchmark
from .oie_eval.matcher import Matcher


logging.basicConfig(level = logging.INFO)

def eval(cfg, test_type):
    """Function for evaluation."""
    
    if test_type == 'test':
        gold_path = cfg.TEST_GOLD
        output_fn = cfg.TEST_OUT
        outfile_path = cfg.TEST_OUTPUT
        error_file = cfg.TEST_ERROR_LOG
    elif test_type == 'dev':
        gold_path = cfg.DEV_GOLD
        output_fn = cfg.DEV_OUT
        outfile_path = cfg.DEV_OUTPUT
        error_file = cfg.DEV_ERROR_LOG
    benchmark = Benchmark(gold_path)
    predicted = GoldReader()
    predicted.read(outfile_path)
    matching_func = Matcher.binary_linient_tuple_match
    logging.info("Writing PR curve of {} to {}".format(predicted.name, output_fn))
    auc, optimal_f1_point= benchmark.compare(  predicted = predicted.oie,
                                        matchingFunc = matching_func,
                                        output_fn = output_fn,
                                        error_file = error_file)
    precision, recall, f_1 = optimal_f1_point[0], optimal_f1_point[1], optimal_f1_point[2]
    print("AUC:{:.5f}, P:{:.5f}, R:{:.5f}, F1:{:.5f}".format(auc,precision, recall, f_1))
    return auc, precision, recall, f_1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Usage for OPENIE evaluation.")
    parser.add_argument('--CFG', type=str, help="Path to config file.")
    parser.add_argument('--DEBUG', action='store_true', help="DEBUG mode.")
    args, extra_args = parser.parse_known_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = IniConfigurator(args.CFG, extra_args)
    args = parser.parse_args()
    eval(cfg, 'test')
