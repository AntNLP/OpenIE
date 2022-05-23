import argparse
import os
import sys
sys.path.append('../')

import torch.nn as nn
import logging
import torch
from torch.utils import data
from utils import vocabs
from utils.antu.io.configurators import IniConfigurator
from utils.oie_eval.oie_readers.goldReader import GoldReader
from utils.ner_dataset import NerDataset, pad
from utils.oie_eval.carb import Benchmark
from utils.oie_eval.oie_readers.seqReader import SeqReader
from utils.oie_eval.matcher import Matcher


logging.basicConfig(level = logging.INFO)

def eval(output_file, gold_file):
    """Function for evaluation."""
    outfile_path = output_file
    gold_path = gold_file
    out_filename = './out'
    benchmark = Benchmark(gold_path)
    predicted = GoldReader()
    predicted.read(outfile_path)
    matching_func = Matcher.binary_linient_tuple_match
    logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
    error_file = './error_log'
    auc, optimal_f1_point= benchmark.compare(  predicted = predicted.oie,
                                        matchingFunc = matching_func,
                                        output_fn = out_filename,
                                        error_file = error_file)
    precision, recall, f_1 = optimal_f1_point[0], optimal_f1_point[1], optimal_f1_point[2]
    print("AUC:{:.5f}, P:{:.5f}, R:{:.5f}, F1:{:.5f}".format(auc,precision, recall, f_1))
    return auc, precision, recall, f_1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Usage for OPENIE.")
    parser.add_argument('--OUTPUT', type=str, help="Path to output file.")
    parser.add_argument('--GOLD', type=str, help="Path to gold file.")
    args = parser.parse_args()
    eval(args.OUTPUT, args.GOLD)
