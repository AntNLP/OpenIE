import logging
from .matcher import Matcher
from .benchmark import Benchmark
from .oie_readers.unireReader import UnireReader
from .oie_readers.tabReader import TabReader
from .oie_readers.carbReader import CaRBReader

logger = logging.getLogger(__name__)

def eval_file(predicted_file,
              gold_file,
              arch="Unire",
              eval_dataset="Oie16",
              match="lexicalMatch",
              compare="Oie16",
              recall_strategy="m2o",
              out_file="out.txt"):    
    """
    Args
        predicted_file (str): predicted file path
        gold_file (str): gold file path
        arch (str): current model architecture for oie task
        eval_dataset (str): dataset for eval, such as Oie16/CaRB
        match (str): matching function between the predicted and the gold extraction
        compare (str): compare method for eval, such as Oie16/CaRB
        recall_strategy (str): recall calculation strategy for CaRB compare method, such as m2o (many-to-one) and o2o (one-to-one)
        out_file (str): output file of the p-r curve points
    """
    if arch == 'Unire':
        predicted = UnireReader()
        predicted.read(predicted_file)
    '''
    ####################Add code here!
    
    First, define a new reader under oie_readers/, for another model's predicted results
    Then call this reader to read the predicted_file
    '''

    if eval_dataset == 'CaRB':
        gold_reader = CaRBReader
    else:
        gold_reader = TabReader
    b = Benchmark(gold_reader, gold_file)
    
    logger.info("arch={}, matchingFunc={}, eval_dataset={}, compare={}, recall_strategy={}".format(arch, match, eval_dataset, compare, recall_strategy)) 
    logger.info("Writing PR curve of {} to {}".format(predicted.name, out_file))
    
    matchingFunc = eval('Matcher.'+match)
    if compare == "CaRB":
        compare_args = "(predicted = predicted.oie, output_fn = out_file, matchingFunc = matchingFunc, recall_strategy = recall_strategy)"
    else:
        compare_args = "(predicted = predicted.oie, output_fn = out_file, matchingFunc = matchingFunc)" 
    
    auc, optimal_f1_point = eval("b." + compare + '_compare' + compare_args)
    
    print("AUC: {}\t Optimal (precision, recall, F1): {}".format( auc, optimal_f1_point ))
    
    logger.info("p: {:3.3f}%".format(100 * optimal_f1_point[0]))
    logger.info("r: {:3.3f}%".format(100 * optimal_f1_point[1]))
    logger.info("f1: {:3.3f}%".format(100 * optimal_f1_point[2]))
    logger.info("auc: {:3.3f}%".format(100 * auc))

    return auc, optimal_f1_point
