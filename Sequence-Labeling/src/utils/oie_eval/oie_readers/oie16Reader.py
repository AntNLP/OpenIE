from .oieReader import OieReader
from .extraction import Extraction
from _collections import defaultdict

class Oie16Reader(OieReader):
    """
    Read gold data from oie16 files
    """

    # Path relative to repo root folder    
    def __init__(self):
        self.name = 'Oie16'
    
    def add_sent_extraction(self, d, words, args):
        text = ' '.join(words)
        for label, arg in args.items():
            args[label] = ' '.join(arg)
        rel = args['P']
        curExtraction = Extraction(pred = rel, sent = text,
                                   head_pred_index = -1, confidence = 1.0)
        del args['P']
        for arg in args.values():
            curExtraction.addArg(arg)
        
        d[text].append(curExtraction)

    def read(self, fn):
        d = defaultdict(list)
        with open(fn) as fin:
            line1 = fin.readline()
            assert line1.startswith("word_id"), f"OpenIE:[source_file] should start with data_field names."
            
            # obtain related data fields ids        
            fields = line1.strip().split('\t')
            word_field_id = fields.index('word')
            label_field_id = fields.index('label')

            args, words = defaultdict(list), []
            for line in fin.readlines():
                line = line.strip()
                if len(line) == 0 and len(words) > 0:
                    # process an instance                
                    self.add_sent_extraction(d, words, args)
                    args, words = defaultdict(list), []
                else:
                    word_data = line.split('\t')
                    assert len(word_data) == len(fields), "word_data doesn't match with fields."
                    word = word_data[word_field_id]
                    words.append(word)
                    word_label = word_data[label_field_id][:2].strip('-')
                    if word_label != 'O':
                        args[word_label].append(word)
            
            if len(words) > 0:
                self.add_sent_extraction(d, words, args)
        
        self.oie = d

if __name__ == '__main__' :
    g = Oie16Reader()
    g.read('../data/OIE2016/origin/test.oie.conll')
    g.output_tabbed("../data/OIE2016/eval/test.txt")