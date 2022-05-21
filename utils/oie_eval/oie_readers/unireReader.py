""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE [--debug]

Convert to tabbed format
"""
# External imports
import json

# Local imports
from .oieReader import OieReader
from .extraction import Extraction


class UnireReader(OieReader):
    """ Read unire system output file, and transform it into unified format for eval
    """
    def __init__(self):
        self.name = 'UnireReader'
        self.extraction_num = 0

    def read(self, fn):
        d = {}
        lines = []
        with open(fn) as fin:
            for line in fin:
                data = json.loads(line.strip())
                text =  data['Sentence']
                curExtraction = Extraction(pred = data['Predicate']['text'],
                                           head_pred_index = -1,
                                           sent = text,
                                           confidence = 1.0,
                                           index = self.extraction_num)
                for arg in data['Arguments']:
                    curExtraction.addArg(arg['text'])

                # remove repeated extractions 
                if line not in lines:
                    lines.append(line)
                    d[text] = d.get(text, []) + [curExtraction]
                    self.extraction_num += 1
                
        self.oie = d
