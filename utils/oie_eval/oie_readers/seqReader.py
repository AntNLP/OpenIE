""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE [--debug]

Convert to tabbed format
"""
# External imports
import logging
from pprint import pprint
from pprint import pformat
from docopt import docopt
from regex import W

# Local imports
from .oieReader import OieReader
from .extraction import Extraction
logger = logging.getLogger(__name__)
#=-----

class SeqReader(OieReader):

    def __init__(self):
        self.name = 'seqIE'

    def read(self, fn):
        logging.info("SeqReader start.")
        d = {}
        self.oie = d
        
        with open(fn) as fin:
            entries = fin.read().strip().split('\n\n')
            if(len(entries) == 0):
                return
            for entry in entries:
                cnt = 0
                aids = []
                rids = []
                rel = ''
                arg = ''
                args = []
                text = ''
                for line in entry.split('\n'):
                    if(len(line) == 0):
                        continue
                    w, t = line.split('\t')[0], line.split('\t')[-1]
                    text += w + ' '
                    if(t[-1] == 'B'):
                        if(t[0] == 'P'):
                            rel += w + ' '
                            rids.append(cnt)
                        elif(t[0] == 'A'):
                            if(len(arg) != 0):
                                tmp = (arg.strip(), aids)
                                args.append(str(tmp))
                                aids = []
                                arg = ''
                            arg += w + ' '
                            aids.append(cnt)
                    elif(t[-1] == 'I'):
                        if(t[0] == 'P'):
                            rel += w + ' '
                            rids.append(cnt)
                        elif(t[0] == 'A'):
                            arg += w + ' '
                            aids.append(cnt)
                    cnt += 1
                tmp = (arg.strip(), aids)
                args.append(str(tmp))
                tmp = (rel.strip(), rids)
                rel = [str(tmp)]
                text = text.strip()
                # print(rel[0])
                # print(args)
                curExtraction = Extraction(pred = rel[0], head_pred_index = -1, sent = text, confidence = float(1))
                for arg in args:
                    curExtraction.addArg(arg)
                d[text] = d.get(text, []) + [curExtraction]
                
                    # data = line.strip().split('\t')
                    # confidence = data[0]
                    # if not all(data[2:5]):
                    #     logging.debug("Skipped line: {}".format(line))
                    #     continue
                    # arg1, rel, arg2 = [s[s.index('(') + 1:s.index(',List(')] for s in data[2:5]]
                    # text = data[5]
                    # curExtraction = Extraction(pred = rel, head_pred_index = -1, sent = text, confidence = float(confidence))
                    # curExtraction.addArg(arg1)
                    #

        self.oie = d



# if __name__ == "__main__":
#     # Parse command line arguments
#     args = docopt(__doc__)
#     inp_fn = args["--in"]
#     out_fn = args["--out"]
#     debug = args["--debug"]
#     if debug:
#         logging.basicConfig(level = logging.DEBUG)
#     else:
#         logging.basicConfig(level = logging.INFO)


#     oie = SeqReader()
#     oie.read(inp_fn)
#     oie.output_tabbed(out_fn)

#     logging.info("DONE")
