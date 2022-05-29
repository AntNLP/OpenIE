class TagSet():
    def __init__(self, cfg):
        self.vocab = None
        self.predicate_tag_B = None
        self.predicate_tag_I = None
        if cfg.TAG_SET_TYPE == 'oie2016':
            self.vocab = (  '<PAD>',    '[CLS]',    '[SEP]',    'O',    'A0-B', 
                            'A1-B',     'A2-B',     'A3-B',     'A4-B', 'A5-B',
                            'A0-I',     'A1-I',     'A2-I',     'A3-I', 'A4-I', 
                            'A5-I',     'P-B',      'P-I')
            self.predicate_tag = ['P-B', 'P-I']
            self.predicate_tag_B = 'P-B'
            self.predicate_tag_I = 'P-I'
            self.argument_tag  = [  'A0-B', 'A1-B', 'A2-B', 'A3-B', 'A4-B', 'A5-B',
                                    'A0-I', 'A1-I', 'A2-I', 'A3-I', 'A4-I', 'A5-I']
        elif cfg.TAG_SET_TYPE == 'SAOKE':
            self.vocab = (  '<PAD>', '[CLS]', '[SEP]', 
                            'O', 'BO', 'IO', 'BS', 
                            'IS', 'BR', 'IR')
            self.predicate_tag = ['BR', 'IR']
            self.predicate_tag_B = 'BR'
            self.predicate_tag_I = 'IR'
            self.argument_tag  = ['BO', 'IO', 'BS', 'IS']
        elif cfg.TAG_SET_TYPE == 'lsoie':
            self.vocab = (  '<PAD>', '[CLS]', '[SEP]', 
                            'O', 'A-B', 'A-I', 'P-B', 
                            'P-I')
            self.predicate_tag = ['P-B', 'P-I']
            self.predicate_tag_B = 'P-B'
            self.predicate_tag_I = 'P-I'
            self.argument_tag  = ['A-B', 'A-I']
        elif cfg.TAG_SET_TYPE == 'rel':
            self.vocab = (  '<PAD>', '[CLS]', '[SEP]', 
                            'O', 'P-B', 'P-I')
            self.predicate_tag = ['P-B', 'P-I']
            self.predicate_tag_B = 'P-B'
            self.predicate_tag_I = 'P-I'
            self.argument_tag  = []
        assert(self.vocab != None)
        assert(self.predicate_tag != None)
        assert(self.predicate_tag != None)
        self.tag2idx = dict(zip(self.vocab, range(len(self.vocab))))
        self.idx2tag = dict(zip(range(len(self.vocab)), self.vocab))

    def is_predicate_tag(self, tag):
        return tag in self.predicate_tag

    def is_predicate_tag_B(self, tag):
        return tag == self.predicate_tag_B
        
    def is_predicate_tag_I(self, tag):
        return tag == self.predicate_tag_I

    def is_argument_tag(self, tag):
        return tag in self.argument_tag

    def get_tag2idx(self):
        return self.tag2idx
    
    def get_idx2tag(self):
        return self.idx2tag

    def get_vocab(self):
        return self.vocab
    