def get_vocab(task):
    vocab = None
    if (task == 'oie2016'):
        vocab = (   '<PAD>',    '[CLS]',    '[SEP]',    'O',    'A0-B', 
                    'A1-B',     'A2-B',     'A3-B',     'A4-B', 'A5-B',
                    'A0-I',     'A1-I',     'A2-I',     'A3-I', 'A4-I', 
                    'A5-I',     'P-B',      'P-I')
    elif (task == 'SAOKE'):
        vocab = (   '<PAD>', '[CLS]', '[SEP]', 
                    'O', 'BO', 'IO', 'BS', 
                    'IS', 'BR', 'IR')
    elif (task == 'lsoie'):
        vocab = (   '<PAD>', '[CLS]', '[SEP]', 
                    'O', 'A-B', 'A-I', 'P-B', 
                    'P-I')
    elif (task == 'rel'):
        vocab = (   '<PAD>', '[CLS]', '[SEP]', 
                    'O', 'P-B', 'P-I')
    assert(vocab != None)
    return vocab
    