"""a module for creating a decoder"""
import torch.nn as nn
import torch
from torchcrf import CRF
class Decoder(nn.Module):
    """a decoder class"""
    def __init__(self, hidden_dim, tagset_size, cfg):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        if cfg.DECODER == 'crf':
            self.crf = CRF(num_tags=self.tagset_size,batch_first=True)
    def loss(self, feats, tags):
        return -self.crf(emissions=feats, tags = tags, reduction = 'mean')
    def forward(self, feats):  
        score, tag_seq = None, self.crf.decode(emissions=feats)
        return score, torch.tensor(tag_seq, dtype=torch.long)
    def reset_parameter(self):
        pass