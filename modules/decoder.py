import torch.nn as nn
import torch
from torchcrf import CRF
from .bilstm import BiLSTM
from transformers import BertModel
class Decoder(nn.Module):
    def __init__(self, hidden_dim, tagset_size, cfg):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        if cfg.DECODER == 'bilstm-crf':
            self.crf = CRF(num_tags=self.tagset_size,batch_first=True)
            self.bilstm = BiLSTM(self.hidden_dim, self.tagset_size)
    def loss(self, embeds, tags):
        feats = self.bilstm(embeds)  #[batch_size, max_len, 16]
        return -self.crf(emissions=feats, tags = tags, mask = None, reduction = 'mean',)
    def forward(self, embeds):  # dont confuse this with _forward_alg above.
        lstm_feats = self.bilstm(embeds)
        score, tag_seq = None, self.crf.decode(emissions=lstm_feats)
        return score, torch.tensor(tag_seq, dtype=torch.long)
    def reset_parameter(self):
        pass