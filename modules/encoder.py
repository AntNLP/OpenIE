import torch.nn as nn
from transformers import BertModel
from .bilstm import BiLSTM
from .bert import Bert
class Encoder(nn.Module):
    def __init__(self,  frozen, seg_num, hidden_dim, tagset_size, cfg):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.frozen = frozen
        if cfg.ENCODER == 'bert-bilstm':
            self.bert = BertModel.from_pretrained('bert-base-cased')
            self.bert.output_hidden_states=True
            self.emb = nn.Embedding(seg_num, self.hidden_dim)
            self.norm = nn.LayerNorm(self.hidden_dim);
            self.bilstm = BiLSTM(self.hidden_dim, self.tagset_size)
        # if(frozen == True):
        #     self.encoder.eval()
        # self.fc = nn.Linear(hidden_dim, self.tagset_size)

    def forward(self, x, seg):  # dont confuse this with _forward_alg above.
        encoded_layer = None
        encoded_layer = self.bert(x)
        embeds = encoded_layer[0]
        embeds = self.norm(embeds + self.emb(seg))
        feats = self.bilstm(embeds)  #[batch_size, max_len, 16]
        return feats
    def reset_parameter(self):
        pass