from base64 import encode
# from ctypes.wintypes import tagSIZE
from re import S
from attr import frozen
import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding
from torchsummary import summary
import module.bilstm as bilstm
# from module.crf import CRF
from module.encoder import Encoder
from torchcrf import CRF

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, encmodel, hidden_dim=768, seg_num = 7):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hidden_dim = hidden_dim
        self.seg_num = seg_num
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.crf = CRF(num_tags=self.tagset_size,batch_first=True)

        self.start_label_id = self.tag_to_ix['[CLS]']
        self.end_label_id = self.tag_to_ix['[SEP]']


        self.bilstm = bilstm.BiLSTM(hidden_dim, self.tagset_size)

        self.encoder = Encoder(frozen = False, encoder_model = encmodel, seg_num = self.seg_num, hidden_dim = self.hidden_dim)
        self.features_in_hook = []
        self.features_out_hook = []
        self.features = []
        


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))



    def neg_log_likelihood(self, sentence, tags, seg, mask = None):
        embeds = self.encoder(sentence, seg)  # [8, 75, 768]
        feats = self.bilstm(embeds)  #[batch_size, max_len, 16]
        return -self.crf(emissions=feats, tags = tags, mask = None, reduction = 'mean',)



    def forward(self, sentence, seg):  # dont confuse this with _forward_alg above.

        embeds = self.encoder(sentence, seg)

        lstm_feats = self.bilstm(embeds)

        # score, tag_seq = self.crf(lstm_feats)
        score, tag_seq = None, self.crf.decode(emissions=lstm_feats)
        return score, torch.tensor(tag_seq, dtype=torch.long)
