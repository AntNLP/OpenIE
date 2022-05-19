import torch.nn as nn
import torch
from transformers import BertModel
class Encoder(nn.Module):
    def __init__(self,  frozen, encoder_model, seg_num, hidden_dim):
        super(Encoder, self).__init__()
        self.frozen = frozen
        self.encoder = BertModel.from_pretrained(encoder_model)
        self.encoder.output_hidden_states=True

        self.norm = nn.LayerNorm(hidden_dim);
        self.emb = nn.Embedding(seg_num, hidden_dim)
        # if(frozen == True):
        #     self.encoder.eval()
        # self.fc = nn.Linear(hidden_dim, self.tagset_size)
    def hook(self, module, input, output):
        self.features = output.last_hidden_state.clone()
    def forward(self, x, seg):  # dont confuse this with _forward_alg above.
        encoded_layer = None
        # if(self.frozen):
        #     with torch.no_grad():
        #     # handle = self.bert.bert.register_forward_hook(self.hook)
        #     # encoded_layer = self.bert(x)
        #     # enc = self.features
        #     # handle.remove()
        #         encoded_layer = self.encoder(x)
        # else:
        encoded_layer = self.encoder(x)
        embeds = encoded_layer[0]
        embeds = self.norm(embeds + self.emb(seg))
        return embeds