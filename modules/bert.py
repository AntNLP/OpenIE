"""
a module for create a bert you want
"""
import torch.nn as nn
from transformers import BertModel
import torch.nn as nn
class Bert(nn.Module):
    """a bert model class"""
    def __init__(self, cfg):
        super().__init__()
        self.bert = BertModel.from_pretrained(cfg.BERT_TYPE)
        self.bert.output_hidden_states=True

    def forward(self, x):
        """return result of bert"""
        # if(self.frozen):
        #     with torch.no_grad():
        #     # handle = self.bert.bert.register_forward_hook(self.hook)
        #     # encoded_layer = self.bert(x)
        #     # enc = self.features
        #     # handle.remove()
        #         encoded_layer = self.encoder(x)
        # else:
        encoded_layer = None
        encoded_layer = self.bert(x)
        embeds = encoded_layer[0]
        return embeds
