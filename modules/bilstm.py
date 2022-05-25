"""
a module for create bilstm model
"""
import torch.nn as nn
class BiLSTM(nn.Module):
    """a bilstm model class"""
    def __init__(self, hidden_dim, tagset_size):
        # super(BiLSTM, self).__init__()
        super().__init__()
        self.tagset_size = tagset_size
        self.tagset_size = tagset_size
        self.lstm = nn.LSTM(bidirectional=True,
                            num_layers=2,
                            input_size=768,
                            hidden_size=hidden_dim//2,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, self.tagset_size)

    def forward(self, embeds):
        """return result of model"""
        enc, _ = self.lstm(embeds)
        enc = self.fc1(enc)
        return enc
