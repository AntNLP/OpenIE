import torch.nn as nn
import torch
class BiLSTM(nn.Module):
    def __init__(self, hidden_dim, tagset_size):
        super(BiLSTM, self).__init__()
        self.tagset_size = tagset_size
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=hidden_dim//2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.tagset_size)
        # self.fc = nn.Linear(hidden_dim, self.tagset_size)
    def forward(self, embeds):  # dont confuse this with _forward_alg above.
        #embeds = self._bert_enc(sentence)  # [8, 75, 768]
        # embeds = self.norm(self.emb(seg) + embeds)
        # 过lstm
        enc, _ = self.lstm(embeds)
        enc = self.fc(enc)
        return enc