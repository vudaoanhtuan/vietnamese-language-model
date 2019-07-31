import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size, embeding_size, hidden_size):
        super(Model, self).__init__()
        self.embeding = nn.Embedding(vocab_size, embeding_size)
        self.lstm = nn.LSTM(embeding_size, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden_state=None):
        # x: BxS
        x = self.embeding(x) # BxSxE
        if hidden_state is None:
            x, hidden_state = self.lstm(x) # BxSx2H
        else:
            x, hidden_state = self.lstm(x, hidden_state) # BxSx2H
        x = F.relu(x)
        x = self.linear(x) # BxSxV
        return x, hidden_state
    
    def predict(self, x, hidden_state=None):
        x, hidden_state = self.forward(x, hidden_state)
        x = F.softmax(x, dim=-1) # BxSxV
        return x, hidden_state


def get_model(vocab_size, embeding_size, hidden_size):
    return Model(vocab_size, embeding_size, hidden_size)