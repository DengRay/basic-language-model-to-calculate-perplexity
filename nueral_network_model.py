import torch
import torch.nn as nn

class NNLM(nn.Module):
    def __init__(self, n_class,embedding_dim,input_size, hidden_size):
        super(NNLM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.emb = nn.Embedding(n_class, embedding_dim)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.n_class)
        
    def forward(self, x):
        x = self.emb(x)
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = torch.tanh(x)
        output = self.fc2(x)
        return output
