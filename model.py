# model.py
import torch.nn as nn
import torch

class SenRNN(nn.Module):
    def __init__(self,vocab_len,dim,hidden_dim,output_dim):
        super().__init__()
        n_filter = 100
        self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=dim)
        self.rnn = nn.RNN(dim , hidden_size = hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        


    def forward(self, x):
        emb = self.embedding(x)
        output, hidden = self.rnn(emb)

        return self.fc(hidden.squeeze(0))