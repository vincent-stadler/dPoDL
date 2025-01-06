import torch
import torch.nn as nn
import numpy as np


# Define the FloatSequenceTransformer class
class FloatSequenceTransformer(nn.Module):

    TRAINING_MEAN = 0.8333610828529446  # to (de)standardize input sequences
    TRAINING_STD = 0.5440675840110392  # to (de)standardize input sequences
    MAX_STD_TRAIN = 0.17382664912384846   # to calculate confidence score
    ALPHA = 0.5   # to calculate confidence score

    def __init__(self, embedding_dim=8, num_heads=1, num_layers=1, dropout=0.2, input_length=60):
        super(FloatSequenceTransformer, self).__init__()

        self.embedding = nn.Linear(1, embedding_dim)
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_layers = num_layers

        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                       batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(embedding_dim, 1)
        self.input_length = input_length

    def generate_positional_encoding(self, seq_length, device):
        position = torch.arange(seq_length, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=device).float() * -(np.log(10000.0) / self.embedding_dim))
        pe = torch.zeros(seq_length, self.embedding_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, key_padding_mask):
        seq_length = x.size(1)
        x = self.embedding(x)
        positional_encoding = self.generate_positional_encoding(seq_length, x.device)
        x = x + 0.1 * positional_encoding

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.output_layer(x)
        return x[:, -1, :]
