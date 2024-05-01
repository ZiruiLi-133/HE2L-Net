import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CustomTransformer(nn.Module):
    def __init__(self, input_size, output_size, model_dim, num_heads, num_encoder_layers, num_decoder_layers,
                 max_seq_length):
        super(CustomTransformer, self).__init__()
        self.model_dim = model_dim

        # Embeddings for input and output tokens
        self.src_embedding = nn.Embedding(input_size, model_dim)
        self.tgt_embedding = nn.Embedding(output_size, model_dim)

        # Positional encodings
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, model_dim))

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

        # Output linear layer
        self.output_linear = nn.Linear(model_dim, output_size)

    def forward(self, src, tgt):
        # Adding embeddings and positional encodings
        src = self.src_embedding(src) + self.positional_encoding[:src.size(0), :]
        tgt = self.tgt_embedding(tgt) + self.positional_encoding[:tgt.size(0), :]

        # Transformer
        output = self.transformer(src, tgt)

        # Output layer
        output = self.output_linear(output)
        return output