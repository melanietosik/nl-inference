import torch
import torch.nn as nn
import torch.nn.functional as F

import fasttext_loader


class BiGRU(nn.Module):
    """
    BiGRU: bi-directional GRU
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fw_gru = nn.GRUCell(input_dim, hidden_dim)  # Forward
        self.bw_gru = nn.GRUCell(input_dim, hidden_dim)  # Backward

    def forward(self, x):
        time = x.size(1)  # x = [batch_size x time x input]
        hx_1 = x.new(x.size(0), self.hidden_dim).zero_()
        for i in range(time):
            hx_1 = self.fw_gru(x[:, i], hx_1)
        hx_2 = x.new(x.size(0), self.hidden_dim).zero_()
        for i in reversed(range(time)):
            hx_2 = self.bw_gru(x[:, i], hx_2)

        # Concatenate forward and backward representations
        h = torch.cat([hx_1, hx_2], dim=1)
        return h


class RNN(nn.Module):
    """
    RNN: single-layer, bi-directional GRU
    """

    def __init__(
        self, vocab_size, emb_dim, hidden_dim, dropout_prob,
            padding_idx, num_classes, id2tok):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx,
        )

        # Load and copy pre-trained embedding weights
        weights = fasttext_loader.create_weights(id2tok)
        self.embedding.weight.data.copy_(torch.from_numpy(weights))
        self.embedding.weight.requires_grad = False  # Freeze embeddings

        # Bi-directional GRU layer
        self.bigru = BiGRU(input_dim=emb_dim, hidden_dim=hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Fully connected layers
        self.fc1 = nn.Linear(2 * 2 * hidden_dim, hidden_dim)  # hx+hy and fw+bw
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, p, h):
        """
        Forward pass

        1. Embed premises
        2. Embed hypotheses
        3. Concatenate encoded sentences

        @param p: premises
        @param h: hypotheses
        """
        # Embed and encode premises
        x = self.embedding(p)
        hx = self.bigru(x)
        hx = self.dropout(hx)  # Dropout regularization

        # Embed and encode hypotheses
        y = self.embedding(h)
        hy = self.bigru(y)
        hy = self.dropout(hy)  # Dropout regularization

        # Concatenate sentence representations
        h = torch.cat((hx, hy), dim=1)

        # Feed concatenation through fully-connected layers
        h = F.relu(self.fc1(h))
        h = self.dropout(h)  # Dropout regularization
        h = self.fc2(h)

        return h
