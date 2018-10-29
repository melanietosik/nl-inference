import torch
import torch.nn as nn
import torch.nn.functional as F

import fasttext_loader


class CNN(nn.Module):
    """
    CNN: 2-layer 1-D convolutional network with ReLU activations
    """

    def __init__(
        self, vocab_size, emb_dim, hidden_dim, kernel_size, dropout_prob,
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

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            emb_dim, hidden_dim, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Fully connected layers
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
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
        p_bsz, p_time = p.shape
        h_bsz, h_time = h.shape

        # Embed and encode premises
        x = self.embedding(p)
        x = x.transpose(1, 2)  # Conv1d expects (bsz x features x length)
        hx = F.relu(self.conv1(x))
        hx = F.relu(self.conv2(hx))
        hx = F.max_pool1d(hx, p_time).squeeze(2)  # Max pooling, drop dim=2
        hx = self.dropout(hx)  # Dropout regularization

        # Embed and encode hypotheses
        y = self.embedding(h)
        y = y.transpose(1, 2)  # Conv1d expects (bsz x features x length)
        hy = F.relu(self.conv1(y))
        hy = F.relu(self.conv2(hy))
        hy = F.max_pool1d(hy, h_time).squeeze(2)  # Max pooling, drop dim=2
        hy = self.dropout(hy)  # Dropout regularization

        # Concatenate sentence representations
        h = torch.cat((hx, hy), dim=1)

        # Feed concatenation through fully-connected layers
        h = F.relu(self.fc1(h))
        h = self.dropout(h)  # Dropout regularization
        h = self.fc2(h)

        return h
