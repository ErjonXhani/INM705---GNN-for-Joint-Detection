"""GAT model for pose classification."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class PoseGAT(nn.Module):
    """Graph Attention Network for yoga pose classification."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        num_layers=3,
        heads=4,
        dropout=0.5
    ):
        super(PoseGAT, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        # storing graph attention layers here so we can loop through them
        self.conv_layers = nn.ModuleList()

        # first layer goes from input features -> hidden features
        # concat=True means output size becomes hidden_dim * heads
        self.conv_layers.append(
            GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        )

        # middle layers take hidden_dim * heads as input because of concatenation
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )

        # final graph layer
        # here concat=False keeps the output size as just hidden_dim
        # this makes the classifier part easier afterwards
        self.conv_layers.append(
            GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        )

        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        # after pooling, classify the whole pose graph
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch=None):
        """
        x: node features
        edge_index: graph connections
        batch: tells us which nodes belong to which graph
        """
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)

            # keep activation/dropout off the last graph layer
            # easier to treat that as the final node embedding stage
            if i != len(self.conv_layers) - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)

        # turn node embeddings into one graph embedding per sample
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # fallback in case we pass one graph without a batch vector
            x = x.mean(dim=0, keepdim=True)

        # final graph-level prediction
        x = self.classifier(x)
        return x