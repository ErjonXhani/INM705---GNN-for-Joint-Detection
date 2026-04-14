"""GraphSAGE model for pose classification."""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class PoseGraphSAGE(nn.Module):
    """GraphSAGE model for yoga pose classification."""

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.5):
        super(PoseGraphSAGE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(SAGEConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))

        self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))

        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)

            if i != len(self.conv_layers) - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)

        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        x = self.classifier(x)
        return x