"""Graph Convolutional Network model for pose classification."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool  # added pooling


class PoseGCN(nn.Module):
    """Graph Convolutional Network for yoga pose classification."""
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.5):
        """
        Initialize the GCN model.
        """
        super(PoseGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GCN layers (kept same structure as before)
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # last conv now stays in hidden space (not directly num_classes)
        self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        # small classifier after pooling
        # added this because we want graph-level prediction not node-level
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the model.
        """
        # apply GCN layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            
            # don't apply activation on last layer before pooling
            if i != len(self.conv_layers) - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)

        # if batch is provided, we pool to get one vector per graph
        # this is needed for classification of whole poses
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # fallback just in case (shouldn't really happen in training)
            x = x.mean(dim=0, keepdim=True)

        # final classification layer
        x = self.classifier(x)

        return x