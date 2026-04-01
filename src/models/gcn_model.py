"""Graph Convolutional Network model for pose classification."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class PoseGCN(nn.Module):
    """Graph Convolutional Network for yoga pose classification."""
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.5):
        """
        Initialize the GCN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            num_classes: Number of pose classes
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super(PoseGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GCN layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.conv_layers.append(GCNConv(hidden_dim, num_classes))
        
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the model.
        
        Args:
            x: Node feature matrix
            edge_index: Edge index
            batch: Batch indices for graph pooling
            
        Returns:
            Output logits
        """
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout_layer(x)
        
        x = self.conv_layers[-1](x, edge_index)
        return x
