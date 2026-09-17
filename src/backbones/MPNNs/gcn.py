import torch.nn as nn
from torch.nn import ReLU
from torch_geometric.nn import GCNConv

# Vanilla GCN_encoder
class GCN_encoder(nn.Module):
    """
    Graph Convolution Network
        - 2 GCN layers
        - ReLU activation
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.activation = ReLU()
        self.gcn2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, data):
        # Data unpacking
        x = data.x
        edge_index = data.edge_index
        
        # Forward pass computation
        h = self.gcn1(x, edge_index) # GCN
        h = self.activation(h) # ReLU
        out = self.gcn2(h, edge_index) # GCN
        return out

