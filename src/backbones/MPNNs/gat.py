import torch.nn as nn
from torch.nn import ELU
from torch_geometric.nn import GATv2Conv

# Vanilla GAT_encoder
class GAT_encoder(nn.Module):
    """
    Graph Attention Network
        - 2 GAT layers with the improved v2 operator and edge feature aggregation
        - ELU activation
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, in_channels_e = None):
        super().__init__()
        self.in_channels_e = in_channels_e
        self.gat1 = GATv2Conv(in_channels, hidden_channels, edge_dim=in_channels_e)
        self.activation = ELU()
        self.gat2 = GATv2Conv(hidden_channels, out_channels, edge_dim=in_channels_e)
    
    def forward(self, data):
        # Data unpacking and dimension fixes are performed
        # into the GNNs forward pass
        x = data.x
        edge_index = data.edge_index
        # Small dimension fix 
        if (getattr(data, "edge_attr", None) is not None): # If there are edge features
            edge_attr = data.edge_attr # Do nothing special
            if edge_attr.dim() == 1: # If there is only one edge feature
                edge_attr = edge_attr.unsqueeze(-1) # Fix the dimension
            if self.in_channels_e == None: # In case that we decide to ignore the edge features
                edge_attr = None # Set them to None
        else:
            edge_attr = None # The same holds if they do not exist
        
        # Forward pass computation
        h = self.gat1(x, edge_index, edge_attr=edge_attr) # GATv2e
        h = self.activation(h) # ELU
        out = self.gat2(h, edge_index, edge_attr=edge_attr) # GATv2e
        return out

