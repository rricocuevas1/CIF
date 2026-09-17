import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv


class GIN_encoder(nn.Module):
    """
    Graph Isomorphisim Network
        - 2 GIN layers each with a 2 layer MLP with ReLU activation
        - ReLU activation
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gin1 = GINConv(
            Sequential(
                Linear(in_channels, hidden_channels), 
                ReLU(), 
                Linear(hidden_channels, hidden_channels)
            ), 
            train_eps=True
        )
        self.activation = ReLU()
        self.gin2 = GINConv(
            Sequential(
                Linear(hidden_channels, out_channels), 
                ReLU(), 
                Linear(out_channels, out_channels)
            ), 
            train_eps=True
        )
    
    def forward(self, data):
        # Data unpacking
        x = data.x
        edge_index = data.edge_index

        # Forward pass computation
        h = self.gin1(x, edge_index) # GIN
        h = self.activation(h) # ReLU
        out = self.gin2(h, edge_index) # GIN
        return out