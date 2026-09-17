import torch
from typing import Optional, Dict, Any
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, ModuleList
from torch_geometric.nn import GPSConv, GINEConv
from torch_geometric.nn.attention import PerformerAttention

# GraphGPS Official repository: https://github.com/rampasek/GraphGPS
# GraphGPS is a 3-part recipe on how to build graph Transformers with linear complexity. 
# The graph GPS recipe consists of choosing 3 ingredients:
#       1. Positional/Structural encoding: LapPE, RWSE, SignNet, EquivStableLapPE
#       2. Local message-passing mechanism: GatedGCN, GINE, PNA
#       3. Global attention mechanism: Transformer, Performer, BigBird

# Following the official PyG documentation: 
# https://pytorch-geometric.readthedocs.io/en/2.7.0/tutorial/graph_transformer.html
# We selected: 
#       1. RWSE as Structural Encoding, 
#       2. GINE as local MPNN, 
#       3. Performer as (linear) Global attention mechanisim (Linear in the number of nodes, i.e. O(|V|))

# Necessary class for the performer attention mechanisim
class RedrawProjection:
    def __init__(
            self, 
            model: torch.nn.Module,
            redraw_interval: Optional[int] = None
        ):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

# GraphGPS (RWSE-GINE/GIN-Performer_Attn)
class GraphGPS_encoder(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            in_channels_e: int,  
            hidden_channels: int,
            out_channels: int, 
            num_layers: int,
            rwse_dim: int,
            pe_dim: int,
            attn_type: str,
            attn_heads: int, 
            attn_kwargs: Dict[str, Any]
        ):
        super().__init__()
        # Node embedding
        self.node_emb = Linear(in_channels, hidden_channels - pe_dim)
        # RWSE
        self.pe_lin = Linear(rwse_dim, pe_dim)
        self.pe_norm = BatchNorm1d(rwse_dim)
        # Edge embeddings
        self.in_channels_e = in_channels_e
        self.edge_emb = Linear(in_channels_e, hidden_channels) if in_channels_e is not None else None
        # Layers
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels),
            ) 
            local_conv = GINEConv(nn, train_eps=True, edge_dim=hidden_channels)
            conv = GPSConv(
                hidden_channels, 
                local_conv, 
                heads=attn_heads,
                attn_type=attn_type, 
                attn_kwargs=attn_kwargs
            )
            self.convs.append(conv)
        
        # Projection to the final embedding space
        self.mlp = Linear(hidden_channels, out_channels) 

        # Redraw random projections for Performer attention
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)
        
    def forward(self, data):
        # Data unpacking
        x = data.x
        rwse = data.RWSE
        edge_index = data.edge_index
        # Dimension fixes
        if (getattr(data, "edge_attr", None) is not None): # If there are edge features
            edge_attr = data.edge_attr # Do nothing special
            if edge_attr.dim() == 1: # If there is only one edge feature
                edge_attr = edge_attr.unsqueeze(-1) # Fix the dimension
            
            # Edge embeddings
            if self.in_channels_e is None:
                # For datasets where we decide to ignore edge features, otehrwise GINE gives error
                edge_attr = torch.zeros(edge_index.size(1), self.convs[0].channels, device=edge_index.device)
            else:
                # Edge embeddings
                edge_attr = self.edge_emb(edge_attr) # Dim = (num_edges, hidden_channels) 
        else:
            # For datasets whith no edge features, otehrwise GINE gives error
            edge_attr = torch.zeros(edge_index.size(1), self.convs[0].channels, device=edge_index.device) 
        batch = data.batch
        
        # Positional Encodings
        x_pe = self.pe_norm(rwse) # Projections to hidden_channels
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1) # Dim = (num_nodes, hidden_channels)
        
        # Forward pass over GPS layers
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        
        # Final projection
        out = self.mlp(x)
        return out 


