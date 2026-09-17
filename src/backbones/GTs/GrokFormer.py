# Adaptation of the file: https://github.com/GGA23/GrokFormer/blob/main/model.py
# Credits to the original authors
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class SineEncoding(nn.Module):
    def __init__(self, k, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = k
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for _ in range(k)])

    def forward(self, e):
        # e: (B, k)
        out_e = []
        ee = e.unsqueeze(-1)  # (B, k, 1)
        for i in range(self.k):
            eeig = torch.ones_like(ee)  # (B, k, 1)
            ei = ee.pow(i + 1)
            div = torch.arange(1, self.hidden_dim // 2 + 1, dtype=torch.float, device=e.device)
            pe = ei * div  # (B, k, hidden_dim//2)
            eeig = torch.cat((eeig, torch.sin(pe), torch.cos(pe)), dim=-1)  # (B, k, hidden_dim+1)
            out_e.append(self.eig_ws[i](eeig))  # (B, k, 1)
        return out_e


class GrokFormerAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.linear_q = nn.Linear(hidden_size, num_heads * hidden_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * hidden_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * hidden_size)
        self.att_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(num_heads * hidden_size, hidden_size)

    def forward(self, q, k, v):
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        # Linear attention: Q @ (K^T @ V)
        x = k.transpose(-2, -1) @ v
        x = self.att_dropout(x)
        x = q @ x
        return self.output_layer(x)


class GrokFormerFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(self.gelu(self.layer1(x)))


class GrokFormer_encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            in_channels_e: int,
            hidden_channels: int,
            out_channels: int,
            num_layers: int = 1,
            k: int = 8,
            nheads: int = 1,
            sine_dim: int = 128,
            tran_dropout: float = 0.0,
            feat_dropout: float = 0.0,
            prop_dropout: float = 0.0,
        ):
        super().__init__()

        self.num_layers = num_layers
        self.k = k

        # Feature encoder
        self.feat_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Eigenvalue encoder
        self.eig_encoder = SineEncoding(k, sine_dim)

        # Learnable spectral filter weights
        self.alpha = nn.Linear(k, 1, bias=False)

        # Transformer layers
        self.mha_norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.mhas = nn.ModuleList([
            GrokFormerAttention(hidden_channels, nheads, tran_dropout) for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            GrokFormerFFN(hidden_channels, hidden_channels, hidden_channels) for _ in range(num_layers)
        ])

        # Dropout layers
        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.prop_dropout = nn.Dropout(prop_dropout)

        # Output projection
        self.out_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        # Data unpacking
        x = data.x
        eigenvalues = data.eigenvalues      # (total_nodes, k)
        eigenvectors = data.eigenvectors    # (total_nodes, k)
        batch = data.batch

        # Encode features
        h = self.feat_dp1(x)
        h = self.feat_encoder(h)
        h = self.feat_dp2(h)

        # Convert to dense batched format for matrix operations
        h_dense, mask = to_dense_batch(h, batch) # (B, max_nodes, hidden)
        u_dense, _ = to_dense_batch(eigenvectors, batch) # (B, max_nodes, k)
        e_dense, _ = to_dense_batch(eigenvalues, batch) # (B, max_nodes, k)
        ut_dense = u_dense.transpose(1, 2) # (B, k, max_nodes)

        # Get one eigenvalue vector per graph (first node of each graph)
        e_per_graph = e_dense[:, 0, :] # (B, k)

        # Encode eigenvalues into spectral filter
        eig = self.eig_encoder(e_per_graph) # list of k tensors, each (B, k, 1)
        new_e = torch.cat(eig, dim=-1) # (B, k, k)
        new_e = self.alpha(new_e) # (B, k, 1)

        # Transformer + spectral convolution layers
        for i in range(self.num_layers):
            # Spectral convolution: u @ (filter * (u^T @ h))
            utx = ut_dense @ h_dense # (B, k, hidden)
            h_encoder = u_dense @ (new_e * utx) # (B, max_nodes, hidden)
            h_encoder = self.prop_dropout(h_encoder)

            # Transformer block
            mha_h = self.mha_norms[i](h_dense)
            mha_h = self.mhas[i](mha_h, mha_h, mha_h)
            h_dense = h_dense + self.mha_dropout(mha_h) + h_encoder

            ffn_h = self.ffn_norms[i](h_dense)
            ffn_h = self.ffns[i](ffn_h)
            h_dense = h_dense + self.ffn_dropout(ffn_h)

        # Convert back to sparse format
        h = h_dense[mask]

        # Project to output
        out = self.out_lin(h)
        return out