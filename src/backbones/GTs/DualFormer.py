# Adaptation of the file: https://github.com/JiamingZhuo/DUALFormer/blob/main/model/dualformer.py
# Credits to the original authors

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, APPNP
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
)
from typing import Optional
from torch.nn import ReLU
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import spmm

# Adaptation for graph-level classification
def full_attention_conv(q, k, v, batch, output_attn=False):
    """Batch-aware version: attention within each graph only."""
    C, H = q.shape[1], q.shape[2]
    
    q_flat = q.reshape(-1, C * H)
    k_flat = k.reshape(-1, C * H)
    v_flat = v.reshape(-1, C * H)
    
    q_dense, mask = to_dense_batch(q_flat, batch)  # (B, N_max, C*H)
    k_dense, _ = to_dense_batch(k_flat, batch)
    v_dense, _ = to_dense_batch(v_flat, batch)
    
    B, N_max = q_dense.shape[0], q_dense.shape[1]
    q_dense = q_dense.reshape(B, N_max, C, H)
    k_dense = k_dense.reshape(B, N_max, C, H)
    v_dense = v_dense.reshape(B, N_max, C, H)
    
    sqrt_n = torch.sqrt(torch.tensor(N_max, dtype=torch.float32, device=q.device))
    
    a = torch.einsum("blmh,bldh->bmdh", k_dense / sqrt_n, v_dense)
    attention = torch.softmax(a, dim=1)
    output = torch.einsum("blmh,bmdh->bldh", q_dense, attention)
    
    output_flat = output.reshape(B, N_max, C * H)
    output_sparse = output_flat[mask].reshape(-1, C, H)
    
    if output_attn:
        return output_sparse, attention
    return output_sparse


class TransConvLayer(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, batch, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.out_channels ,
                                             self.num_heads)
        key = self.Wk(source_input).reshape(-1, self.out_channels ,
                                            self.num_heads)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.out_channels,
                                                  self.num_heads)
        else:
            value = source_input.reshape(-1, self.out_channels, 1)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, batch, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(query, key, value, batch, output_attn)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=-1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, activation=ReLU(), num_layers=1, num_heads=2,
                 alpha=0.1, dropout=0.3, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))
        self.dropout = dropout
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act
        self.activation = activation
        self.reset_parameters()
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, batch):
        layer_ = []
        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x, batch=batch)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)
        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class Graph_Conv(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        super().reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  
                        edge_index, edge_weight, x.size(self.node_dim))
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm( 
                        edge_index, edge_weight, x.size(self.node_dim))
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
    

class DualFormer_encoder(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 hidden_channels,
                 out_channels,
                 activation,
                 num_gnns,
                 num_trans,
                 num_heads,
                 dropout_trans,
                 dropout,
                 alpha,
                 use_bn,
                 lammda=0.1,
                 GraphConv='sgc'):
        super(DualFormer_encoder, self).__init__()
        self.activation = activation
        self.num_gnns = num_gnns
        self.layers_trans = TransConv(in_channels, hidden_channels, self.activation,
                                      num_layers=num_trans, num_heads=num_heads,
                                      alpha=alpha, dropout=dropout_trans,
                                      use_bn=use_bn, use_residual=True,
                                      use_weight=True, use_act=True)

        if GraphConv == 'sgc':
            self.convs = torch.nn.ModuleList()
            for _ in range(num_gnns):
                self.convs.append(Graph_Conv())
        elif GraphConv == 'gcn':
            self.convs = torch.nn.ModuleList()
            for _ in range(num_gnns):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
        elif GraphConv == 'appnp':
            self.convs = APPNP(num_gnns, lammda)

        self.GraphConv = GraphConv
        self.linear_project = Linear(hidden_channels, out_channels)
        self.dropout = dropout

        self.params1 = list(self.layers_trans.parameters())
        self.params2 = list(self.linear_project.parameters())

        self.traning = True
        self.reset_parameters()

    def reset_parameters(self):

        self.layers_trans.reset_parameters()
        self.linear_project.reset_parameters()

    def forward(self, data):
        # Data unpacking
        x = data.x
        edge_index = data.edge_index
        batch=data.batch

        z = self.layers_trans(x, batch=batch)
        if self.GraphConv in ['sgc', 'gcn']:#sgc, gcn
            for _, conv in enumerate(self.convs):
                z = conv(z, edge_index)
        else:
            z = self.convs(z, edge_index) #appnp
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.linear_project(z)

        return z # Return the final embedding