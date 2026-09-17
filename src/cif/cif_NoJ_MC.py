import torch
import math
import torch.nn.functional as F
from src.hyper_parameters import POOLING
import torch.nn as nn
from typing import Dict, Any, Type
from src.baselines.cgnn.cgnn import CGNN


class CIF_NoJ_MC(CGNN):
    """Causal-Information-Flow-based training"""
    # No Jensen -> 2LRs
    # MC
    def __init__(
        self,
        gnn_backbone: Type[nn.Module],
        in_channels: int,
        in_channels_e: int,
        num_classes: int,
        model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any] 
    ):
        super().__init__(
            gnn_backbone,
            in_channels,
            in_channels_e,
            num_classes,
            optimizer_hparams
        )
        # Update hyperparameters
        self.cgnn_config.update(model_hparams)        
    
    # Loss
    def compute_loss(self, batch, remix=True):
        # CIF loss
        # Edge feature existance flag: True if edge features exist / false otherwise
        exist_edge_features = (self.in_channels_e != None)
        if not exist_edge_features: # No edge features
            batch.edge_attr = None

        # Parameters and hyper-parameters
        h_params = self.cgnn_config
        num_classes = self.num_classes
        sigma = h_params['sigma']
        n_samples_c = h_params['n_samples_c']
        n_samples_s = h_params['n_samples_s']

        # Data
        edge_index = batch.edge_index # Dim = (2, n_edges_batch)
        row, col = edge_index
        
        # 1. REPRESENTATION: GNN encoder over the data
        h = self.gnn(batch)  # Dim = (n_nodes_batch, emb_dim)

        # 2. DISENTANGLEMENT:
        if not exist_edge_features: # No edge features
            edge_rep = torch.cat([h[row], h[col]], dim=-1)
        else: # Edge features
            edge_rep = self.edge_embedding_mlp(batch.edge_attr)
        # Edge attention
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 1]  # Dim = (num_edges,)
        edge_weight_s = edge_att[:, 0]  # Dim = (num_edges,)

        # Node attention 
        node_att = F.softmax(self.node_att_conv(h, edge_index), dim=-1)
        node_weight_c = node_att[:, 1]
        node_weight_s = node_att[:, 0]

        # Branch convolutions (Node-level disentangled representations)
        h_c = self.branch_dropout(
            F.elu(self.causal_conv(
                self.bnc(node_weight_c.view(-1, 1) * h), 
                edge_index, 
                edge_weight=edge_weight_c)
            )
        )
        h_s = self.branch_dropout(
            F.elu(self.spurious_conv(
                self.bns(node_weight_s.view(-1, 1) * h), 
                edge_index, 
                edge_weight=edge_weight_s)
            )
        )

        # Pooling (Graph-level disentangled representations)
        h_c = POOLING(h_c, batch.batch)  # (batch_size, emb_dim)
        h_s = POOLING(h_s, batch.batch)  # (batch_size, emb_dim)

        # 3. Sampling
        # Causal sampling: G_i ~ P(G), c_ik ~ P(c|G_i), Dim = (n_samples_c, n_nodes_batch, emb_dim)
        h_c = h_c.unsqueeze(0).expand(n_samples_c, -1, -1)
        h_c = h_c + sigma * torch.randn_like(h_c)
        # Spurious sampling: G_j ~ P(G), s_jl ~ P(s|G_j), Dim = (n_samples_s, n_nodes_batch, emb_dim)
        h_s = h_s.unsqueeze(0).expand(n_samples_s, -1, -1)
        h_s = h_s + sigma * torch.randn_like(h_s)
        
        # 4. PREDICTION 
        # Forward pass over the MLP, Dim = (n_samples_z, n_samples_c, batch_size, n_samples_s, batch_size, n_classes)
        #                                  (n         , k          , i         , l          , j          , y)
        logits = self.remix_readout(
            h_c=h_c, 
            h_s=h_s, 
            remix=remix
        ) # returns logits
        
        # No Jensen
        log_probs = F.log_softmax(logits, dim=-1) # (n_samples_z, n_samples_c, batch_size, n_samples_s, batch_size, num_classes)
        n = log_probs.size(0) * log_probs.size(3) * log_probs.size(4)
        log_avg_probs = torch.logsumexp(log_probs, dim=(0, 3, 4)) - math.log(n) # Dim = (n_samples_c, batch_size, n_classes)
        log_avg_probs = log_avg_probs.reshape(-1, num_classes) # Dim = (n_samples_c * n_nodes_batch, num_classes)
        
        # Labels
        labels = batch.y # Dim = (n_nodes_batch, )
        labels = labels.unsqueeze(0).expand(n_samples_c, -1).long() # Dim = (n_samples_c, n_nodes_batch)
        labels = labels.reshape(-1) # Dim = (n_samples_c * n_nodes_batch,)
        
        # Overall CIF loss
        total_loss = F.nll_loss(log_avg_probs, labels)
        return total_loss
      
    # Training step
    def training_step(self, batch, batch_idx):
        # Loss
        total_loss = self.compute_loss(batch, remix=True)
        
        # Logging
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return total_loss
    
    # Validation
    def validation_step(self, batch, batch_idx):
        # Logits 
        h_c, h_s, _, _, combined_logits = self(
            data=batch, 
            c_i_f=self.cgnn_config['c_i_f']
        )

        # Labels
        labels = batch.y.long()

        # Loss
        total_loss = self.compute_loss(batch)

        # Logging
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        
        # Predictions
        probs = self.predict_probabilities(logits=combined_logits)

        # Classification metrics
        self.val_metrics.update(probs, labels)

        return total_loss

    # Two-timescale optimizer (slow theta, fast phi)
    def configure_optimizers(self):
        head_param_ids = {id(p) for p in self.mlp.parameters()}  # phi

        head_params, base_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (head_params if id(p) in head_param_ids else base_params).append(p)

        assert len(head_params) > 0, "No head (phi) params matched"
        assert len(base_params) > 0, "No base (theta) params matched"

        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": self.optimizer_config["lr_base"]},  # theta — slow
                {"params": head_params, "lr": self.optimizer_config["lr_head"]},  # phi  — fast
            ],
            weight_decay=self.optimizer_config["wd"],
        )
        return optimizer