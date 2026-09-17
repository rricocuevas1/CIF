import torch
import math
import torch.nn.functional as F
from src.hyper_parameters import POOLING
import torch.nn as nn
from typing import Dict, Any, Type
from src.baselines.cgnn.cgnn import CGNN
from cif.remix_readout_NoMC import ReMix_Readout_NoMC


class CIF_J_NoMC(CGNN):
    # No MC -> ReMix_Readout_NoMC
    # 
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
        # Overwritte Remix Readout to not use MC
        self.remix_readout = ReMix_Readout_NoMC(
            # Projection
            projection= self.projection_fc1,
            # Core MLP classifier (Classification head)
            mlp = self.mlp
        )        
    
    # Loss
    def compute_loss(self, batch, remix=True):
        # CIF loss
        # Edge feature existance flag: True if edge features exist / false otherwise
        exist_edge_features = (self.in_channels_e != None)
        if not exist_edge_features: # No edge features
            batch.edge_attr = None
        
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
        
        # 3. PREDICTION 
        # Forward pass over the MLP, Dim = (batch_size, batch_size, n_classes)
        #                                  (i         , j         ,y         )
        logits = self.remix_readout(
            h_c=h_c, 
            h_s=h_s, 
            remix=remix
        ) # returns logits
        
        # Application of Jensen's inequality
        log_probs = F.log_softmax(logits, dim=-1) # (batch_size, batch_size, num_classes)
        avg_log_probs = log_probs.mean(dim=1)    # (batch_size, num_classes)
        
        # Labels
        labels = batch.y # Dim = (batch_size, )

        # Overall CIF loss
        total_loss = F.nll_loss(avg_log_probs, labels)
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