# This file contains the implementation of the CGNN model class
# There are several ways of training a CGNN
# Such training modes are implemented as sub-classes of CGNN
# The corresponding files can be found in the same directory
from typing import Dict, Any, Type
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from src.hyper_parameters import POOLING
from torch_geometric.nn import GCNConv
from baselines.general_models import GraphNN
from src.hyper_parameters import CGNN_hparams
from src.cif.remix_readout import ReMix_Readout


class CGNN(GraphNN):
    """The CGNN model class"""
    def __init__(
        self,
        gnn_backbone: Type[nn.Module],
        in_channels: int,
        in_channels_e: int,
        num_classes: int,
        optimizer_hparams: Dict[str, Any]
    ):
        super().__init__( 
            gnn_backbone,
            in_channels,
            in_channels_e,
            num_classes,
            optimizer_hparams
        )    
        # CGNN Model Class hyperparameters
        self.cgnn_config = {**CGNN_hparams}    
        
        # Dropout configs
        self.branch_dropout = nn.Dropout(self.cgnn_config["dropout"])
        self.causal_dropout = nn.Dropout1d(p=self.cgnn_config["dropout"]) 

        # Attention Maps 
        gnn_out_dim = self.backbone_config["gnn_out_channels"]
        # Edge Attention Map
        if self.in_channels_e != None: # i.e., if there are edge features
            # Edge embedding module
            self.edge_embedding_mlp = Sequential( # 2 layer MLP over the edge features
                    Linear(in_channels_e, self.cgnn_config['hidden_channels_mlp']), 
                    ReLU(), 
                    Linear(self.cgnn_config['hidden_channels_mlp'], gnn_out_dim)
                )
            self.edge_att_mlp = Linear(gnn_out_dim, 2) # 1 layer MLP over edges
        else: 
            self.edge_att_mlp = Linear(gnn_out_dim * 2, 2) # 1 layer MLP over edges
        # Node Attention Map
        self.node_att_conv = GCNConv(gnn_out_dim, 2) # 1 layer full Conv over nodes

        # Convolutions 
        # Causal Branch Convolution 
        self.bnc = BatchNorm1d(gnn_out_dim)
        self.causal_conv = GCNConv(gnn_out_dim, gnn_out_dim)

        # Spurious Branch Convolution
        self.bns = BatchNorm1d(gnn_out_dim)
        self.spurious_conv = GCNConv(gnn_out_dim, gnn_out_dim)

        # Readouts
        # Causal Readout 
        self.fc1_bn_c = BatchNorm1d(gnn_out_dim)
        self.fc1_c = Linear(gnn_out_dim, gnn_out_dim)
        self.fc2_bn_c = BatchNorm1d(gnn_out_dim)
        self.fc2_c = Linear(gnn_out_dim, num_classes) # Logits

        # Spurious Readout
        self.fc1_bn_s = BatchNorm1d(gnn_out_dim)
        self.fc1_s = Linear(gnn_out_dim, gnn_out_dim)
        self.fc2_bn_s = BatchNorm1d(gnn_out_dim)
        self.fc2_s = Linear(gnn_out_dim, num_classes) # Logits

        # Combined Readout
        # Projection
        if self.cgnn_config['cat_or_add'] == "cat": # Concatenation
            # Here it is assumed that out_channels_causal == out_channels_spurious
            # Projection from gnn_out_dim * 2 -> gnn_out_dim
            self.projection_fc1 = Linear(gnn_out_dim * 2, gnn_out_dim, bias=False)
        else: # If "add" then no projection needed
            self.projection_fc1 = nn.Identity() 
        
        # Remix Readout
        self.remix_readout = ReMix_Readout(
            # Projection
            projection= self.projection_fc1,
            # Core MLP classifier (Classification head)
            mlp = self.mlp
        )
        
        # Batch Normalization Initialization 
        for m in self.modules():
            if isinstance(m, BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)
        
    
    # Readout heads
    def causal_readout(self, h_c):
        """Causal readout: SHOULD predict the label"""
        h_c = self.causal_dropout(h_c) 
        h_c = F.relu(self.fc1_c(self.fc1_bn_c(h_c)))
        logits = self.fc2_c(self.fc2_bn_c(h_c))
        return logits # Logits
     
    def spurious_readout(self, h_s):
        """Spurious readout: should NOT predict the label"""
        h_s = F.relu(self.fc1_s(self.fc1_bn_s(h_s)))
        logits = self.fc2_s(self.fc2_bn_s(h_s))
        return logits # Logits

    def combined_readout(self, h_c, h_s, eval_random):
        """Combined readout with backdoor adjustment"""
        num = h_s.shape[0]
        l = list(range(num))
        if self.cgnn_config['with_random']:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l, device=h_s.device)
        
        if self.cgnn_config['cat_or_add'] == "cat":
            # [causal || spurious]
            h_z = torch.cat((h_c, h_s[random_idx]), dim=-1) # [h_c || h_s]
        else:
            h_z = h_c + h_s[random_idx] # h_c + h_s

        # Core MLP classifier (Classification head)
        logits = self.mlp(self.projection_fc1(h_z))
        return logits # Logits
    
    # Forward pass
    def forward(self, data, eval_random=None, c_i_f=False):
        # Edge feature existance flag: True if edge features exist / false otherwise
        exist_edge_features = self.in_channels_e is not None
        if not exist_edge_features: # No edge features
            data.edge_attr = None
        
        # Eval mode settup
        if eval_random is None:
            # eval_random False by defaut 
            eval_random = self.cgnn_config['eval_random']

        # Data unpacking
        edge_index = data.edge_index
        row, col = edge_index

        # 1. REPRESENTATION: Forward pass of the GNN encoder over the data
        h = self.gnn(data)

        # 2. DISENTANGLEMENT:
        # 2.1. Attention map over the edges in the embedding space
        # (the :0/:1 softmax decomposition is equivalent to att and 1-att with sigmoid)
        if not exist_edge_features: # If no edge features, we concatenate the node reps
            edge_rep = torch.cat([h[row], h[col]], dim=-1)
        else: # If edge features, we use the embedd the edges via an MLP and use that as reps
            edge_rep = self.edge_embedding_mlp(data.edge_attr)
        # Edge attention over the latent representations
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 1] # Causal
        edge_weight_s = edge_att[:, 0] # Spurious 

        # 2.2. Attention map over the nodes in the embedding space
        node_att = F.softmax(self.node_att_conv(h, edge_index), dim=-1)
        node_weight_c = node_att[:, 1] # Causal
        node_weight_s = node_att[:, 0] # Spurious

        # 2.3. Node-level-representation disentanglenent 
        # (Actual disentanglement step in the latent space over nodes)
        h_c = node_weight_c.view(-1, 1) * h # Causal
        h_s = node_weight_s.view(-1, 1) * h # Spurious

        # Branch convolutions (Spurious and Causal representation updates)
        h_c = self.branch_dropout( # Causal Branch convolution
            F.elu(self.causal_conv(
                self.bnc(h_c), # Causal rep
                edge_index, 
                edge_weight=edge_weight_c) # Causal edge weight
            )
        )
        h_s = self.branch_dropout( # Spurious Branch convolution
            F.elu(self.spurious_conv(
                self.bns(h_s), # Spurious rep
                edge_index, 
                edge_weight=edge_weight_s) # Spurious edge weight
            )
        )
        
        # 2.4. Graph-level pooling (mean pool)
        h_c = POOLING(h_c, data.batch)
        h_s = POOLING(h_s, data.batch)

        # 3. PREDICTION
        if c_i_f:
            c_logits = None # Not computed
            s_logits = None # Not computed 
            combined_logits = self.remix_readout(h_c=h_c, h_s=h_s)
        else: # CAL, ICL, ACE (CGNN SOTA)
            c_logits = self.causal_readout(h_c=h_c)
            s_logits = self.spurious_readout(h_s=h_s)
            combined_logits = self.combined_readout(h_c=h_c, h_s=h_s, eval_random=eval_random)
        
        # RETURN:
        return h_c, h_s, c_logits, s_logits, combined_logits

    # Validation
    def on_validation_epoch_end(self):
        # Classification Metrics
        super().on_validation_epoch_end() 

    
    # Test
    def on_test_epoch_end(self):
        # Classification Metrics
        super().on_test_epoch_end() 
    
    # Test step is shared across all CGNNs, 
    # Only the training and validation steps differ because of the different losses
    def test_step(self, batch, batch_idx):
        # Logits
        h_c, h_s, _, _, combined_logits = self(
            batch, 
            eval_random=self.cgnn_config['eval_random'],
            c_i_f=self.cgnn_config['c_i_f']
        )
        
        # Labels
        labels = batch.y.long()

        # Predictions
        probs = self.predict_probabilities(logits=combined_logits)

        # Classification metrics
        self.test_metrics.update(probs, labels)
        

    @torch.no_grad()
    def extract_causal_attention(self, data):
        """Returns per-node causal attention for a single graph."""
        h = self.gnn(data)
        node_att = F.softmax(self.node_att_conv(h, data.edge_index), dim=-1)
        node_weight_c = node_att[:, 1]   # Causal node scores
        return node_weight_c.cpu()   