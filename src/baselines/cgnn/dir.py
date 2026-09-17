from src.baselines.cgnn.cgnn import CGNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Type
from collections import deque
#from torchmetrics.classification import Accuracy
from src.hyper_parameters import POOLING


class DIR(CGNN):
    """DIR"""

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
        self.cgnn_config.update(model_hparams)
        memory_bank_size = self.cgnn_config.get('memory_bank_size', 256)
        self.memory_bank = deque(maxlen=memory_bank_size)
        
        # Tracking metrics for variance computation across batch interventions
        self.register_buffer('batch_risks', torch.zeros(1))
        self.risk_history = []
        

    def add_to_memory_bank(self, h_s_detached):
        """Add non-causal (spurious) representations to memory bank for interventions.
        
        Args:
            h_s_detached: Detached spurious graph representations [batch_size, hidden_dim]
        """
        # Convert to list and add each sample to the memory bank
        for rep in h_s_detached:
            self.memory_bank.append(rep.detach().clone())

    def create_intervention(self, h_s, batch_size):
        if len(self.memory_bank) < batch_size:
            # If memory bank doesn't have enough samples, return original
            return h_s.clone()
        
        # Sample indices from memory bank
        memory_indices = torch.randperm(len(self.memory_bank))[:batch_size]
        memory_bank_list = list(self.memory_bank)
        
        # Stack sampled representations
        h_s_intervened = torch.stack([
            memory_bank_list[idx].to(h_s.device) 
            for idx in memory_indices
        ])
        
        return h_s_intervened

    def compute_intervention_risks(self, h_c, h_s, labels):
        risks = []
        batch_size = h_c.shape[0]
        
        # Risk for original distribution
        combined_logits_orig = self.combined_readout(
            h_c=h_c,
            h_s=h_s,
            eval_random=False
        )
        combined_log_probs_orig = F.log_softmax(combined_logits_orig, dim=-1)
        combined_risk_orig = F.nll_loss(combined_log_probs_orig, labels, reduction='mean')
        risks.append(combined_risk_orig)
        
        # Create and evaluate interventional distributions
        num_interventions = max(1, min(3, len(self.memory_bank) // batch_size))
        
        for _ in range(num_interventions):
            h_s_intervened = self.create_intervention(h_s, batch_size)
            
            # Compute combined risk under the intervened spurious distribution
            combined_logits_int = self.combined_readout(
                h_c=h_c,
                h_s=h_s_intervened,
                eval_random=False
            )
            combined_log_probs_int = F.log_softmax(combined_logits_int, dim=-1)
            combined_risk_int = F.nll_loss(combined_log_probs_int, labels, reduction='mean')
            
            risks.append(combined_risk_int)
        
        return risks

    def compute_dir_loss(self, c_logits, s_logits, combined_logits, labels, h_c, h_s):
        lambda_var = self.cgnn_config.get('lambda', 1.0)
        causal_weight = self.cgnn_config.get('causal_weight', 1.0)
        spurious_weight = self.cgnn_config.get('spurious_weight', 1.0)
        
        # Compute log probabilities
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        combined_log_probs = F.log_softmax(combined_logits, dim=-1)
        
        # Primary losses
        causal_loss = F.nll_loss(c_log_probs, labels)
        
        # Spurious should be uninformative (uniform distribution)
        uniform_target = torch.ones_like(s_log_probs, dtype=torch.float, device=s_log_probs.device) / self.num_classes
        spurious_loss = F.kl_div(s_log_probs, uniform_target, reduction='batchmean')
        
        combined_loss = F.nll_loss(combined_log_probs, labels)
        
        # Invariance regularization via intervention risk variance
        # Compute risks under different interventional distributions
        risks = self.compute_intervention_risks(h_c, h_s, labels)
        
        # Compute mean and variance of risks
        risks_tensor = torch.stack(risks)
        mean_risk = torch.mean(risks_tensor)
        var_risk = torch.var(risks_tensor)
        
        # DIR objective: minimize average risk and variance of risks
        # This encourages the causal features to have stable predictions across interventions
        invariance_loss = mean_risk + lambda_var * var_risk
        
        # Total loss combines all components
        total_loss = (
            causal_weight * causal_loss +
            spurious_weight * spurious_loss +
            combined_loss +
            invariance_loss
        )
        
        return total_loss, {
            'causal_loss': causal_loss.detach(),
            'spurious_loss': spurious_loss.detach(),
            'combined_loss': combined_loss.detach(),
            'invariance_loss': invariance_loss.detach()
        }

    def training_step(self, batch, batch_idx):
        # Forward pass through disentanglement module
        data = batch
        edge_index = data.edge_index
        row, col = edge_index
        
        # Get representations from backbone
        h = self.gnn(data)
        
        # Disentanglement (edge and node attention)
        exist_edge_features = self.in_channels_e is not None
        if not exist_edge_features:
            edge_rep = torch.cat([h[row], h[col]], dim=-1)
        else:
            edge_rep = self.edge_embedding_mlp(data.edge_attr)
        
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 1]
        edge_weight_s = edge_att[:, 0]
        
        node_att = F.softmax(self.node_att_conv(h, edge_index), dim=-1)
        node_weight_c = node_att[:, 1]
        node_weight_s = node_att[:, 0]
        
        # Split representations into causal and spurious
        h_c = node_weight_c.view(-1, 1) * h
        h_s = node_weight_s.view(-1, 1) * h
        
        # Branch convolutions
        h_c = self.branch_dropout(
            F.elu(self.causal_conv(
                self.bnc(h_c),
                edge_index,
                edge_weight=edge_weight_c
            ))
        )
        h_s = self.branch_dropout(
            F.elu(self.spurious_conv(
                self.bns(h_s),
                edge_index,
                edge_weight=edge_weight_s
            ))
        )
        
        # Pool to graph level
        h_c = POOLING(h_c, data.batch)
        h_s = POOLING(h_s, data.batch)
        
        # Add current spurious representations to memory bank
        self.add_to_memory_bank(h_s.detach())
        
        # Generate predictions
        c_logits = self.causal_readout(h_c=h_c)
        s_logits = self.spurious_readout(h_s=h_s)
        combined_logits = self.combined_readout(h_c=h_c, h_s=h_s, eval_random=self.cgnn_config['with_random'])
        
        # Get labels
        labels = batch.y.long()
        
        # Compute DIR loss with invariance regularization
        total_loss, loss_dict = self.compute_dir_loss(
            c_logits=c_logits,
            s_logits=s_logits,
            combined_logits=combined_logits,
            labels=labels,
            h_c=h_c,
            h_s=h_s
        )
        
        # Logging
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        data = batch
        edge_index = data.edge_index
        row, col = edge_index
        
        # Get representations
        h = self.gnn(data)
        
        # Disentanglement
        exist_edge_features = self.in_channels_e is not None
        if not exist_edge_features:
            edge_rep = torch.cat([h[row], h[col]], dim=-1)
        else:
            edge_rep = self.edge_embedding_mlp(data.edge_attr)
        
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 1]
        edge_weight_s = edge_att[:, 0]
        
        node_att = F.softmax(self.node_att_conv(h, edge_index), dim=-1)
        node_weight_c = node_att[:, 1]
        node_weight_s = node_att[:, 0]
        
        h_c = node_weight_c.view(-1, 1) * h
        h_s = node_weight_s.view(-1, 1) * h
        
        # Branch convolutions
        h_c = self.branch_dropout(
            F.elu(self.causal_conv(
                self.bnc(h_c),
                edge_index,
                edge_weight=edge_weight_c
            ))
        )
        h_s = self.branch_dropout(
            F.elu(self.spurious_conv(
                self.bns(h_s),
                edge_index,
                edge_weight=edge_weight_s
            ))
        )
        
        # Pool to graph level
        h_c = POOLING(h_c, data.batch)
        h_s = POOLING(h_s, data.batch)
        
        # Generate predictions
        c_logits = self.causal_readout(h_c=h_c)
        s_logits = self.spurious_readout(h_s=h_s)
        combined_logits = self.combined_readout(h_c=h_c, h_s=h_s, eval_random=self.cgnn_config['eval_random'])
        
        # Get labels
        labels = batch.y.long()
        
        # Compute loss
        total_loss, loss_dict = self.compute_dir_loss(
            c_logits=c_logits,
            s_logits=s_logits,
            combined_logits=combined_logits,
            labels=labels,
            h_c=h_c,
            h_s=h_s
        )
        
        # Logging
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        
        # Classification metrics
        probs = self.predict_probabilities(logits=combined_logits)
        self.val_metrics.update(probs, labels)
        
        return total_loss
    