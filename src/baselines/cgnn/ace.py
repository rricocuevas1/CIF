from src.baselines.cgnn.cgnn import CGNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Type


class CCE(nn.Module):
    # From https://github.com/12chen20/ACE/blob/main/cce.py
    def __init__(self, balancing_factor=0.5):
        super(CCE, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.balancing_factor = balancing_factor

    def forward(self, yHat, y,weight=None):
        # Device
        device = yHat.device
        # Note: yHat.shape[1] <=> number of classes
        batch_size = len(y)
        # cross entropy
        cross_entropy = self.nll_loss(F.log_softmax(yHat, dim=1), y)
        # complement entropy
        yHat = F.softmax(yHat, dim=1)
        Yg = yHat.gather(dim=1, index=torch.unsqueeze(y, 1))
        # 互补概率分布Px时添加一个较小的偏置项 
        Px = yHat / (1 - Yg+ 1e-4) 
        
        # Px = yHat / (1 - Yg) + 1e-7  
        Px_log = torch.log(Px + 1e-10 )
      

        y_zerohot = torch.ones(batch_size, yHat.shape[1]).scatter_(
            1, y.view(batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.to(device=device)
        complement_entropy = torch.sum(output) / (float(batch_size) * float(yHat.shape[1]))

        return cross_entropy - self.balancing_factor * complement_entropy


class ACE(CGNN):
    """ACE: Adaptive Causality-Enhanced framework for graph classification.
    github.com/12chen20/ACE
    """

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
        self.cce_loss = CCE(balancing_factor=self.cgnn_config.get("balancing_factor", 0.5))
    
    # ACE Loss
    def compute_loss(self, c_logits, s_logits, combined_logits, labels):
        # Weights
        c_w  = self.cgnn_config.get("c_weight",  0.7)
        o_w  = self.cgnn_config.get("o_weight",  1.0)
        co_w = self.cgnn_config.get("co_weight", 0.5)

        # Log_probs
        s_log_probs        = F.log_softmax(s_logits,        dim=-1)

        # Loss computation
        uniform_target = torch.ones_like(s_log_probs, dtype=torch.float, device=s_log_probs.device) / self.num_classes
        o_loss  = self.cce_loss(c_logits,        labels)
        s_loss  = F.kl_div(s_log_probs, uniform_target, reduction='batchmean')         
        co_loss = self.cce_loss(combined_logits, labels)                              

        return o_w * o_loss + c_w * s_loss + co_w * co_loss

    # Training step
    def training_step(self, batch, batch_idx):
        # Logits
        _, _, c_logits, s_logits, combined_logits = self(
            data=batch,
            eval_random=self.cgnn_config['with_random'],
            c_i_f=False
        )

        # Labels
        labels = batch.y.long()

        # Loss
        loss = self.compute_loss(c_logits, s_logits, combined_logits, labels)
        
        # Logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    # Validation step
    def validation_step(self, batch, batch_idx):
        # Logits
        h_c, h_s, c_logits, s_logits, combined_logits = self(
            data=batch, 
            eval_random=self.cgnn_config['eval_random'],
            c_i_f=False
        )
        
        # Labels
        labels = batch.y.long()

        # Loss
        total_loss = self.compute_loss(
            c_logits=c_logits, 
            s_logits=s_logits, 
            combined_logits=combined_logits, 
            labels=labels
        )
        
        # Logging
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        # Predictions
        probs = self.predict_probabilities(logits=combined_logits)

        # Classification metrics
        self.val_metrics.update(probs, labels)

        return total_loss
