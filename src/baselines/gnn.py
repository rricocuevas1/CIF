# General GNN architecture [MPNNs/GTs] + training + inference 
# pipeline for graph-level classification [binary AND multi-class]
import torch.nn as nn
import torch.nn.functional as F
from src.hyper_parameters import POOLING
from typing import Dict, Any
from typing import Type
from baselines.general_models import GraphNN


# Graph Neural Network model class. It can take MPNN and GT graph encoders
class GNN(GraphNN):
    # Initialization of the GNN model
    def __init__(
        self,
        gnn_backbone: Type[nn.Module],
        in_channels: int,
        in_channels_e: int,
        num_classes: int,
        model_hparams,
        optimizer_hparams: Dict[str, Any]
    ):
        super().__init__(
            gnn_backbone,
            in_channels,
            in_channels_e,
            num_classes,
            optimizer_hparams
        )
            
    # Forward pass
    def forward(self, data):
        h = self.gnn(data) # Node-level embeddings      
        h = POOLING(h, data.batch) # Graph-level embeddings
        logits = self.mlp(h) # Logits
        return logits
    
    # Train (BCE loss)
    def compute_loss(self, logits, labels):
        # Log_probs
        log_probs = F.log_softmax(logits, dim=-1)

        # Loss computation
        total_loss = F.nll_loss(log_probs, labels)
        return total_loss
        
    def training_step(self, batch, batch_idx):
        # Logits
        logits = self(batch)

        # Labels
        labels = batch.y.long()

        # Loss
        total_loss = self.compute_loss(logits, labels)

        # Logging
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return total_loss
    
    # Validation
    def validation_step(self, batch, batch_idx):
        # Logits
        logits = self(batch)
        
        # Labels
        labels = batch.y.long()

        # Validation loss computation
        val_loss = self.compute_loss(logits, labels)
        
        # Logging
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        
        # Predictions
        probs = self.predict_probabilities(logits=logits)

        # Classification metrics
        self.val_metrics.update(probs, labels)
        return val_loss
    
    # Test
    def test_step(self, batch, batch_idx):
        # Logits
        logits = self(batch)
        
        # Labels
        labels = batch.y.long()

        # Predictions
        probs = self.predict_probabilities(logits=logits)

        # Classification metrics
        self.test_metrics.update(probs, labels)

    
