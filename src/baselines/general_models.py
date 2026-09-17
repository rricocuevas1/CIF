# General class for graph representation learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any
from typing import Type
from torch.nn import Sequential, Linear, BatchNorm1d, ELU
# Hyperparameter configuration of the GNNs
from src.hyper_parameters import GNN_encoders_hparams
from src.utils import get_gnn_kwargs
# Classification Metrics
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision


class MLP(torch.nn.Module):
    """
    MLP classifier (Classification head):
    BatchNorm -> Linear -> ELU -> BatchNorm -> Linear
    """
    def __init__(self, gnn_out_dim, num_classes):
        super().__init__()
        self.readout = Sequential(
            BatchNorm1d(gnn_out_dim),
            Linear(gnn_out_dim, gnn_out_dim),
            ELU(),
            BatchNorm1d(gnn_out_dim),
            Linear(gnn_out_dim, num_classes) # Logits
        )

    def forward(self, x):
        logits = self.readout(x)
        return logits


class GraphNN(pl.LightningModule):
    """General class for graph representation learning"""
    # Initialization of the model
    def __init__(
        self,
        gnn_backbone: Type[nn.Module],
        in_channels: int,
        in_channels_e: int,
        num_classes: int,
        optimizer_hparams: Dict[str, Any]
    ):
        super().__init__()
        # In/Out Channels 
        self.in_channels = in_channels
        self.in_channels_e = in_channels_e
        self.num_classes = num_classes

        # Hyperparameters
        self.backbone_config = {**GNN_encoders_hparams[gnn_backbone.__name__]} # Backbone
        self.optimizer_config = {**optimizer_hparams} # Optimizer

        # GNN Backbone initialization
        gnn_kwargs = get_gnn_kwargs(
            in_channels=in_channels,
            in_channels_e=in_channels_e,
            model_hparams=self.backbone_config,
            gnn_backbone_name=gnn_backbone.__name__
        )
        self.gnn = gnn_backbone(**gnn_kwargs)

        # MLP classifier (Classification head) (Readout):
        # BatchNorm -> Linear -> ELU -> BatchNorm -> Linear
        gnn_out_dim = self.backbone_config["gnn_out_channels"] # Dimension of the embedding space
        self.mlp = MLP(gnn_out_dim, num_classes)

        # Metrics
        self.val_metrics = self.build_metrics()
        self.test_metrics = self.build_metrics()

    # Metrics 
    def build_metrics(self):
        if self.num_classes == 2:
            metrics = MetricCollection({
                "auroc": BinaryAUROC(),
                "auprc": BinaryAveragePrecision(),
                "accuracy": Accuracy(task="binary")
            })
        else:
            metrics = MetricCollection({
                "auroc": MulticlassAUROC(num_classes=self.num_classes),
                "auprc": MulticlassAveragePrecision(num_classes=self.num_classes),
                "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes)
            })
        return metrics
    
    # Prediction step: logits -> probs
    def predict_probabilities(self, logits):
        # Predictions
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1] if self.num_classes == 2 else probs
    
    # Validation
    def on_validation_epoch_end(self):
        # Compute metrics
        metrics = self.val_metrics.compute()
        #print(f"Val metrics computed: {metrics}")

        # Log metrics
        self.log_dict({f"val_{k}": v for k, v in metrics.items()})
        
        # Reset metrics
        self.val_metrics.reset()
    
    # Test
    def on_test_epoch_end(self):
        # Compute metrics
        metrics = self.test_metrics.compute()
        
        # Log metrics
        self.log_dict({f"test_{k}": v for k, v in metrics.items()})

        # Reset metrics
        self.test_metrics.reset()

    # Configure the optimizer    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config["lr"],
            weight_decay=self.optimizer_config["wd"]
        )
        return optimizer