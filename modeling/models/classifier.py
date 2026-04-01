"""
classifier.py - Neural network classifier for ASTMH abstracts

Simple feedforward network with configurable layers and dropout.
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy


class ASTMHClassifier(LightningModule):
    """
    PyTorch Lightning classifier for ASTMH abstract classification.
    """

    def __init__(
        self,
        input_dim: int = 768,
        layer_dims: list = None,
        num_classes: int = 54,
        learning_rate: float = 1e-4,
        dropout: float = 0.4,
        loss_fn: nn.Module = None,
        temperature: float = 1.0,
    ):
        """
        Args:
            input_dim: dimension of input embeddings (default 768)
            layer_dims: list of hidden layer dimensions (e.g., [512, 256])
            num_classes: number of output classes
            learning_rate: learning rate for optimizer
            dropout: dropout probability
            loss_fn: custom loss function (default: NLLLoss)
            temperature: temperature for softmax
        """
        super(ASTMHClassifier, self).__init__()

        if layer_dims is None:
            layer_dims = [512, 256]

        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout_prob = dropout
        self.temperature = temperature
        self.loss_fn = loss_fn if loss_fn is not None else nn.NLLLoss()

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, layer_dims[0]))
        layers.append(nn.ReLU())

        # Hidden layers with dropout
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(layer_dims[-1], num_classes))

        self.model = nn.Sequential(*layers)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters(ignore=["loss_fn"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: input tensor (batch_size, input_dim)

        Returns:
            log softmax output (batch_size, num_classes)
        """
        logits = self.model(x)
        # Apply temperature scaled log softmax
        log_probs = torch.nn.functional.log_softmax(logits / self.temperature, dim=1)
        return log_probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x)
        loss = self.loss_fn(log_probs, y)
        probs = torch.exp(log_probs)
        acc = self.accuracy(probs, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x)
        loss = self.loss_fn(log_probs, y)
        probs = torch.exp(log_probs)
        acc = self.accuracy(probs, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x)
        loss = self.loss_fn(log_probs, y)
        probs = torch.exp(log_probs)
        acc = self.accuracy(probs, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        log_probs = self.forward(x)
        probs = torch.exp(log_probs)
        pred_classes = torch.argmax(probs, dim=1)
        return {
            "predictions": pred_classes,
            "probabilities": probs,
            "log_probs": log_probs,
            "targets": y,
        }

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
