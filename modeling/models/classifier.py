"""
classifier.py - Neural network classifier for ASTMH abstracts

Improved architecture with batch normalization, residual connections, and better regularization.
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torchmetrics import Accuracy


class ResidualBlock(nn.Module):
    """Residual block for deeper networks with improved gradient flow."""
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation: nn.Module = None
    ):
        super(ResidualBlock, self).__init__()
        
        if activation is None:
            activation = nn.ReLU()
        
        layers = [
            nn.Linear(in_dim, out_dim),
        ]
        
        if use_batch_norm and not use_layer_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
            
        layers.append(activation)
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(out_dim, out_dim))
        
        if use_batch_norm and not use_layer_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        
        self.network = nn.Sequential(*layers)
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.activation(x + self.network(x))


class ASTMHClassifier(LightningModule):
    """
    PyTorch Lightning classifier for ASTMH abstract classification.
    
    Improvements:
    - Batch normalization for training stability
    - Optional layer normalization
    - Residual connections for deeper networks
    - Optional feature attention mechanism
    - Gradient clipping support
    - Flexible architecture with layer config
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
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_residual: bool = True,
        use_attention: bool = False,
        activation: str = "relu",
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
            use_batch_norm: apply batch normalization after linear layers
            use_layer_norm: apply layer normalization instead of batch norm
            use_residual: use residual connections for layers with matching dims
            use_attention: apply self-attention over features
            activation: activation function ('relu', 'gelu', 'elu')
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
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.loss_fn = loss_fn if loss_fn is not None else nn.NLLLoss()

        # Select activation function
        if activation.lower() == "gelu":
            activation_fn = nn.GELU()
        elif activation.lower() == "elu":
            activation_fn = nn.ELU()
        else:
            activation_fn = nn.ReLU()

        # Optional: Feature attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 8),
                nn.ReLU(),
                nn.Linear(input_dim // 8, input_dim),
                nn.Sigmoid()
            )
        else:
            self.attention = None

        # Build network with improved modules
        layers = []

        # Input layer -> first hidden layer
        layers.append(nn.Linear(input_dim, layer_dims[0]))
        if use_batch_norm and not use_layer_norm:
            layers.append(nn.BatchNorm1d(layer_dims[0]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(layer_dims[0]))
        layers.append(activation_fn)

        # Hidden layers with residual connections
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Dropout(p=dropout))
            
            # Create residual block if dimensions match
            if use_residual and layer_dims[i] == layer_dims[i + 1]:
                layers.append(ResidualBlock(
                    layer_dims[i], 
                    layer_dims[i + 1],
                    dropout,
                    use_batch_norm,
                    use_layer_norm,
                    activation_fn
                ))
            else:
                layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                if use_batch_norm and not use_layer_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(layer_dims[i + 1]))
                layers.append(activation_fn)

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
        # Optional: apply attention to input features
        if self.attention is not None:
            attention_weights = self.attention(x)
            x = x * attention_weights

        logits = self.model(x)
        # Apply temperature scaled log softmax
        log_probs = torch.nn.functional.log_softmax(logits / self.temperature, dim=1)
        return log_probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        # ImportantCategoryLoss needs raw logits, others need log-softmax
        from loss.custom_loss import ImportantCategoryLoss
        if isinstance(self.loss_fn, ImportantCategoryLoss):
            loss = self.loss_fn(logits, y)
        else:
            log_probs = torch.nn.functional.log_softmax(logits / self.temperature, dim=1)
            loss = self.loss_fn(log_probs, y)
        
        # Always compute accuracy from probabilities
        log_probs = torch.nn.functional.log_softmax(logits / self.temperature, dim=1)
        probs = torch.exp(log_probs)
        acc = self.accuracy(probs, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        from loss.custom_loss import ImportantCategoryLoss
        if isinstance(self.loss_fn, ImportantCategoryLoss):
            loss = self.loss_fn(logits, y)
        else:
            log_probs = torch.nn.functional.log_softmax(logits / self.temperature, dim=1)
            loss = self.loss_fn(log_probs, y)
        
        log_probs = torch.nn.functional.log_softmax(logits / self.temperature, dim=1)
        probs = torch.exp(log_probs)
        acc = self.accuracy(probs, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        from loss.custom_loss import ImportantCategoryLoss
        if isinstance(self.loss_fn, ImportantCategoryLoss):
            loss = self.loss_fn(logits, y)
        else:
            log_probs = torch.nn.functional.log_softmax(logits / self.temperature, dim=1)
            loss = self.loss_fn(log_probs, y)
        
        log_probs = torch.nn.functional.log_softmax(logits / self.temperature, dim=1)
        probs = torch.exp(log_probs)
        acc = self.accuracy(probs, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        log_probs = torch.nn.functional.log_softmax(logits / self.temperature, dim=1)
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
        
        # Use ReduceLROnPlateau for simpler scheduling
        # or CosineAnnealingWarmRestarts for more sophisticated scheduling
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=0.5, 
            patience=10, 
            min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
