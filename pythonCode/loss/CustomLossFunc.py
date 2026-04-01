"""
nn.Module for custom loss function for ASTMH abstract sorting problem.
"""
import torch
from torch import nn
import numpy as np

class CustomLossFunc(nn.Module):
    """
    """
    def __init__(self):
        super(CustomLossFunc, self).__init__()

        self.loss = nn.NLLLoss()

    def forward(self, logits, target):
        """
        Parameters:
            output: torch.Tensor
            target: torch.Tensor

        Returns:
            loss: torch.Tensor - loss.
        """
        ex = torch.exp(logits)
        summ = torch.sum(ex, axis=0)

        return loss
