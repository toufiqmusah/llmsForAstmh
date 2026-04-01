"""
custom_loss.py - Custom loss function for ASTMH classification

Focuses loss computation on "important" categories only.
"""

import torch
import torch.nn as nn
import numpy as np


class ImportantCategoryLoss(nn.Module):
    """
    Custom loss function that concentrates loss computation on important categories.
    For samples where neither predicted nor target category is important,
    predictions are set to perfect to exclude them from loss.
    """

    def __init__(self, important_class_indices, device="cuda"):
        """
        Args:
            important_class_indices: list of indices of important classes
            device: torch device
        """
        super(ImportantCategoryLoss, self).__init__()
        self.loss = nn.NLLLoss()
        self.device = device
        self.important_class_indices = set(important_class_indices)

    def forward(self, log_probs, target):
        """
        Calculate loss focusing on important categories.

        Args:
            log_probs: torch.Tensor - log softmax probabilities (batch_size, num_classes)
            target: torch.Tensor - target class indices (batch_size,)

        Returns:
            loss: torch.Tensor - computed loss
        """
        batch_size, num_classes = log_probs.shape

        # Convert log probs to probabilities for analysis
        probs = torch.exp(log_probs)

        # For each sample, determine if predicted class is important
        pred_classes = torch.argmax(probs, dim=1)
        pred_is_important = torch.tensor(
            [p.item() in self.important_class_indices for p in pred_classes],
            device=self.device,
            dtype=torch.bool
        )

        # For each sample, determine if target class is important
        target_is_important = torch.tensor(
            [t.item() in self.important_class_indices for t in target],
            device=self.device,
            dtype=torch.bool
        )

        # Samples to keep: predicted OR target is important
        keep_mask = pred_is_important | target_is_important

        # Adjust log_probs for ignored samples: make them perfect predictions
        adjusted_log_probs = log_probs.clone()
        for i in range(batch_size):
            if not keep_mask[i]:
                # Set perfect prediction for this sample
                adjusted_log_probs[i, :] = -1e10
                adjusted_log_probs[i, target[i]] = 0.0

        # Compute loss on adjusted predictions
        loss = self.loss(adjusted_log_probs, target)
        return loss
