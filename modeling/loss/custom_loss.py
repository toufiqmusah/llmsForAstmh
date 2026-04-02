"""
custom_loss.py - Custom loss function for ASTMH classification

Focuses loss computation on "important" categories only.
Based on the approach from lossOverImportantClassesOnly_fn_9april2024.py
"""

import torch
import torch.nn as nn
import numpy as np


class ImportantCategoryLoss(nn.Module):
    """
    Custom loss function that concentrates loss computation on important categories.
    
    Strategy:
    1. Takes logits from the network
    2. Applies softmax to get probabilities
    3. Identifies samples where NEITHER predicted NOR target class is important
    4. For those "ignored" samples, sets softmax to near-perfect (1.0 on target, ~0.001 elsewhere)
    5. Takes log of adjusted softmax and applies NLLLoss
    
    This ensures samples outside important classes don't contribute to loss,
    while samples inside important classes receive proper gradient signals.
    """

    def __init__(self, important_class_indices, num_classes=54, device="cuda"):
        super(ImportantCategoryLoss, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.important_class_indices = set(important_class_indices)
        self.num_classes = num_classes
        self.epsilon = 0.001  # Value for non-target classes in ignored samples

    def forward(self, logits, target):
        """
        Calculate loss focusing on important categories.

        Args:
            logits: torch.Tensor - raw model output (batch_size, num_classes)
            target: torch.Tensor - target class indices (batch_size,)

        Returns:
            loss: torch.Tensor - computed loss
        """
        batch_size, num_classes = logits.shape

        # Step 1: Apply softmax to logits
        softmax = torch.softmax(logits, dim=1)  # (batch_size, num_classes)

        # Step 2: Identify predicted class for each sample (argmax of softmax)
        pred_classes = torch.argmax(softmax, dim=1)  # (batch_size,)

        # Step 3: Determine which samples are "important"
        # A sample is important if EITHER its predicted OR target class is important
        pred_is_important = torch.tensor(
            [p.item() in self.important_class_indices for p in pred_classes],
            device=self.device,
            dtype=torch.bool
        )
        
        target_is_important = torch.tensor(
            [t.item() in self.important_class_indices for t in target],
            device=self.device,
            dtype=torch.bool
        )

        keep_samples = pred_is_important | target_is_important  # (batch_size,)
        ignore_samples = ~keep_samples  # Samples to ignore

        # Step 4: Adjust softmax for ignored samples
        # Make them "perfect predictions" so they contribute 0 to loss
        adjusted_softmax = softmax.clone()
        
        if ignore_samples.any():
            # For ignored samples: set epsilon for all classes, then set target class to high prob
            adjusted_softmax[ignore_samples, :] = self.epsilon
            
            # Set target class to near-1 (accounting for epsilon on other classes)
            target_vals = target[ignore_samples]
            target_prob = 1.0 - (num_classes - 1) * self.epsilon
            adjusted_softmax[ignore_samples, target_vals] = target_prob

        # Step 5: Take log (to get log-softmax for NLLLoss)
        log_softmax = torch.log(adjusted_softmax)

        # Step 6: Apply NLLLoss
        loss = self.nll_loss(log_softmax, target)

        return loss
