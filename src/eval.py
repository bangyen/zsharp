"""Evaluation utilities for model assessment.

This module provides functions for evaluating model performance
on test datasets, including accuracy computation.
"""

import torch

from src.constants import PERCENTAGE_MULTIPLIER


def evaluate_model(model, testloader, device):
    """Evaluate model on test set.

    Args:
        model: PyTorch model to evaluate
        testloader: DataLoader for test data
        device: Device to run evaluation on (cpu/cuda)

    Returns:
        float: Test accuracy as a percentage
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return PERCENTAGE_MULTIPLIER * correct / total
