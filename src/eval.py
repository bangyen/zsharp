"""Evaluation utilities for model assessment.

This module provides functions for evaluating model performance
on test datasets, including accuracy computation.
"""

from __future__ import annotations

import torch
import torch.nn
import torch.utils.data

from src.constants import PERCENTAGE_MULTIPLIER


def evaluate_model(
    model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader[torch.Tensor],
    device: torch.device | str,
) -> float:
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
    try:
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully
        return 0.0
    return PERCENTAGE_MULTIPLIER * correct / total if total > 0 else 0.0
