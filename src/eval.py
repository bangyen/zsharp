# Placeholder for evaluation utilities
# import torch.nn as nn
import torch


def evaluate_model(model, testloader, device):
    """Evaluate model on test set"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total
