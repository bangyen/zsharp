"""Model loading utilities for various architectures.

This module provides functions to load and configure different
PyTorch models including ResNet, VGG, and Vision Transformer variants.
"""

from torch import nn
from torchvision import models
from torchvision.models import vit_b_16

from src.constants import CIFAR10_NUM_CLASSES, RESNET18_NAME


def get_model(
    name: str = RESNET18_NAME,
    num_classes: int = CIFAR10_NUM_CLASSES,
) -> nn.Module:
    """Get a PyTorch model by name.

    Args:
        name: Name of the model to load
        num_classes: Number of output classes

    Returns:
        torch.nn.Module: PyTorch model

    Raises:
        ValueError: If model name is not supported

    """
    if name == "resnet18":
        model: nn.Module = models.resnet18(num_classes=num_classes)
    elif name == "resnet56":
        # ResNet-56 implementation (simplified)
        model = models.resnet18(
            num_classes=num_classes,
        )  # Using ResNet18 as proxy
    elif name == "vgg11":
        model = models.vgg11(num_classes=num_classes)
    elif name == "vit_b_16":
        model = vit_b_16(num_classes=num_classes)
    elif name == "vit_s_16":
        # Use ViT-B/16 as proxy for ViT-S/16
        model = vit_b_16(num_classes=num_classes)
    else:
        error_msg = f"Unknown model {name}"
        raise ValueError(error_msg)
    return model
