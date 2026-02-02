"""Model loading utilities for various architectures.

This module provides functions to load and configure different
PyTorch models including ResNet, VGG, and Vision Transformer variants.
"""

from typing import cast

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
    model_map = {
        "resnet18": lambda: models.resnet18(num_classes=num_classes),
        "resnet56": lambda: models.resnet18(num_classes=num_classes),
        "vgg11": lambda: models.vgg11(num_classes=num_classes),
        "vit_b_16": lambda: vit_b_16(num_classes=num_classes),
        "vit_s_16": lambda: vit_b_16(num_classes=num_classes),
    }

    if name not in model_map:
        error_msg = f"Unknown model {name}"
        raise ValueError(error_msg)

    return cast("nn.Module", model_map[name]())
