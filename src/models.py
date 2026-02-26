"""Model loading utilities for various architectures.

This module provides functions to load and configure different
PyTorch models including ResNet, VGG, and Vision Transformer variants.
"""

from typing import cast

from torch import nn
from torchvision import models
from torchvision.models import vit_b_16

from src.constants import RESNET18_NAME


def get_model(
    model_name: str = RESNET18_NAME,
    num_classes: int = 10,
) -> nn.Module:
    """Get a PyTorch model by name.

    Args:
        model_name: Name of the model to load
        num_classes: Number of output classes

    Returns:
        torch.nn.Module: PyTorch model

    Raises:
        ValueError: If model name is not supported

    """
    model_map = {
        "resnet18": lambda: models.resnet18(num_classes=num_classes),
        "vgg11": lambda: models.vgg11(num_classes=num_classes),
        "vit_b_16": lambda: vit_b_16(num_classes=num_classes),
    }

    if model_name not in model_map:
        error_msg = f"Unknown model {model_name}"
        raise ValueError(error_msg)

    return cast("nn.Module", model_map[model_name]())
