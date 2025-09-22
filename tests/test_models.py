"""Test suite for model creation and management functions."""

import pytest
import torch
from torch import nn

from src.models import get_model


class TestModels:
    """Test cases for model loading functions"""

    def test_get_model_resnet18(self):
        """Test get_model with resnet18"""
        model = get_model("resnet18", num_classes=10)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "fc")  # ResNet has final classification layer

        # Test forward pass with smaller input for faster testing
        x = torch.randn(
            1, 3, 224, 224
        )  # Reduced from (2, 3, 224, 224) to (1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 10)  # Updated expected shape

    def test_get_model_resnet56(self):
        """Test get_model with resnet56"""
        model = get_model("resnet56", num_classes=100)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "fc")

        # Test forward pass with smaller input for faster testing
        x = torch.randn(
            1, 3, 224, 224
        )  # Reduced from (2, 3, 224, 224) to (1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 100)  # Updated expected shape

    def test_get_model_vgg11(self):
        """Test get_model with vgg11"""
        model = get_model("vgg11", num_classes=10)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "classifier")  # VGG has classifier attribute

        # Test forward pass with smaller input for faster testing
        x = torch.randn(
            1, 3, 224, 224
        )  # Reduced from (2, 3, 224, 224) to (1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 10)  # Updated expected shape

    def test_get_model_vit_b_16(self):
        """Test get_model with vit_b_16"""
        model = get_model("vit_b_16", num_classes=10)

        assert isinstance(model, nn.Module)

        # Test forward pass with smaller input for faster testing
        x = torch.randn(
            1, 3, 224, 224
        )  # Reduced from (2, 3, 224, 224) to (1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 10)  # Updated expected shape

    def test_get_model_vit_s_16(self):
        """Test get_model with vit_s_16"""
        model = get_model("vit_s_16", num_classes=10)

        assert isinstance(model, nn.Module)

        # Test forward pass with smaller input for faster testing
        x = torch.randn(
            1, 3, 224, 224
        )  # Reduced from (2, 3, 224, 224) to (1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 10)  # Updated expected shape

    def test_get_model_unknown_model(self):
        """Test get_model with unknown model raises error"""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("unknown_model", num_classes=10)
