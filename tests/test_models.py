# Copyright (c) 2025 Bangyen Pham
"""Test suite for model creation and management functions."""

import pytest
import torch
from torch import nn

from zsharp.models import get_model


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

    def test_get_model_resnet56(self):
        """Test get_model with the CIFAR-style resnet56"""
        model = get_model("resnet56", num_classes=10)

        assert isinstance(model, nn.Module)

        # CIFAR ResNets take 32x32 inputs, not the ImageNet 224x224.
        output = model(torch.randn(1, 3, 32, 32))
        assert output.shape == (1, 10)

        # ResNet-56 is 6n+2 with n=9, and He et al. report 0.85M params.
        conv_and_fc = sum(
            1 for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
        )
        assert conv_and_fc == 56
        params = sum(p.numel() for p in model.parameters())
        assert 0.8e6 < params < 0.9e6

    def test_get_model_resnet110(self):
        """Test get_model with the CIFAR-style resnet110"""
        model = get_model("resnet110", num_classes=100)

        assert isinstance(model, nn.Module)
        output = model(torch.randn(1, 3, 32, 32))
        assert output.shape == (1, 100)

        # 6n+2 with n=18, reported at 1.7M params.
        conv_and_fc = sum(
            1 for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
        )
        assert conv_and_fc == 110
        params = sum(p.numel() for p in model.parameters())
        assert 1.6e6 < params < 1.8e6

    def test_get_model_vgg16_bn(self):
        """Test get_model with the CIFAR-adapted vgg16_bn"""
        model = get_model("vgg16_bn", num_classes=10)

        assert isinstance(model, nn.Module)

        # Runs natively on 32x32, unlike the torchvision ImageNet VGG.
        output = model(torch.randn(1, 3, 32, 32))
        assert output.shape == (1, 10)

        # A single 512-unit classifier, not torchvision's 4096-wide stack,
        # so the model is ~15M parameters rather than ~134M.
        assert isinstance(model.classifier, nn.Linear)
        assert model.classifier.in_features == 512
        params = sum(p.numel() for p in model.parameters())
        assert params < 20e6

    @pytest.mark.parametrize(
        ("name", "num_heads", "mlp_hidden"),
        [("vit_7_8_8_384", 8, 384), ("vit_7_8_12_768", 12, 768)],
    )
    def test_get_model_paper_vit(self, name, num_heads, mlp_hidden):
        """Test the paper's compact ViT variants.

        Both variants use 8 patches per side and differ in head count and
        MLP width, so both must run at the native CIFAR resolution.
        """
        model = get_model(name, num_classes=10, image_size=32)

        assert isinstance(model, nn.Module)
        assert model.patches_per_side == 8
        assert len(model.encoder.layers) == 7

        layer = model.encoder.layers[0]
        assert layer.self_attn.num_heads == num_heads
        assert layer.linear1.out_features == mlp_hidden

        output = model(torch.randn(2, 3, 32, 32))
        assert output.shape == (2, 10)

    def test_paper_vit_patch_size_follows_image_size(self):
        """Test that patch size is derived from the input resolution.

        The paper's third field is patches per side, so a 64x64
        Tiny-ImageNet input yields 8x8 pixel patches rather than 4x4.
        """
        assert get_model("vit_7_8_8_384", image_size=32).patch_size == 4
        assert get_model("vit_7_8_8_384", image_size=64).patch_size == 8

    def test_get_model_unknown_model(self):
        """Test get_model with unknown model raises error"""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("unknown_model", num_classes=10)
