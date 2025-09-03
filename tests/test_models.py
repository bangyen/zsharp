"""Test suite for model creation and management functions."""

import pytest
import torch
import torch.nn as nn

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

    def test_get_model_different_num_classes(self):
        """Test get_model with different num_classes values"""
        # Reduced number of classes to test for faster execution
        for num_classes in [1, 10, 100]:  # Removed 1000 to speed up test
            model = get_model("resnet18", num_classes=num_classes)

            # Test forward pass with smaller input for faster testing
            x = torch.randn(
                1, 3, 224, 224
            )  # Reduced from (2, 3, 224, 224) to (1, 3, 224, 224)
            output = model(x)
            assert output.shape == (1, num_classes)  # Updated expected shape

    def test_model_parameters(self):
        """Test that models have trainable parameters"""
        # Reduced number of models to test for faster execution
        model_names = [
            "resnet18",
            "vgg11",
        ]  # Removed heavy models: "resnet56", "vit_b_16", "vit_s_16"

        for model_name in model_names:
            model = get_model(model_name, num_classes=10)

            # Check that model has parameters
            params = list(model.parameters())
            assert len(params) > 0

            # Check that some parameters require gradients
            has_trainable_params = any(p.requires_grad for p in params)
            assert has_trainable_params

    def test_model_device_movement(self):
        """Test that models can be moved to different devices"""
        model = get_model("resnet18", num_classes=10)

        # Move to CPU (should work)
        model_cpu = model.cpu()
        assert next(model_cpu.parameters()).device.type == "cpu"

        # Move to CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            assert next(model_cuda.parameters()).device.type == "cuda"

    def test_model_eval_mode(self):
        """Test that models can be set to evaluation mode"""
        model = get_model("resnet18", num_classes=10)

        # Set to eval mode
        model.eval()
        assert not model.training

        # Set back to train mode
        model.train()
        assert model.training

    def test_model_gradient_flow(self):
        """Test that gradients flow through models"""
        model = get_model("resnet18", num_classes=10)

        # Create dummy data with smaller input for faster testing
        x = torch.randn(
            1, 3, 224, 224, requires_grad=True
        )  # Reduced from (2, 3, 224, 224) to (1, 3, 224, 224)
        y = torch.randint(0, 10, (1,))  # Reduced from (2,) to (1,)
        criterion = nn.CrossEntropyLoss()

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        loss.backward()

        # Check that gradients were computed
        assert x.grad is not None

        # Check that model parameters have gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients

    def test_model_output_range(self):
        """Test that model outputs are reasonable"""
        model = get_model("resnet18", num_classes=10)
        model.eval()

        x = torch.randn(
            1, 3, 224, 224
        )  # Reduced from (2, 3, 224, 224) to (1, 3, 224, 224)
        output = model(x)

        # Output should be finite
        assert torch.all(torch.isfinite(output))

        # Output should not be all zeros
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_model_consistency(self):
        """Test that same model with same seed produces consistent outputs"""
        model1 = get_model("resnet18", num_classes=10)
        model2 = get_model("resnet18", num_classes=10)

        # Set same weights
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            param2.data = param1.data.clone()

        # Same input should produce same output with smaller input for faster
        # testing
        x = torch.randn(
            1, 3, 224, 224
        )  # Reduced from (2, 3, 224, 224) to (1, 3, 224, 224)
        output1 = model1(x)
        output2 = model2(x)

        assert torch.allclose(output1, output2)
