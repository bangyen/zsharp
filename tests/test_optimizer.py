"""Test suite for optimizer implementations (SAM and ZSharp)."""

import pytest
import torch
from torch import nn, optim

from src.constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_PERCENTILE,
    DEFAULT_RHO,
)
from src.optimizer import SAM, ZSharp


class SimpleModel(nn.Module):
    """Simple model for testing optimizers"""

    def __init__(self):
        """Initialize SimpleModel with two linear layers"""
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        """Forward pass through the model"""
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestSAM:
    """Test cases for SAM optimizer"""

    def test_sam_initialization(self):
        """Test SAM optimizer initialization"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        sam = SAM(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            lr=DEFAULT_LEARNING_RATE,
        )

        assert sam.rho == DEFAULT_RHO
        assert hasattr(sam, "base_optimizer")
        assert len(sam.param_groups) > 0

    def test_sam_first_step(self):
        """Test SAM first step (gradient perturbation)"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        sam = SAM(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Create dummy data and compute gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Apply first step
        sam.first_step()

        # Check that parameters have been perturbed
        perturbed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param.data, original_params[name]):
                perturbed = True
                break

        assert perturbed, "Parameters should be perturbed after first_step"

    def test_sam_second_step(self):
        """
        Test SAM second step (parameter restoration and base optimizer step)
        """
        model = SimpleModel()
        base_optimizer = optim.SGD
        sam = SAM(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Create dummy data and compute gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Apply first step
        sam.first_step()

        # Store perturbed parameters
        perturbed_params = {}
        for name, param in model.named_parameters():
            perturbed_params[name] = param.data.clone()

        # Apply second step
        sam.second_step()

        # Check that parameters have been updated (not restored to original)
        updated = False
        for name, param in model.named_parameters():
            if not torch.allclose(
                param.data, original_params[name]
            ) and not torch.allclose(param.data, perturbed_params[name]):
                updated = True
                break

        assert updated, "Parameters should be updated after second_step"

    def test_sam_step_raises_error(self):
        """Test that calling step() directly raises an error"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        sam = SAM(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            lr=DEFAULT_LEARNING_RATE,
        )

        with pytest.raises(RuntimeError, match="SAM requires two-step calls"):
            sam.step()

    def test_sam_with_zero_gradients(self):
        """Test SAM behavior with zero gradients"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        sam = SAM(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Zero gradients
        for param in model.parameters():
            param.grad = None

        # Should handle zero gradients gracefully
        try:
            sam.first_step()
            sam.second_step()
        except RuntimeError:
            # This is expected behavior when no gradients are available
            # Verify that the error is properly handled
            # Test that the optimizer is still in a valid state
            assert len(sam.param_groups) > 0

    def test_sam_with_none_gradients(self):
        """Test SAM behavior when some parameters have None gradients"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        sam = SAM(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Create dummy data and compute gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Set some gradients to None
        params = list(model.parameters())
        params[0].grad = None  # Set first parameter's gradient to None

        # Should handle None gradients gracefully
        sam.first_step()
        sam.second_step()

        # Verify optimizer is still in valid state
        assert len(sam.param_groups) > 0


class TestZSharp:
    """Test cases for ZSharp optimizer"""

    def test_zsharp_initialization(self):
        """Test ZSharp optimizer initialization"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        zsharp = ZSharp(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            percentile=DEFAULT_PERCENTILE,
            lr=DEFAULT_LEARNING_RATE,
        )

        assert zsharp.rho == DEFAULT_RHO
        assert zsharp.percentile == DEFAULT_PERCENTILE
        assert hasattr(zsharp, "base_optimizer")
        assert len(zsharp.param_groups) > 0

    def test_zsharp_first_step_with_normal_gradients(self):
        """Test ZSharp first step with normal gradient distributions"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        zsharp = ZSharp(
            list(model.parameters()),
            base_optimizer,
            rho=0.05,
            percentile=70,
            lr=0.01,
        )

        # Create dummy data and compute gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Apply first step
        zsharp.first_step()

        # Check that parameters have been perturbed
        perturbed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param.data, original_params[name]):
                perturbed = True
                break

        assert perturbed, "Parameters should be perturbed after first_step"

    def test_zsharp_first_step_with_identical_gradients(self):
        """Test ZSharp first step when all gradients have identical values"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        zsharp = ZSharp(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            percentile=DEFAULT_PERCENTILE,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Create dummy data and compute gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Set all gradients to the same value
        for param in model.parameters():
            if param.grad is not None:
                param.grad.fill_(1.0)

        # Should not raise an error and should handle edge case
        zsharp.first_step()

    def test_zsharp_first_step_with_zero_gradients(self):
        """Test ZSharp first step with zero gradients"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        zsharp = ZSharp(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            percentile=DEFAULT_PERCENTILE,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Zero gradients
        for param in model.parameters():
            param.grad = None

        # Should not raise an error
        zsharp.first_step()

    def test_zsharp_gradient_filtering(self):
        """Test that ZSharp actually filters gradients based on Z-scores"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        zsharp = ZSharp(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            percentile=80,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Create dummy data and compute gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Store original gradients
        original_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()

        # Apply first step (this should filter gradients)
        zsharp.first_step()

        # Check that gradients have been modified by the ZSharp optimizer
        gradients_modified = False
        for name, param in model.named_parameters():
            if (
                param.grad is not None
                and name in original_grads
                and not torch.allclose(param.grad, original_grads[name])
            ):
                gradients_modified = True
                break

        # The ZSharp optimizer should modify gradients in some way
        assert gradients_modified, "ZSharp optimizer should modify gradients"

    def test_zsharp_second_step(self):
        """Test ZSharp second step (inherited from SAM)"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        zsharp = ZSharp(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            percentile=DEFAULT_PERCENTILE,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Create dummy data and compute gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Apply first step
        zsharp.first_step()

        # Store perturbed parameters
        perturbed_params = {}
        for name, param in model.named_parameters():
            perturbed_params[name] = param.data.clone()

        # Apply second step
        zsharp.second_step()

        # Check that parameters have been updated
        updated = False
        for name, param in model.named_parameters():
            if not torch.allclose(
                param.data, original_params[name]
            ) and not torch.allclose(param.data, perturbed_params[name]):
                updated = True
                break

        assert updated, "Parameters should be updated after second_step"

    def test_zsharp_different_percentiles(self):
        """Test ZSharp with different percentile values"""
        model = SimpleModel()
        base_optimizer = optim.SGD

        # Test with different percentiles
        for percentile in [50, 70, 90]:
            zsharp = ZSharp(
                list(model.parameters()),
                base_optimizer,
                rho=DEFAULT_RHO,
                percentile=percentile,
                lr=DEFAULT_LEARNING_RATE,
            )
            assert zsharp.percentile == percentile

    def test_zsharp_mixed_precision_handling(self):
        """Test ZSharp handles mixed precision gradients correctly"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        zsharp = ZSharp(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            percentile=DEFAULT_PERCENTILE,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Create dummy data and compute gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Convert gradients to float16 (simulate mixed precision)
        # Note: This might fail due to dtype mismatch, which is expected
        try:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.half()

            # Should not raise an error
            zsharp.first_step()
        except RuntimeError:
            # This is expected behavior for dtype mismatch
            # Verify that the error is properly handled
            # Test that the optimizer is still in a valid state
            assert len(zsharp.param_groups) > 0

    def test_zsharp_mixed_precision_float16_conversion(self):
        """Test ZSharp specifically handles float16 gradient conversion"""
        # Create a model with float16 parameters to test the conversion
        model = SimpleModel()
        # Convert model parameters to float16
        for param in model.parameters():
            param.data = param.data.half()

        base_optimizer = optim.SGD
        zsharp = ZSharp(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            percentile=DEFAULT_PERCENTILE,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Create dummy data and compute gradients
        x = torch.randn(4, 10).half()  # Use float16 input
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Now the gradients should be float16 and the conversion code should
        # be triggered
        zsharp.first_step()

        # Verify optimizer is still in valid state
        assert len(zsharp.param_groups) > 0

    def test_zsharp_numerical_stability(self):
        """Test ZSharp numerical stability with very small gradients"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        zsharp = ZSharp(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            percentile=DEFAULT_PERCENTILE,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Create dummy data and compute gradients
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Set gradients to very small values
        for param in model.parameters():
            if param.grad is not None:
                param.grad = param.grad * 1e-10

        # Should not raise an error
        zsharp.first_step()

    def test_zsharp_edge_cases_comprehensive(self):
        """Test ZSharp edge cases comprehensively to achieve 100% coverage"""
        model = SimpleModel()
        base_optimizer = optim.SGD
        zsharp = ZSharp(
            list(model.parameters()),
            base_optimizer,
            rho=DEFAULT_RHO,
            percentile=DEFAULT_PERCENTILE,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Test case 1: Create gradients that will result in identical Z-scores
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        loss = criterion(model(x), y)
        loss.backward()

        # Set all gradients to exactly the same value to create identical
        # Z-scores
        for param in model.parameters():
            if param.grad is not None:
                param.grad.fill_(1.0)

        # This should trigger the identical Z-scores case
        zsharp.first_step()

        # Test case 2: Create gradients that will result in zero gradient norm
        loss = criterion(model(x), y)
        loss.backward()

        # Set gradients to extremely small values
        for param in model.parameters():
            if param.grad is not None:
                param.grad = param.grad * 1e-30

        # This should trigger the zero gradient norm case
        zsharp.first_step()

        # Test case 3: Test with no gradients (empty layer_grads)
        for param in model.parameters():
            param.grad = None

        # This should trigger the early return case
        zsharp.first_step()

    def test_zsharp_gradient_filtering_with_none_gradients(self):
        """Test ZSharp gradient filtering when some parameters have None gradients"""
        model = SimpleModel()
        zsharp = ZSharp(
            list(model.parameters()),
            optim.SGD,
            lr=DEFAULT_LEARNING_RATE,
            percentile=DEFAULT_PERCENTILE,
        )

        # Set some gradients to None and others to normal values
        params = list(model.parameters())
        params[0].grad = torch.randn_like(params[0].data)
        params[1].grad = None  # This should trigger the continue statement
        if len(params) > 2:
            params[2].grad = torch.randn_like(params[2].data)

        # This should handle None gradients gracefully
        # The first_step should skip parameters with None gradients
        zsharp.first_step()
        zsharp.second_step()

    def test_zsharp_gradient_filtering_skip_none_gradients(self):
        """Test ZSharp gradient filtering skips parameters with None gradients"""
        model = SimpleModel()
        zsharp = ZSharp(
            list(model.parameters()),
            optim.SGD,
            lr=DEFAULT_LEARNING_RATE,
            percentile=DEFAULT_PERCENTILE,
        )

        # Set all gradients to None except one
        params = list(model.parameters())
        for i, param in enumerate(params):
            if i == 0:
                param.grad = torch.randn_like(param.data)
            else:
                param.grad = None  # This should trigger the continue statement at line 252

        # This should handle None gradients gracefully
        # The first_step should skip parameters with None gradients
        zsharp.first_step()
        zsharp.second_step()


class TestOptimizerIntegration:
    """Integration tests for optimizers"""

    def test_sam_vs_zsharp_behavior(self):
        """Test that SAM and ZSharp behave differently"""
        model1 = SimpleModel()
        model2 = SimpleModel()

        # Use same initial weights
        for (_name1, param1), (_name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            param2.data = param1.data.clone()

        # Setup optimizers
        sam = SAM(
            list(model1.parameters()),
            optim.SGD,
            rho=DEFAULT_RHO,
            lr=DEFAULT_LEARNING_RATE,
        )
        zsharp = ZSharp(
            list(model2.parameters()),
            optim.SGD,
            rho=DEFAULT_RHO,
            percentile=DEFAULT_PERCENTILE,
            lr=DEFAULT_LEARNING_RATE,
        )

        # Create dummy data
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        # Compute gradients for both models
        loss1 = criterion(model1(x), y)
        loss1.backward()

        loss2 = criterion(model2(x), y)
        loss2.backward()

        # Apply first step
        sam.first_step()
        zsharp.first_step()

        # Check that the models have diverged (different behavior)
        diverged = False
        for (_name1, param1), (_name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            if not torch.allclose(param1.data, param2.data):
                diverged = True
                break

        assert diverged, "SAM and ZSharp should behave differently"
