"""Test suite for model evaluation functions."""

import torch
from torch import nn

from src.eval import evaluate_model


class SimpleTestModel(nn.Module):
    """Simple model for testing evaluation"""

    def __init__(self, num_classes=10):
        """Initialize SimpleTestModel with specified number of classes"""
        super().__init__()
        self.linear = nn.Linear(10, num_classes)

    def forward(self, x):
        """Forward pass through the model"""
        return self.linear(x)


class TestEval:
    """Test cases for evaluation functions"""

    def test_evaluate_model_basic(self):
        """Test basic model evaluation"""
        model = SimpleTestModel(num_classes=10)
        model.eval()

        # Create dummy test data
        x = torch.randn(10, 10)
        y = torch.randint(0, 10, (10,))

        # Create DataLoader-like object
        class MockDataLoader:
            """Mock DataLoader for testing purposes"""

            def __init__(self, x, y):
                """Initialize with data and targets"""
                self.x = x
                self.y = y

            def __iter__(self):
                """Return iterator over data batches"""
                return iter([(self.x, self.y)])

        testloader = MockDataLoader(x, y)
        device = torch.device("cpu")

        accuracy = evaluate_model(model, testloader, device)

        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100

    def test_evaluate_model_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is handled gracefully during evaluation"""
        model = SimpleTestModel(num_classes=10)
        model.eval()

        # Create a mock DataLoader that raises KeyboardInterrupt
        class MockDataLoaderWithInterrupt:
            """Mock DataLoader that raises KeyboardInterrupt"""

            def __init__(self):
                """Initialize the mock DataLoader"""

            def __iter__(self):
                """Return iterator that raises KeyboardInterrupt"""
                raise KeyboardInterrupt("Simulated keyboard interrupt")

        testloader = MockDataLoaderWithInterrupt()
        device = torch.device("cpu")

        # Should handle KeyboardInterrupt gracefully and return 0.0
        accuracy = evaluate_model(model, testloader, device)
        assert accuracy == 0.0
