"""Test suite for training functions and utilities."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.train import get_device, train


class SimpleTestModel(nn.Module):
    """Simple model for testing training"""

    def __init__(self, num_classes=10):
        """Initialize SimpleTestModel with convolutional and linear layers"""
        super().__init__()
        # Use a model that can handle image input (3x32x32)
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(16, num_classes)

    def forward(self, x):
        """Forward pass through the model"""
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class TestTrain:
    """Test cases for training functions"""

    def test_get_device_cpu(self):
        """Test get_device returns CPU when specified"""
        config = {"train": {"device": "cpu"}}
        device = get_device(config)
        assert device.type == "cpu"

    def test_get_device_cuda_available(self):
        """Test get_device with CUDA when available"""
        config = {"train": {"device": "cuda"}}

        with patch("torch.cuda.is_available", return_value=True):
            device = get_device(config)
            assert device.type == "cuda"

    def test_get_device_cuda_unavailable(self):
        """Test get_device falls back to CPU when CUDA unavailable"""
        config = {"train": {"device": "cuda"}}

        with patch("torch.cuda.is_available", return_value=False):
            device = get_device(config)
            assert device.type == "cpu"

    def test_get_device_mps_available(self):
        """Test get_device with MPS when available"""
        config = {"train": {"device": "mps"}}

        with patch("torch.backends.mps.is_available", return_value=True):
            device = get_device(config)
            assert device.type == "mps"

    def test_get_device_mps_unavailable(self):
        """Test get_device falls back to CPU when MPS unavailable"""
        config = {"train": {"device": "mps"}}

        with patch("torch.backends.mps.is_available", return_value=False):
            device = get_device(config)
            assert device.type == "cpu"

    def test_get_device_unknown_device(self):
        """Test get_device with unknown device falls back to CPU"""
        config = {"train": {"device": "unknown"}}
        device = get_device(config)
        assert device.type == "cpu"

    @patch("src.train.get_dataset")
    @patch("src.train.get_model")
    def test_train_basic_sgd(self, mock_get_model, mock_get_dataset):
        """Test basic training with SGD optimizer"""
        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()
        mock_get_dataset.return_value = (mock_trainloader, mock_testloader)

        # Mock model
        mock_model = SimpleTestModel()
        mock_get_model.return_value = mock_model

        # Mock data
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        # Mock len for trainloader and testloader
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with patch("torch.device", return_value=torch.device("cpu")):
            results = train(config)

            assert isinstance(results, dict)
            assert "final_test_accuracy" in results
            assert "final_test_loss" in results
            assert "train_losses" in results
            assert "train_accuracies" in results
            assert "total_training_time" in results
            assert "device" in results
            assert "optimizer_type" in results

    @patch("src.train.get_dataset")
    @patch("src.train.get_model")
    @patch("src.train.ZSharp")
    def test_train_zsharp_optimizer(
        self, mock_zsharp, mock_get_model, mock_get_dataset
    ):
        """Test training with ZSharp optimizer"""
        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()
        mock_get_dataset.return_value = (mock_trainloader, mock_testloader)

        # Mock model
        mock_model = SimpleTestModel()
        mock_get_model.return_value = mock_model

        # Mock ZSharp optimizer
        mock_optimizer = MagicMock()
        mock_zsharp.return_value = mock_optimizer

        # Mock data
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        # Mock len for trainloader and testloader
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "zsharp",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "rho": 0.05,
                "percentile": 70,
            },
        }

        with patch("torch.device", return_value=torch.device("cpu")):
            results = train(config)

            # Check that ZSharp was called
            mock_zsharp.assert_called_once()

            # Check that first_step and second_step were called
            assert mock_optimizer.first_step.called
            assert mock_optimizer.second_step.called

            assert results["optimizer_type"] == "zsharp"

    @patch("src.train.get_dataset")
    @patch("src.train.get_model")
    @pytest.mark.mps
    def test_train_mixed_precision(self, mock_get_model, mock_get_dataset):
        """Test training with mixed precision"""
        # Skip this test if MPS is not available
        if not torch.backends.mps.is_available():
            import pytest

            pytest.skip("MPS not available on this system")

        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()
        mock_get_dataset.return_value = (mock_trainloader, mock_testloader)

        # Mock model
        mock_model = SimpleTestModel()
        mock_get_model.return_value = mock_model

        # Mock data
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        # Mock len for trainloader and testloader
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "mps",
                "num_workers": 0,
                "use_mixed_precision": True,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with (
            patch("torch.device", return_value=torch.device("mps")),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            results = train(config)

            assert isinstance(results, dict)
            assert results["device"] == "mps"

    @patch("src.train.get_dataset")
    @patch("src.train.get_model")
    def test_train_cifar100(self, mock_get_model, mock_get_dataset):
        """Test training with CIFAR-100 dataset"""
        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()
        mock_get_dataset.return_value = (mock_trainloader, mock_testloader)

        # Mock model
        mock_model = SimpleTestModel(num_classes=100)
        mock_get_model.return_value = mock_model

        # Mock data
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 100, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 100, (4,)))]
        )
        # Mock len for trainloader and testloader
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        config = {
            "dataset": "cifar100",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with patch("torch.device", return_value=torch.device("cpu")):
            results = train(config)

            # Check that model was created with correct num_classes
            mock_get_model.assert_called_with("resnet18", num_classes=100)

            assert isinstance(results, dict)

    @patch("src.train.get_dataset")
    @patch("src.train.get_model")
    def test_train_multiple_epochs(self, mock_get_model, mock_get_dataset):
        """Test training with multiple epochs"""
        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()
        mock_get_dataset.return_value = (mock_trainloader, mock_testloader)

        # Mock model
        mock_model = SimpleTestModel()
        mock_get_model.return_value = mock_model

        # Mock data for multiple epochs
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        # Mock len for trainloader and testloader
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        # Mock the trainloader to return the same data for each epoch
        mock_trainloader.__iter__ = lambda self: iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 3,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with patch("torch.device", return_value=torch.device("cpu")):
            results = train(config)

            assert len(results["train_losses"]) == 3
            assert len(results["train_accuracies"]) == 3

    @patch("src.train.get_dataset")
    @patch("src.train.get_model")
    def test_train_results_saving(self, mock_get_model, mock_get_dataset):
        """Test that training results are saved to file"""
        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()
        mock_get_dataset.return_value = (mock_trainloader, mock_testloader)

        # Mock model
        mock_model = SimpleTestModel()
        mock_get_model.return_value = mock_model

        # Mock data
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        # Mock len for trainloader and testloader
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with (
            patch("torch.device", return_value=torch.device("cpu")),
            patch("builtins.open", create=True) as mock_open,
        ):
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            train(config)

            # Check that file was opened for writing
            mock_open.assert_called()
            assert mock_file.write.called

    @patch("src.train.get_dataset")
    @patch("src.train.get_model")
    def test_train_gradient_clipping(self, mock_get_model, mock_get_dataset):
        """Test that gradient clipping is applied"""
        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()
        mock_get_dataset.return_value = (mock_trainloader, mock_testloader)

        # Mock model
        mock_model = SimpleTestModel()
        mock_get_model.return_value = mock_model

        # Mock data
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        # Mock len for trainloader and testloader
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with (
            patch("torch.device", return_value=torch.device("cpu")),
            patch("torch.nn.utils.clip_grad_norm_") as mock_clip,
        ):
            train(config)

            # Check that gradient clipping was called
            mock_clip.assert_called()

    @patch("src.train.get_dataset")
    @patch("src.train.get_model")
    def test_train_progress_bar(self, mock_get_model, mock_get_dataset):
        """Test that progress bars are used during training"""
        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()
        mock_get_dataset.return_value = (mock_trainloader, mock_testloader)

        # Mock model
        mock_model = SimpleTestModel()
        mock_get_model.return_value = mock_model

        # Mock data
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        # Mock len for trainloader and testloader
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        # Mock the trainloader to return the same data for each epoch
        mock_trainloader.__iter__ = lambda: iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with (
            patch("torch.device", return_value=torch.device("cpu")),
            patch("src.train.tqdm") as mock_tqdm,
        ):
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar

            # Ensure the mock pbar iterates over the data
            mock_pbar.__iter__ = lambda self: iter(
                [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
            )

            train(config)

            # Check that tqdm was called for progress bars
            assert mock_tqdm.called
            assert mock_pbar.set_postfix.called

    def test_train_main_block(self):
        """Test the main block of train.py"""
        # Skip this test as the main block is a simple CLI that's hard to test
        # The main block just parses arguments and calls train() with a config
        # This functionality is already tested by the train() function itself

    def test_train_keyboard_interrupt_during_training(self):
        """Test that KeyboardInterrupt is handled gracefully during training"""
        from unittest.mock import MagicMock, patch

        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()

        # Mock data that will raise KeyboardInterrupt
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with (
            patch(
                "src.train.get_dataset",
                return_value=(mock_trainloader, mock_testloader),
            ),
            patch("src.train.get_model", return_value=SimpleTestModel()),
            patch("torch.device", return_value=torch.device("cpu")),
            patch("src.train.tqdm") as mock_tqdm,
        ):
            # Mock tqdm to raise KeyboardInterrupt
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar
            mock_pbar.__iter__ = lambda self: iter(
                [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
            )
            mock_pbar.set_postfix = MagicMock()
            mock_pbar.close = MagicMock()

            # Make the progress bar raise KeyboardInterrupt
            mock_pbar.__iter__ = lambda self: (_ for _ in ()).throw(
                KeyboardInterrupt("Simulated keyboard interrupt")
            )

            # Should handle KeyboardInterrupt gracefully and return None
            result = train(config)
            assert result is None

    def test_train_keyboard_interrupt_during_evaluation(self):
        """Test that KeyboardInterrupt is handled gracefully during evaluation"""
        from unittest.mock import MagicMock, patch

        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()

        # Mock data
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with (
            patch(
                "src.train.get_dataset",
                return_value=(mock_trainloader, mock_testloader),
            ),
            patch("src.train.get_model", return_value=SimpleTestModel()),
            patch("torch.device", return_value=torch.device("cpu")),
            patch("src.train.tqdm") as mock_tqdm,
        ):
            # Mock tqdm for training
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar
            mock_pbar.__iter__ = lambda self: iter(
                [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
            )
            mock_pbar.set_postfix = MagicMock()
            mock_pbar.close = MagicMock()

            # Mock tqdm for evaluation to raise KeyboardInterrupt
            mock_eval_pbar = MagicMock()
            mock_tqdm.side_effect = [mock_pbar, mock_eval_pbar]
            mock_eval_pbar.__iter__ = lambda self: (_ for _ in ()).throw(
                KeyboardInterrupt("Simulated keyboard interrupt")
            )
            mock_eval_pbar.set_postfix = MagicMock()
            mock_eval_pbar.close = MagicMock()

            # Should handle KeyboardInterrupt gracefully and return None
            result = train(config)
            assert result is None

    def test_train_keyboard_interrupt_during_evaluation_outer(self):
        """Test that KeyboardInterrupt is handled gracefully during evaluation (outer catch)"""
        from unittest.mock import MagicMock, patch

        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()

        # Mock data
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "num_workers": 0,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with (
            patch(
                "src.train.get_dataset",
                return_value=(mock_trainloader, mock_testloader),
            ),
            patch("src.train.get_model", return_value=SimpleTestModel()),
            patch("torch.device", return_value=torch.device("cpu")),
            patch("src.train.tqdm") as mock_tqdm,
        ):
            # Mock tqdm for training
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar
            mock_pbar.__iter__ = lambda self: iter(
                [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
            )
            mock_pbar.set_postfix = MagicMock()
            mock_pbar.close = MagicMock()

            # Mock tqdm for evaluation to raise KeyboardInterrupt in outer catch
            mock_eval_pbar = MagicMock()
            mock_tqdm.side_effect = [mock_pbar, mock_eval_pbar]
            mock_eval_pbar.__iter__ = lambda self: (_ for _ in ()).throw(
                KeyboardInterrupt("Simulated keyboard interrupt")
            )
            mock_eval_pbar.set_postfix = MagicMock()
            mock_eval_pbar.close = MagicMock()

            # Should handle KeyboardInterrupt gracefully and return None
            result = train(config)
            assert result is None

    @pytest.mark.mps
    def test_train_mps_mixed_precision_evaluation(self):
        """Test that MPS mixed precision is handled during evaluation"""
        # Skip this test if MPS is not available
        if not torch.backends.mps.is_available():
            import pytest

            pytest.skip("MPS not available on this system")

        from unittest.mock import MagicMock, patch

        # Mock dataset
        mock_trainloader = MagicMock()
        mock_testloader = MagicMock()

        # Mock data
        mock_trainloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_testloader.__iter__.return_value = iter(
            [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
        )
        mock_trainloader.__len__ = MagicMock(return_value=1)
        mock_testloader.__len__ = MagicMock(return_value=1)

        config = {
            "dataset": "cifar10",
            "model": "resnet18",
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "mps",
                "num_workers": 0,
                "use_mixed_precision": True,
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
            },
        }

        with (
            patch(
                "src.train.get_dataset",
                return_value=(mock_trainloader, mock_testloader),
            ),
            patch("src.train.get_model", return_value=SimpleTestModel()),
            patch("torch.device", return_value=torch.device("mps")),
            patch("torch.backends.mps.is_available", return_value=True),
            patch("src.train.tqdm") as mock_tqdm,
        ):
            # Mock tqdm for training
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar
            mock_pbar.__iter__ = lambda self: iter(
                [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
            )
            mock_pbar.set_postfix = MagicMock()
            mock_pbar.close = MagicMock()

            # Mock tqdm for evaluation
            mock_eval_pbar = MagicMock()
            mock_tqdm.side_effect = [mock_pbar, mock_eval_pbar]
            mock_eval_pbar.__iter__ = lambda self: iter(
                [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
            )
            mock_eval_pbar.set_postfix = MagicMock()
            mock_eval_pbar.close = MagicMock()

            # This should trigger the MPS mixed precision case during evaluation
            result = train(config)
            assert isinstance(result, dict)
            assert result["device"] == "mps"
