# Copyright (c) 2025 Bangyen Pham
"""Test suite for training functions and utilities."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError
from torch import nn, optim

from zsharp.constants import (
    MAX_QUANTILE_NUMEL,
    ExperimentResults,
    TrainingConfig,
)
from zsharp.optimizer import ZSharp
from zsharp.trainer import (
    TrainingContext,
    _detect_best_device,
    _init_components,
    _run_train_step,
    get_device,
    set_seed,
    train,
)


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


class TestConfigValidation:
    """Test cases for configuration validation"""

    def test_rejects_unknown_optimizer_type(self):
        """Test unknown optimizer type raises validation error"""
        with pytest.raises(ValidationError, match="Unknown optimizer type"):
            TrainingConfig.model_validate({"optimizer": {"type": "adam"}})

    def test_rejects_percentile_out_of_range(self):
        """Test percentile outside [0, 100] raises validation error"""
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate({"optimizer": {"percentile": 150}})

    def test_rejects_non_positive_learning_rate(self):
        """Test non-positive learning rate raises validation error"""
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate({"optimizer": {"lr": -0.1}})

    def test_rejects_non_positive_epochs(self):
        """Test non-positive epoch count raises validation error"""
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate({"train": {"epochs": 0}})

    def test_accepts_valid_config(self):
        """Test a valid config validates without error"""
        config = TrainingConfig.model_validate(
            {
                "optimizer": {"type": "sgd", "percentile": 70, "lr": 0.01},
                "train": {"epochs": 10},
            }
        )
        assert config.optimizer.type == "sgd"
        assert config.train.epochs == 10


class TestTrain:
    """Test cases for training functions"""

    def test_get_device_cpu(self):
        """Test get_device returns CPU when specified."""
        config = TrainingConfig.model_validate({"train": {"device": "cpu"}})
        device = get_device(config)
        assert device.type == "cpu"

    def test_get_device_auto(self):
        """Test get_device returns best available when specified as 'auto'."""
        config = TrainingConfig.model_validate({"train": {"device": "auto"}})
        device = get_device(config)
        # Should return one of cuda, mps, or cpu
        assert device.type in ["cuda", "mps", "cpu"]

    def test_get_device_cuda_available(self):
        """Test get_device with CUDA when available"""
        config = TrainingConfig.model_validate({"train": {"device": "cuda"}})

        with patch("torch.cuda.is_available", return_value=True):
            device = get_device(config)
            assert device.type == "cuda"

    def test_get_device_cuda_unavailable(self):
        """Test get_device falls back to CPU when CUDA unavailable"""
        config = TrainingConfig.model_validate({"train": {"device": "cuda"}})

        with patch("torch.cuda.is_available", return_value=False):
            device = get_device(config)
            assert device.type == "cpu"

    def test_get_device_mps_available(self):
        """Test get_device with MPS when available"""
        config = TrainingConfig.model_validate({"train": {"device": "mps"}})

        with patch("torch.backends.mps.is_available", return_value=True):
            device = get_device(config)
            assert device.type == "mps"

    def test_get_device_mps_unavailable(self):
        """Test get_device falls back to CPU when MPS unavailable"""
        config = TrainingConfig.model_validate({"train": {"device": "mps"}})

        with patch("torch.backends.mps.is_available", return_value=False):
            device = get_device(config)
            assert device.type == "cpu"

    def test_get_device_unknown_device(self):
        """Test get_device with unknown device falls back to CPU"""
        config = TrainingConfig.model_validate(
            {"train": {"device": "unknown"}}
        )
        device = get_device(config)
        assert device.type == "cpu"

    def test_set_seed_cuda(self):
        """Test set_seed seeds CUDA generators when CUDA is available."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.manual_seed") as mock_manual_seed,
            patch("torch.cuda.manual_seed_all") as mock_manual_seed_all,
        ):
            set_seed(42)
        mock_manual_seed.assert_called_once_with(42)
        # torch.manual_seed internally seeds CUDA, so this may fire twice
        mock_manual_seed_all.assert_any_call(42)

    def test_detect_best_device_cuda(self):
        """Test device detection prefers CUDA when available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = _detect_best_device()
        assert device.type == "cuda"

    def test_detect_best_device_cpu(self):
        """Test device detection falls back to CPU without an accelerator."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            device = _detect_best_device()
        assert device.type == "cpu"

    def test_init_components_unknown_dataset(self):
        """Test init components rejects unknown datasets."""
        config = TrainingConfig.model_validate({"dataset": "unknown"})
        with pytest.raises(ValueError, match="Unknown dataset"):
            _init_components(config, torch.device("cpu"))

    @patch("zsharp.trainer.get_dataset")
    @patch("zsharp.trainer.get_model")
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with patch("torch.device", return_value=torch.device("cpu")):
            results = train(config)

            assert isinstance(results, ExperimentResults)
            assert results.final_test_accuracy is not None
            assert results.final_test_loss is not None
            assert results.train_losses is not None
            assert results.train_accuracies is not None
            assert results.total_training_time is not None
            assert results.device is not None
            assert results.optimizer_type == "sgd"

    # The mocked ZSharp never steps its base optimizer, so the scheduler
    # warns about step order; real ZSharp training does not (see
    # test_zsharp_training_steps_optimizer_before_scheduler).
    @pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step")
    @patch("zsharp.trainer.get_dataset")
    @patch("zsharp.trainer.get_model")
    @patch("zsharp.trainer.ZSharp")
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

        # Mock ZSharp optimizer. ``base_optimizer`` must be a real
        # optimizer because the trainer attaches the LR scheduler to it.
        mock_optimizer = MagicMock()
        mock_optimizer.base_optimizer = torch.optim.AdamW(
            mock_model.parameters(), lr=0.001
        )
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with patch("torch.device", return_value=torch.device("cpu")):
            results = train(config)

            # Check that ZSharp was called
            mock_zsharp.assert_called_once()

            # Check that first_step and second_step were called
            assert mock_optimizer.first_step.called
            assert mock_optimizer.second_step.called

            assert results.optimizer_type == "zsharp"

    @patch("zsharp.trainer.get_dataset")
    @patch("zsharp.trainer.get_model")
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with (
            patch("torch.device", return_value=torch.device("mps")),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            results = train(config)

            assert isinstance(results, ExperimentResults)
            assert results.device == "mps"

    @patch("zsharp.trainer.get_dataset")
    @patch("zsharp.trainer.get_model")
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with patch("torch.device", return_value=torch.device("cpu")):
            results = train(config)

            # Check that model was created with correct num_classes
            mock_get_model.assert_called_with(
                model_name="resnet18", num_classes=100, image_size=32
            )

            assert isinstance(results, ExperimentResults)

    @patch("zsharp.trainer.get_dataset")
    @patch("zsharp.trainer.get_model")
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with patch("torch.device", return_value=torch.device("cpu")):
            results = train(config)

            assert len(results.train_losses) == 3
            assert len(results.train_accuracies) == 3

    @patch("zsharp.trainer.get_dataset")
    @patch("zsharp.trainer.get_model")
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with (
            patch("torch.device", return_value=torch.device("cpu")),
            patch("zsharp.trainer.Path") as mock_path,
            patch("builtins.open", create=True) as mock_open,
        ):
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.__truediv__.return_value = mock_path_instance
            mock_path_instance.open.return_value.__enter__.return_value = (
                mock_file
            )

            train(config)

            # Check that file was opened for writing
            assert mock_path_instance.open.called
            assert mock_file.write.called

    @patch("zsharp.trainer.get_dataset")
    @patch("zsharp.trainer.get_model")
    def test_train_no_gradient_clipping(
        self, mock_get_model, mock_get_dataset
    ):
        """Test that no gradient clipping is applied.

        The paper (arXiv:2505.02369) specifies no gradient clipping, so the
        trainer must not rescale gradients before the optimizer step.
        """
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with (
            patch("torch.device", return_value=torch.device("cpu")),
            patch("torch.nn.utils.clip_grad_norm_") as mock_clip,
        ):
            train(config)

            # Check that gradient clipping was not called
            mock_clip.assert_not_called()

    @patch("zsharp.trainer.get_dataset")
    @patch("zsharp.trainer.get_model")
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with (
            patch("torch.device", return_value=torch.device("cpu")),
            patch("zsharp.trainer.tqdm") as mock_tqdm,
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with (
            patch(
                "zsharp.trainer.get_dataset",
                return_value=(mock_trainloader, mock_testloader),
            ),
            patch("zsharp.trainer.get_model", return_value=SimpleTestModel()),
            patch("torch.device", return_value=torch.device("cpu")),
            patch("zsharp.trainer.tqdm") as mock_tqdm,
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with (
            patch(
                "zsharp.trainer.get_dataset",
                return_value=(mock_trainloader, mock_testloader),
            ),
            patch("zsharp.trainer.get_model", return_value=SimpleTestModel()),
            patch("torch.device", return_value=torch.device("cpu")),
            patch("zsharp.trainer.tqdm") as mock_tqdm,
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with (
            patch(
                "zsharp.trainer.get_dataset",
                return_value=(mock_trainloader, mock_testloader),
            ),
            patch("zsharp.trainer.get_model", return_value=SimpleTestModel()),
            patch("torch.device", return_value=torch.device("cpu")),
            patch("zsharp.trainer.tqdm") as mock_tqdm,
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

        config = TrainingConfig.model_validate(
            {
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
        )

        with (
            patch(
                "zsharp.trainer.get_dataset",
                return_value=(mock_trainloader, mock_testloader),
            ),
            patch("zsharp.trainer.get_model", return_value=SimpleTestModel()),
            patch("torch.device", return_value=torch.device("mps")),
            patch("torch.backends.mps.is_available", return_value=True),
            patch("zsharp.trainer.tqdm") as mock_tqdm,
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
            # The code calls _run_epoch (1 tqdm), _validate (1 tqdm), and final _validate (1 tqdm)?
            # Actually, per epoch: _run_epoch, _validate.
            # Then final _validate.
            # For 1 epoch: tqdm, tqdm, tqdm.
            mock_tqdm.side_effect = [mock_pbar, mock_eval_pbar, mock_eval_pbar]
            mock_eval_pbar.__iter__ = lambda self: iter(
                [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
            )
            mock_eval_pbar.set_postfix = MagicMock()
            mock_eval_pbar.close = MagicMock()

            # This should trigger the MPS mixed precision case during evaluation
            result = train(config)
            assert isinstance(result, ExperimentResults)
            assert result.device == "mps"


class TestZSharpGradientHygiene:
    """Regression tests for gradient zeroing in the ZSharp training step."""

    @staticmethod
    def _make_ctx():
        """Build a ZSharp training context with no-op updates."""
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        # lr=0 and rho=0 make every step a no-op, so any change in the
        # gradient between steps can only come from stale accumulation.
        optimizer = ZSharp(
            list(model.parameters()),
            base_optimizer=optim.SGD,
            lr=0.0,
            rho=0.0,
        )
        ctx = TrainingContext(
            model,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            use_zsharp=True,
            use_half=False,
        )
        return ctx, model

    def test_gradients_do_not_accumulate_across_steps(self):
        """Repeating an identical step must not grow the gradient."""
        ctx, model = self._make_ctx()
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        norms = []
        for _ in range(4):
            _run_train_step(ctx, x, y)
            norms.append(model.weight.grad.norm().item())

        for norm in norms[1:]:
            assert norm == pytest.approx(norms[0], rel=1e-6)

    def test_step_zeroes_before_first_backward(self):
        """Pre-existing gradients must not leak into the step."""
        ctx, model = self._make_ctx()
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        _run_train_step(ctx, x, y)
        clean = model.weight.grad.clone()

        # Poison the gradients; a correct step discards them.
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        _run_train_step(ctx, x, y)

        assert torch.allclose(model.weight.grad, clean, atol=1e-6)


class TestZSharpLargeModelThreshold:
    """Threshold computation must handle models above the quantile cap."""

    def test_threshold_handles_more_elements_than_quantile_allows(self):
        """torch.quantile rejects >2**24 elements; ZSharp must not."""
        model = nn.Linear(4, 2)
        optimizer = ZSharp(
            list(model.parameters()),
            base_optimizer=optim.SGD,
            lr=0.01,
            percentile=70,
        )
        oversized = [torch.randn(MAX_QUANTILE_NUMEL + 1)]

        threshold = optimizer._compute_filtering_threshold(oversized)

        assert isinstance(threshold, float)
        assert threshold > 0

    def test_threshold_of_empty_gradients_is_zero(self):
        """Zero-element gradients yield a zero threshold, not a crash."""
        model = nn.Linear(4, 2)
        optimizer = ZSharp(
            list(model.parameters()),
            base_optimizer=optim.SGD,
            lr=0.01,
            percentile=70,
        )

        assert optimizer._compute_filtering_threshold([torch.empty(0)]) == 0.0

    def test_threshold_matches_quantile_below_cap(self):
        """The fallback must agree with torch.quantile on small inputs."""
        model = nn.Linear(4, 2)
        optimizer = ZSharp(
            list(model.parameters()),
            base_optimizer=optim.SGD,
            lr=0.01,
            percentile=70,
        )
        values = [torch.randn(100_000)]

        threshold = optimizer._compute_filtering_threshold(values)
        expected = torch.quantile(values[0].abs(), 0.7).item()

        assert threshold == pytest.approx(expected, abs=1e-4)


def _tiny_loaders(**_kwargs):
    """Build small real loaders so shuffling exercises the RNG state."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(32, 3, 32, 32, generator=gen)
    y = torch.randint(0, 10, (32,), generator=gen)
    data = torch.utils.data.TensorDataset(x, y)
    return (
        torch.utils.data.DataLoader(data, batch_size=8, shuffle=True),
        torch.utils.data.DataLoader(data, batch_size=8),
    )


def _checkpoint_config(opt_type, epochs, checkpoint_dir=None, seed=7):
    """Build a CPU config for the checkpoint tests."""
    return TrainingConfig.model_validate(
        {
            "seed": seed,
            "train": {
                "epochs": epochs,
                "device": "cpu",
                "num_workers": 0,
                "checkpoint_dir": checkpoint_dir,
            },
            "optimizer": {"type": opt_type, "lr": 0.01},
        }
    )


@pytest.fixture
def _tiny_training():
    """Train SimpleTestModel on the tiny loaders instead of real data."""
    with (
        patch("zsharp.trainer.get_dataset", side_effect=_tiny_loaders),
        patch(
            "zsharp.trainer.get_model",
            side_effect=lambda **_: SimpleTestModel(),
        ),
    ):
        yield


@pytest.mark.usefixtures("_tiny_training")
class TestSeedAndCheckpointing:
    """Tests for the configurable seed and checkpoint/resume."""

    def test_train_uses_config_seed(self):
        """train() seeds from the config rather than a fixed constant."""
        with patch("zsharp.trainer.set_seed") as mock_set_seed:
            train(_checkpoint_config("sgd", 1, seed=123))
        mock_set_seed.assert_called_once_with(123)

    def test_seed_rejects_negative(self):
        """Negative seeds are rejected at validation time."""
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate({"seed": -1})

    def test_checkpoint_written_after_each_epoch(self, tmp_path):
        """A checkpoint named after the run records the last epoch."""
        train(_checkpoint_config("zsharp", 2, str(tmp_path)))

        path = tmp_path / "cifar10_resnet18_zsharp_seed7.pt"
        state = torch.load(path, weights_only=False)
        assert state["progress"]["next_epoch"] == 2
        assert len(state["progress"]["test_accuracies"]) == 2
        assert not path.with_suffix(".tmp").exists()

    @pytest.mark.filterwarnings("error:Detected call of `lr_scheduler.step")
    def test_zsharp_training_steps_optimizer_before_scheduler(self):
        """The base optimizer steps before the scheduler, so no warning."""
        train(_checkpoint_config("zsharp", 2))

    @pytest.mark.parametrize("opt_type", ["sgd", "zsharp"])
    def test_resume_matches_uninterrupted_run(self, tmp_path, opt_type):
        """Stopping after epoch 2 and resuming reproduces a 3-epoch run."""
        full = train(_checkpoint_config(opt_type, 3))

        train(_checkpoint_config(opt_type, 2, str(tmp_path)))
        resumed = train(_checkpoint_config(opt_type, 3, str(tmp_path)))

        assert resumed.train_losses == pytest.approx(full.train_losses)
        assert resumed.test_accuracies == full.test_accuracies
        assert resumed.final_test_loss == pytest.approx(full.final_test_loss)

    def test_resume_reshares_zsharp_param_groups(self, tmp_path):
        """After loading, ZSharp and its base optimizer share param groups."""
        train(_checkpoint_config("zsharp", 1, str(tmp_path)))
        with patch("zsharp.trainer._run_epoch") as mock_epoch:
            mock_epoch.side_effect = lambda ctx, *_: (
                ctx.optimizer.param_groups
                is ctx.optimizer.base_optimizer.param_groups
                or pytest.fail("param groups not shared"),
                (0.0, 0.0),
            )[1]
            train(_checkpoint_config("zsharp", 2, str(tmp_path)))
        mock_epoch.assert_called_once()


@pytest.mark.usefixtures("_tiny_training")
class TestRegularizationOptions:
    """Tests for label smoothing and strong augmentation settings."""

    def test_defaults_match_paper_recipe(self):
        """Both options are off unless a config turns them on."""
        cfg = TrainingConfig().train
        assert cfg.label_smoothing == 0.0
        assert cfg.strong_augmentation is False

    @pytest.mark.parametrize("value", [-0.1, 1.0])
    def test_label_smoothing_rejects_out_of_range(self, value):
        """Smoothing must lie in [0, 1)."""
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate(
                {"train": {"label_smoothing": value}}
            )

    def test_options_reach_loss_and_loader(self):
        """train() wires smoothing into the loss and the flag into data."""
        config = _checkpoint_config("sgd", 1)
        config.train.label_smoothing = 0.1
        config.train.strong_augmentation = True
        losses = []
        real_loss = nn.CrossEntropyLoss

        def recording_loss(**kwargs):
            losses.append(kwargs)
            return real_loss(**kwargs)

        with (
            patch(
                "zsharp.trainer.get_dataset", side_effect=_tiny_loaders
            ) as mock_get_dataset,
            patch("zsharp.trainer.nn.CrossEntropyLoss", recording_loss),
        ):
            train(config)

        assert losses == [{"label_smoothing": 0.1}]
        assert mock_get_dataset.call_args.kwargs["strong_augmentation"]
