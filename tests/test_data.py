# Copyright (c) 2025 Bangyen Pham
"""Test suite for data loading and processing functions."""

from unittest.mock import patch

import pytest
import torch

from zsharp.data import get_cifar10, get_cifar100, get_dataset

# Patch the dataset classes so tests exercise the loader wiring without
# downloading the real CIFAR datasets (which are ~163MB each and would
# otherwise be fetched on every CI run).
FAKE_DATASET_CLASSES = {
    "cifar10": type(
        "FakeCIFAR10",
        (),
        {
            "__init__": lambda self, **kwargs: None,
            "__len__": lambda self: 8,
            "__getitem__": lambda self, idx: (
                torch.randn(3, 32, 32),
                torch.tensor(idx % 10),
            ),
        },
    ),
    "cifar100": type(
        "FakeCIFAR100",
        (),
        {
            "__init__": lambda self, **kwargs: None,
            "__len__": lambda self: 8,
            "__getitem__": lambda self, idx: (
                torch.randn(3, 32, 32),
                torch.tensor(idx % 100),
            ),
        },
    ),
}


class TestDataModule:
    """Test cases for data loading functions"""

    @patch("zsharp.data._DATASET_CLASSES", FAKE_DATASET_CLASSES)
    def test_get_dataset_cifar10(self):
        """Test get_dataset function with cifar10"""
        trainloader, testloader = get_dataset(
            "cifar10", batch_size=4, num_workers=0
        )

        assert isinstance(trainloader, torch.utils.data.DataLoader)
        assert isinstance(testloader, torch.utils.data.DataLoader)

        # Check data shape
        for data, _target in trainloader:
            assert data.shape[1:] == (3, 32, 32)
            break

    @patch("zsharp.data._DATASET_CLASSES", FAKE_DATASET_CLASSES)
    def test_get_dataset_cifar100(self):
        """Test get_dataset function with cifar100"""
        trainloader, testloader = get_dataset(
            "cifar100", batch_size=4, num_workers=0
        )

        assert isinstance(trainloader, torch.utils.data.DataLoader)
        assert isinstance(testloader, torch.utils.data.DataLoader)

        # Check data shape
        for data, _target in trainloader:
            assert data.shape[1:] == (3, 32, 32)
            break

    def test_get_dataset_unknown_dataset(self):
        """Test get_dataset function with unknown dataset raises error"""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset("unknown_dataset", batch_size=32, num_workers=0)

    @patch("zsharp.data._get_cifar")
    def test_get_cifar10_delegates(self, mock_get_cifar):
        """Test get_cifar10 delegates to the generic loader."""
        get_cifar10(batch_size=64, num_workers=1)
        mock_get_cifar.assert_called_once_with(
            "cifar10",
            batch_size=64,
            num_workers=1,
            pin_memory=False,
        )

    @patch("zsharp.data._get_cifar")
    def test_get_cifar100_delegates(self, mock_get_cifar):
        """Test get_cifar100 delegates to the generic loader."""
        get_cifar100(batch_size=64, num_workers=1)
        mock_get_cifar.assert_called_once_with(
            "cifar100",
            batch_size=64,
            num_workers=1,
            pin_memory=False,
        )


def _recording_dataset_classes(calls):
    """Build fake dataset classes that record their constructor kwargs."""

    def make(name, num_classes):
        def __init__(self, **kwargs):  # noqa: N807
            calls.append({"dataset": name, **kwargs})

        return type(
            f"Recording{name}",
            (),
            {
                "__init__": __init__,
                "__len__": lambda self: 8,
                "__getitem__": lambda self, idx: (
                    torch.randn(3, 32, 32),
                    torch.tensor(idx % num_classes),
                ),
            },
        )

    return {"cifar10": make("cifar10", 10), "cifar100": make("cifar100", 100)}


class TestDatasetWiring:
    """Assert the values get_dataset threads through to the loaders."""

    def test_known_dataset_is_dispatched_not_rejected(self):
        """A known name must load; only unknown names raise."""
        calls = []
        with patch(
            "zsharp.data._DATASET_CLASSES", _recording_dataset_classes(calls)
        ):
            get_dataset("cifar10", batch_size=4, num_workers=0)

        assert [c["dataset"] for c in calls] == ["cifar10", "cifar10"]

    def test_train_and_test_splits_are_requested(self):
        """One loader must be train=True and the other train=False."""
        calls = []
        with patch(
            "zsharp.data._DATASET_CLASSES", _recording_dataset_classes(calls)
        ):
            get_dataset("cifar100", batch_size=4, num_workers=0)

        assert [c["train"] for c in calls] == [True, False]
        assert all(c["download"] is True for c in calls)
        assert all(c["transform"] is not None for c in calls)

    def test_loader_params_reach_the_dataloaders(self):
        """batch_size / num_workers / pin_memory must not be dropped."""
        with patch(
            "zsharp.data._DATASET_CLASSES", _recording_dataset_classes([])
        ):
            train, test = get_dataset(
                "cifar10", batch_size=2, num_workers=0, pin_memory=False
            )

        for loader in (train, test):
            assert loader.batch_size == 2
            assert loader.num_workers == 0
            assert loader.pin_memory is False

    def test_only_train_loader_shuffles(self):
        """Train shuffles; test must not, or eval order becomes random."""
        with patch(
            "zsharp.data._DATASET_CLASSES", _recording_dataset_classes([])
        ):
            train, test = get_dataset("cifar10", batch_size=2, num_workers=0)

        assert isinstance(train.sampler, torch.utils.data.RandomSampler), (
            "train loader must shuffle"
        )
        assert isinstance(test.sampler, torch.utils.data.SequentialSampler), (
            "test loader must not shuffle"
        )
