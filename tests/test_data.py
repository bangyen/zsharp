# Copyright (c) 2025 Bangyen Pham
"""Test suite for data loading and processing functions."""

from unittest.mock import patch

import pytest
import torch

from src.data import get_cifar10, get_cifar100, get_dataset

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

    @patch("src.data._DATASET_CLASSES", FAKE_DATASET_CLASSES)
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

    @patch("src.data._DATASET_CLASSES", FAKE_DATASET_CLASSES)
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

    @patch("src.data._get_cifar")
    def test_get_cifar10_delegates(self, mock_get_cifar):
        """Test get_cifar10 delegates to the generic loader."""
        get_cifar10(batch_size=64, num_workers=1)
        mock_get_cifar.assert_called_once_with(
            "cifar10",
            batch_size=64,
            num_workers=1,
            pin_memory=False,
        )

    @patch("src.data._get_cifar")
    def test_get_cifar100_delegates(self, mock_get_cifar):
        """Test get_cifar100 delegates to the generic loader."""
        get_cifar100(batch_size=64, num_workers=1)
        mock_get_cifar.assert_called_once_with(
            "cifar100",
            batch_size=64,
            num_workers=1,
            pin_memory=False,
        )
