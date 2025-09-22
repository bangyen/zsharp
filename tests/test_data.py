"""Test suite for data loading and processing functions."""

import pytest
import torch

from src.data import get_dataset


class TestDataModule:
    """Test cases for data loading functions"""

    def test_get_dataset_cifar10(self):
        """Test get_dataset function with cifar10"""
        trainloader, testloader = get_dataset(
            "cifar10", batch_size=32, num_workers=0
        )

        assert isinstance(trainloader, torch.utils.data.DataLoader)
        assert isinstance(testloader, torch.utils.data.DataLoader)

        # Check data shape
        for data, _target in trainloader:
            assert data.shape[1:] == (3, 32, 32)
            break

    def test_get_dataset_cifar100(self):
        """Test get_dataset function with cifar100"""
        trainloader, testloader = get_dataset(
            "cifar100", batch_size=32, num_workers=0
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
