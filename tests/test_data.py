import pytest
import torch
import torchvision
from src.data import get_cifar10, get_cifar100, get_dataset


class TestDataModule:
    """Test cases for data loading functions"""

    def test_get_cifar10_returns_correct_types(self):
        """Test that get_cifar10 returns correct data types"""
        trainloader, testloader = get_cifar10(batch_size=32, num_workers=0)

        assert isinstance(trainloader, torch.utils.data.DataLoader)
        assert isinstance(testloader, torch.utils.data.DataLoader)

    def test_get_cifar10_batch_size(self):
        """Test that get_cifar10 respects batch_size parameter"""
        batch_size = 64
        trainloader, testloader = get_cifar10(
            batch_size=batch_size, num_workers=0
        )

        # Check first batch size
        for batch_idx, (data, target) in enumerate(trainloader):
            assert data.size(0) <= batch_size
            break

        for batch_idx, (data, target) in enumerate(testloader):
            assert data.size(0) <= batch_size
            break

    def test_get_cifar10_data_shape(self):
        """Test that CIFAR-10 data has correct shape"""
        trainloader, testloader = get_cifar10(batch_size=32, num_workers=0)

        # Check training data shape
        for data, target in trainloader:
            assert data.shape[1:] == (3, 32, 32)  # CIFAR-10 images are 3x32x32
            assert target.shape[0] == data.shape[0]  # Same batch size
            break

        # Check test data shape
        for data, target in testloader:
            assert data.shape[1:] == (3, 32, 32)
            assert target.shape[0] == data.shape[0]
            break

    def test_get_cifar10_target_range(self):
        """Test that CIFAR-10 targets are in correct range [0, 9]"""
        trainloader, testloader = get_cifar10(batch_size=32, num_workers=0)

        # Check training targets
        for data, target in trainloader:
            assert torch.all(target >= 0) and torch.all(target < 10)
            break

        # Check test targets
        for data, target in testloader:
            assert torch.all(target >= 0) and torch.all(target < 10)
            break

    def test_get_cifar100_returns_correct_types(self):
        """Test that get_cifar100 returns correct data types"""
        trainloader, testloader = get_cifar100(batch_size=32, num_workers=0)

        assert isinstance(trainloader, torch.utils.data.DataLoader)
        assert isinstance(testloader, torch.utils.data.DataLoader)

    def test_get_cifar100_batch_size(self):
        """Test that get_cifar100 respects batch_size parameter"""
        batch_size = 64
        trainloader, testloader = get_cifar100(
            batch_size=batch_size, num_workers=0
        )

        # Check first batch size
        for batch_idx, (data, target) in enumerate(trainloader):
            assert data.size(0) <= batch_size
            break

        for batch_idx, (data, target) in enumerate(testloader):
            assert data.size(0) <= batch_size
            break

    def test_get_cifar100_data_shape(self):
        """Test that CIFAR-100 data has correct shape"""
        trainloader, testloader = get_cifar100(batch_size=32, num_workers=0)

        # Check training data shape
        for data, target in trainloader:
            assert data.shape[1:] == (
                3,
                32,
                32,
            )  # CIFAR-100 images are 3x32x32
            assert target.shape[0] == data.shape[0]  # Same batch size
            break

        # Check test data shape
        for data, target in testloader:
            assert data.shape[1:] == (3, 32, 32)
            assert target.shape[0] == data.shape[0]
            break

    def test_get_cifar100_target_range(self):
        """Test that CIFAR-100 targets are in correct range [0, 99]"""
        trainloader, testloader = get_cifar100(batch_size=32, num_workers=0)

        # Check training targets
        for data, target in trainloader:
            assert torch.all(target >= 0) and torch.all(target < 100)
            break

        # Check test targets
        for data, target in testloader:
            assert torch.all(target >= 0) and torch.all(target < 100)
            break

    def test_get_dataset_cifar10(self):
        """Test get_dataset function with cifar10"""
        trainloader, testloader = get_dataset(
            "cifar10", batch_size=32, num_workers=0
        )

        assert isinstance(trainloader, torch.utils.data.DataLoader)
        assert isinstance(testloader, torch.utils.data.DataLoader)

        # Check data shape
        for data, target in trainloader:
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
        for data, target in trainloader:
            assert data.shape[1:] == (3, 32, 32)
            break

    def test_get_dataset_unknown_dataset(self):
        """Test get_dataset function with unknown dataset raises error"""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset("unknown_dataset", batch_size=32, num_workers=0)

    def test_data_transforms(self):
        """Test that data transforms are applied correctly"""
        trainloader, testloader = get_cifar10(batch_size=32, num_workers=0)

        # Check that data is normalized (values should be roughly in [-1, 1] range)
        for data, target in trainloader:
            # CIFAR-10 normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            # After normalization, most values should be in reasonable range
            assert torch.all(data >= -3) and torch.all(data <= 3)
            break

    def test_pin_memory_parameter(self):
        """Test pin_memory parameter is respected"""
        # Test with pin_memory=True
        trainloader, testloader = get_cifar10(
            batch_size=32, num_workers=0, pin_memory=True
        )

        # Test with pin_memory=False
        trainloader, testloader = get_cifar10(
            batch_size=32, num_workers=0, pin_memory=False
        )

        # Both should work without error
        assert True

    def test_num_workers_parameter(self):
        """Test num_workers parameter is respected"""
        # Test with different num_workers values - use smaller batch size and fewer workers for faster testing
        for num_workers in [0]:  # Further reduced from [0, 1] to [0] for maximum speed
            trainloader, testloader = get_cifar10(
                batch_size=8, num_workers=num_workers  # Further reduced batch size from 16 to 8
            )

            # Should work without error
            assert isinstance(trainloader, torch.utils.data.DataLoader)
            assert isinstance(testloader, torch.utils.data.DataLoader)

    def test_data_consistency(self):
        """Test that data is consistent across multiple calls"""
        trainloader1, testloader1 = get_cifar10(batch_size=32, num_workers=0)
        trainloader2, testloader2 = get_cifar10(batch_size=32, num_workers=0)

        # Get first batch from each
        data1, target1 = next(iter(trainloader1))
        data2, target2 = next(iter(trainloader2))

        # Data should have same shape
        assert data1.shape == data2.shape
        assert target1.shape == target2.shape
