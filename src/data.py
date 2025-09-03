"""Data loading utilities for CIFAR-10 and CIFAR-100 datasets.

This module provides functions to load and preprocess CIFAR-10 and CIFAR-100
datasets with appropriate data augmentation and normalization.
"""

import torch
import torchvision
import torchvision.transforms as T

from src.constants import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    CIFAR_CROP_PADDING,
    CIFAR_IMAGE_SIZE,
    DATA_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
)


def get_cifar10(
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
    pin_memory=DEFAULT_PIN_MEMORY,
):
    """Get CIFAR-10 dataset with train and test data loaders.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, test_loader) for CIFAR-10 dataset
    """
    transform_train = T.Compose(
        [
            T.RandomCrop(CIFAR_IMAGE_SIZE, padding=CIFAR_CROP_PADDING),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # CIFAR-10 normalization
        ]
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # CIFAR-10 normalization
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


def get_cifar100(
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
    pin_memory=DEFAULT_PIN_MEMORY,
):
    """Get CIFAR-100 dataset with train and test data loaders.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, test_loader) for CIFAR-100 dataset
    """
    transform_train = T.Compose(
        [
            T.RandomCrop(CIFAR_IMAGE_SIZE, padding=CIFAR_CROP_PADDING),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                CIFAR100_MEAN, CIFAR100_STD
            ),  # CIFAR-100 normalization
        ]
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                CIFAR100_MEAN, CIFAR100_STD
            ),  # CIFAR-100 normalization
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    testset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


def get_dataset(
    dataset_name,
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
    pin_memory=DEFAULT_PIN_MEMORY,
):
    """Get dataset by name with train and test data loaders.

    Args:
        dataset_name: Name of the dataset ('cifar10' or 'cifar100')
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, test_loader) for the specified dataset

    Raises:
        ValueError: If dataset name is not supported
    """
    if dataset_name == "cifar10":
        return get_cifar10(batch_size, num_workers, pin_memory)
    elif dataset_name == "cifar100":
        return get_cifar100(batch_size, num_workers, pin_memory)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
