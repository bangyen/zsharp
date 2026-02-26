"""Data loading utilities for CIFAR-10 and CIFAR-100 datasets.

This module provides functions to load and preprocess CIFAR-10 and CIFAR-100
datasets with appropriate data augmentation and normalization.
"""

from typing import Union

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T

from src.constants import (
    DATA_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
)

# Dataset metadata registry
DatasetValue = Union[tuple[float, float, float], int, str]
DATASET_METADATA: dict[str, dict[str, DatasetValue]] = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "num_classes": 10,
        "image_size": 32,
        "crop_padding": 4,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
        "image_size": 32,
        "crop_padding": 4,
    },
}


def get_cifar10(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    *,  # Force keyword-only arguments
    pin_memory: bool = DEFAULT_PIN_MEMORY,
) -> tuple[
    torch.utils.data.DataLoader[torch.Tensor],
    torch.utils.data.DataLoader[torch.Tensor],
]:
    """Get CIFAR-10 dataset with train and test data loaders.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, test_loader) for CIFAR-10 dataset

    """
    meta = DATASET_METADATA["cifar10"]
    transform_train = T.Compose(
        [
            T.RandomCrop(meta["image_size"], padding=meta["crop_padding"]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(meta["mean"], meta["std"]),  # CIFAR-10 normalization
        ],
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(meta["mean"], meta["std"]),  # CIFAR-10 normalization
        ],
    )

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform_train,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform_test,
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
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    *,  # Force keyword-only arguments
    pin_memory: bool = DEFAULT_PIN_MEMORY,
) -> tuple[
    torch.utils.data.DataLoader[torch.Tensor],
    torch.utils.data.DataLoader[torch.Tensor],
]:
    """Get CIFAR-100 dataset with train and test data loaders.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, test_loader) for CIFAR-100 dataset

    """
    meta = DATASET_METADATA["cifar100"]
    transform_train = T.Compose(
        [
            T.RandomCrop(meta["image_size"], padding=meta["crop_padding"]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                meta["mean"],
                meta["std"],
            ),  # CIFAR-100 normalization
        ],
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                meta["mean"],
                meta["std"],
            ),  # CIFAR-100 normalization
        ],
    )

    trainset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform_train,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    testset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform_test,
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
    dataset_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    *,  # Force keyword-only arguments
    pin_memory: bool = DEFAULT_PIN_MEMORY,
) -> tuple[
    torch.utils.data.DataLoader[torch.Tensor],
    torch.utils.data.DataLoader[torch.Tensor],
]:
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
        return get_cifar10(batch_size, num_workers, pin_memory=pin_memory)
    if dataset_name == "cifar100":
        return get_cifar100(batch_size, num_workers, pin_memory=pin_memory)
    error_msg = f"Unknown dataset: {dataset_name}"
    raise ValueError(error_msg)
