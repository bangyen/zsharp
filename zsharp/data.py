# Copyright (c) 2025 Bangyen Pham
"""Data loading utilities for the datasets used in the ZSharp paper.

This module provides functions to load and preprocess CIFAR-10, CIFAR-100
and Tiny-ImageNet with appropriate data augmentation and normalization.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Optional, Union, cast

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive

from zsharp.constants import (
    DATA_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
    TINY_IMAGENET_DATASET,
    TINY_IMAGENET_DIRNAME,
    TINY_IMAGENET_URL,
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
    # Tiny-ImageNet: 200 classes of 64x64 images. The paper does not state
    # normalization statistics, so the commonly cited Tiny-ImageNet values
    # are applied here; augmentation mirrors the CIFAR recipe above.
    TINY_IMAGENET_DATASET: {
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2770, 0.2691, 0.2821),
        "num_classes": 200,
        "image_size": 64,
        "crop_padding": 8,
    },
}

_DATASET_CLASSES: dict[str, type[torchvision.datasets.VisionDataset]] = {
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
}


def _get_cifar(
    dataset_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    *,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
) -> tuple[
    torch.utils.data.DataLoader[torch.Tensor],
    torch.utils.data.DataLoader[torch.Tensor],
]:
    """Load a CIFAR dataset by name with train and test data loaders.

    Args:
        dataset_name: Name of the CIFAR dataset ('cifar10' or 'cifar100')
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, test_loader) for the specified CIFAR dataset

    """
    meta = DATASET_METADATA[dataset_name]
    dataset_cls = _DATASET_CLASSES[dataset_name]

    transform_train = T.Compose(
        [
            T.RandomCrop(meta["image_size"], padding=meta["crop_padding"]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(meta["mean"], meta["std"]),
        ],
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(meta["mean"], meta["std"]),
        ],
    )

    trainset = dataset_cls(
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

    testset = dataset_cls(
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


class TinyImageNet(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    """Tiny-ImageNet-200: 200 classes of 64x64 images.

    Not available through torchvision, so this downloads and extracts the
    canonical archive on first use. The train split is laid out per class,
    but the validation split ships as a flat directory whose labels live in
    ``val_annotations.txt``, so both are resolved to a common index here.
    """

    def __init__(
        self,
        root: str = DATA_ROOT,
        *,
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Build the sample index for one split.

        Args:
            root: Directory holding (or to receive) the extracted dataset.
            train: Load the training split rather than the validation split.
            download: Fetch the archive if it is not already present.
            transform: Optional transform applied to each PIL image.

        Raises:
            RuntimeError: If the dataset is missing and ``download`` is
                False.
        """
        self.transform = transform
        self.root = Path(root) / TINY_IMAGENET_DIRNAME

        if not self.root.exists():
            if not download:
                msg = (
                    f"Tiny-ImageNet not found at {self.root}. "
                    "Pass download=True to fetch it."
                )
                raise RuntimeError(msg)
            download_and_extract_archive(TINY_IMAGENET_URL, download_root=root)

        wnids = sorted((self.root / "wnids.txt").read_text().split())
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
        self.samples = self._train_samples() if train else self._val_samples()

    def _train_samples(self) -> list[tuple[Path, int]]:
        """Index the per-class training directories."""
        samples: list[tuple[Path, int]] = []
        for wnid, label in self.class_to_idx.items():
            image_dir = self.root / "train" / wnid / "images"
            samples.extend(
                (path, label) for path in sorted(image_dir.glob("*.JPEG"))
            )
        return samples

    def _val_samples(self) -> list[tuple[Path, int]]:
        """Index the flat validation directory via its annotations file."""
        annotations = self.root / "val" / "val_annotations.txt"
        samples = []
        for line in annotations.read_text().splitlines():
            if not line.strip():
                continue
            filename, wnid = line.split("\t")[:2]
            path = self.root / "val" / "images" / filename
            samples.append((path, self.class_to_idx[wnid]))
        return samples

    def __len__(self) -> int:
        """Return the number of samples in the split.

        Returns:
            int: Sample count.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load one sample.

        Args:
            index: Position of the sample within the split.

        Returns:
            tuple: The transformed image and its class index.
        """
        path, label = self.samples[index]
        with Image.open(path) as image:
            # Tiny-ImageNet contains a few grayscale images.
            converted = image.convert("RGB")
        if self.transform is not None:
            return self.transform(converted), label
        return T.functional.to_tensor(converted), label


def _get_tiny_imagenet(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    *,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
) -> tuple[
    torch.utils.data.DataLoader[torch.Tensor],
    torch.utils.data.DataLoader[torch.Tensor],
]:
    """Load Tiny-ImageNet with train and test data loaders.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, test_loader) for Tiny-ImageNet

    """
    meta = DATASET_METADATA[TINY_IMAGENET_DATASET]
    transform_train = T.Compose(
        [
            T.RandomCrop(meta["image_size"], padding=meta["crop_padding"]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(meta["mean"], meta["std"]),
        ],
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(meta["mean"], meta["std"]),
        ],
    )

    # Pass DATA_ROOT explicitly: the constructor's default is bound at
    # import time, so it would ignore a patched or reconfigured root.
    trainset = TinyImageNet(
        root=DATA_ROOT, train=True, transform=transform_train
    )
    testset = TinyImageNet(
        root=DATA_ROOT, train=False, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return (
        cast("torch.utils.data.DataLoader[torch.Tensor]", trainloader),
        cast("torch.utils.data.DataLoader[torch.Tensor]", testloader),
    )


def get_cifar10(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    *,
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
    return _get_cifar(
        "cifar10",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_cifar100(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    *,
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
    return _get_cifar(
        "cifar100",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_dataset(
    dataset_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    *,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
) -> tuple[
    torch.utils.data.DataLoader[torch.Tensor],
    torch.utils.data.DataLoader[torch.Tensor],
]:
    """Get dataset by name with train and test data loaders.

    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100' or
            'tiny_imagenet')
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        tuple: (train_loader, test_loader) for the specified dataset

    Raises:
        ValueError: If dataset name is not supported

    """
    if dataset_name in _DATASET_CLASSES:
        return _get_cifar(
            dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    if dataset_name == TINY_IMAGENET_DATASET:
        return _get_tiny_imagenet(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    error_msg = f"Unknown dataset: {dataset_name}"
    raise ValueError(error_msg)
