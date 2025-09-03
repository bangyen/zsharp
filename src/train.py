"""Training utilities for deep learning models.

This module provides comprehensive training functionality including
device management, data loading, model training, and result saving.
"""

import json
import logging
import os
import time
from typing import Any, Optional, Union

import torch
import torch.nn
import torch.optim
from torch import nn, optim
from tqdm import tqdm

from src.constants import (
    BATCH_SIZE_KEY,
    CIFAR10_DATASET,
    CIFAR10_NUM_CLASSES,
    CIFAR100_DATASET,
    CIFAR100_NUM_CLASSES,
    CPU_DEVICE,
    CUDA_DEVICE,
    DATASET_CONFIG_KEY,
    DEFAULT_SEED,
    DEVICE_KEY,
    EPOCHS_KEY,
    LR_KEY,
    MAX_GRADIENT_NORM,
    MODEL_CONFIG_KEY,
    MOMENTUM_KEY,
    MPS_DEVICE,
    NUM_WORKERS_KEY,
    OPTIMIZER_CONFIG_KEY,
    PERCENTAGE_MULTIPLIER,
    PERCENTILE_KEY,
    PIN_MEMORY_KEY,
    RESULTS_DIR,
    RHO_KEY,
    SGD_OPTIMIZER,
    TRAIN_CONFIG_KEY,
    TYPE_KEY,
    USE_MIXED_PRECISION_KEY,
    WEIGHT_DECAY_KEY,
    ZSHARP_OPTIMIZER,
)
from src.data import get_dataset
from src.models import get_model
from src.optimizer import ZSharp
from src.utils import set_seed

logger = logging.getLogger(__name__)


def get_device(config: dict[str, Any]) -> torch.device:
    """Get the best available device for training.

    Args:
        config: Configuration dictionary containing device settings

    Returns:
        torch.device: Best available device (mps/cuda/cpu)
    """
    device_config = config[TRAIN_CONFIG_KEY][DEVICE_KEY]

    if device_config == MPS_DEVICE and torch.backends.mps.is_available():
        return torch.device(MPS_DEVICE)
    if device_config == CUDA_DEVICE and torch.cuda.is_available():
        return torch.device(CUDA_DEVICE)
    return torch.device(CPU_DEVICE)


def train(config: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Train a model using the provided configuration.

    Args:
        config: Configuration dictionary containing all training parameters

    Returns:
        dict: Training results including losses, accuracies, and timing
    """
    # Set seed for reproducibility
    set_seed(DEFAULT_SEED)

    device = get_device(config)

    # Get training parameters
    batch_size = int(config[TRAIN_CONFIG_KEY][BATCH_SIZE_KEY])
    num_workers = int(config[TRAIN_CONFIG_KEY].get(NUM_WORKERS_KEY, 2))
    pin_memory = config[TRAIN_CONFIG_KEY].get(PIN_MEMORY_KEY, False)
    use_mixed_precision = config[TRAIN_CONFIG_KEY].get(
        USE_MIXED_PRECISION_KEY, False
    )

    # Get dataset
    dataset_name = config.get(DATASET_CONFIG_KEY, CIFAR10_DATASET)
    trainloader, testloader = get_dataset(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Get model
    model_name = config.get(MODEL_CONFIG_KEY, "resnet18")
    num_classes = (
        CIFAR100_NUM_CLASSES
        if dataset_name == CIFAR100_DATASET
        else CIFAR10_NUM_CLASSES
    )
    model = get_model(model_name, num_classes=num_classes).to(device)

    # Enable mixed precision for faster training on Apple Silicon (optional)
    if device.type == "mps" and use_mixed_precision:
        model = model.half()  # Use float16 for better performance

    # Setup optimizer based on config
    optimizer_type = config[OPTIMIZER_CONFIG_KEY].get(
        TYPE_KEY, ZSHARP_OPTIMIZER
    )

    optimizer: Union[torch.optim.Optimizer, ZSharp]

    if optimizer_type == SGD_OPTIMIZER:
        optimizer = optim.SGD(
            list(model.parameters()),
            lr=float(config[OPTIMIZER_CONFIG_KEY][LR_KEY]),
            momentum=float(config[OPTIMIZER_CONFIG_KEY][MOMENTUM_KEY]),
            weight_decay=float(config[OPTIMIZER_CONFIG_KEY][WEIGHT_DECAY_KEY]),
        )
        use_zsharp = False
    else:
        # ZSharp optimizer
        base_opt = optim.SGD
        optimizer = ZSharp(
            list(model.parameters()),
            base_optimizer=base_opt,
            rho=float(config[OPTIMIZER_CONFIG_KEY][RHO_KEY]),
            lr=float(config[OPTIMIZER_CONFIG_KEY][LR_KEY]),
            momentum=float(config[OPTIMIZER_CONFIG_KEY][MOMENTUM_KEY]),
            weight_decay=float(config[OPTIMIZER_CONFIG_KEY][WEIGHT_DECAY_KEY]),
            percentile=int(config[OPTIMIZER_CONFIG_KEY][PERCENTILE_KEY]),
        )
        use_zsharp = True

    criterion = nn.CrossEntropyLoss()

    # Training loop with timing
    start_time = time.time()
    train_losses = []
    train_accuracies = []

    try:
        for epoch in range(int(config[TRAIN_CONFIG_KEY][EPOCHS_KEY])):
            model.train()
            epoch_loss = 0.0
            correct, total = 0, 0

            desc = f"Epoch {epoch + 1}/{config[TRAIN_CONFIG_KEY][EPOCHS_KEY]}"
            pbar: tqdm = tqdm(trainloader, desc=desc)

            try:
                for _i, (x, y) in enumerate(pbar):
                    x, y = x.to(device), y.to(device)

                    # Convert to half precision if using MPS and mixed precision is
                    # enabled
                    if device.type == "mps" and use_mixed_precision:
                        x = x.half()

                    if use_zsharp:
                        # ZSharp two-step training
                        loss = criterion(model(x), y)
                        loss.backward()

                        # Add gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=MAX_GRADIENT_NORM
                        )

                        # Type check to ensure optimizer is ZSharp
                        if isinstance(optimizer, ZSharp):
                            optimizer.first_step()

                            criterion(model(x), y).backward()
                            optimizer.second_step()
                    else:
                        # Standard SGD training
                        optimizer.zero_grad()
                        loss = criterion(model(x), y)
                        loss.backward()

                        # Add gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=MAX_GRADIENT_NORM
                        )

                        optimizer.step()

                    # Track metrics
                    epoch_loss += loss.item()
                    preds = model(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "Loss": f"{loss.item():.4f}",
                            "Acc": (
                                f"{PERCENTAGE_MULTIPLIER * correct / total:.2f}%"
                            ),
                        }
                    )

            except KeyboardInterrupt:
                pbar.close()
                logger.warning("Training interrupted by user.")
                raise

            # Record epoch metrics
            avg_loss = epoch_loss / len(trainloader)
            train_accuracy = PERCENTAGE_MULTIPLIER * correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(train_accuracy)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return None

    total_time = time.time() - start_time

    # Evaluate on test set
    model.eval()
    test_correct, test_total = 0, 0
    test_loss = 0.0

    try:
        with torch.no_grad():
            eval_pbar: tqdm = tqdm(testloader, desc="Evaluating")
            try:
                for x, y in eval_pbar:
                    x, y = x.to(device), y.to(device)
                    if device.type == "mps" and use_mixed_precision:
                        x = x.half()

                    outputs = model(x)
                    loss = criterion(outputs, y)
                    test_loss += loss.item()

                    preds = outputs.argmax(dim=1)
                    test_correct += (preds == y).sum().item()
                    test_total += y.size(0)

                    acc = PERCENTAGE_MULTIPLIER * test_correct / test_total
                    eval_pbar.set_postfix({"Test Acc": f"{acc:.2f}%"})
            except KeyboardInterrupt:
                eval_pbar.close()
                logger.warning("Evaluation interrupted by user.")
                raise
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user.")
        return None

    test_accuracy = PERCENTAGE_MULTIPLIER * test_correct / test_total
    avg_test_loss = test_loss / len(testloader)

    # Save results
    results = {
        "config": config,
        "final_test_accuracy": test_accuracy,
        "final_test_loss": avg_test_loss,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "total_training_time": total_time,
        "device": str(device),
        "optimizer_type": optimizer_type,
    }

    # Save results to file
    results_file = (
        f"{RESULTS_DIR}/zsharp_{dataset_name}"
        f"_{model_name}_{optimizer_type}.json"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    return results
