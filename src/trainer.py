# Copyright (c) 2025 Bangyen Pham
"""Training utilities for deep learning models.

This module provides comprehensive training functionality including
device management, data loading, model training, and result saving.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

import torch
import torch.optim
from torch import nn, optim
from tqdm import tqdm

from src.constants import (
    AUTO_DEVICE,
    CPU_DEVICE,
    CUDA_DEVICE,
    DEFAULT_SEED,
    MAX_GRADIENT_NORM,
    MPS_DEVICE,
    RESULTS_DIR,
    SGD_OPTIMIZER,
    ZSHARP_OPTIMIZER,
    ExperimentResults,
    TrainingConfig,
)
from src.data import DATASET_METADATA, get_dataset
from src.models import get_model
from src.optimizer import ZSharp

logger = logging.getLogger(__name__)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value for all random number generators

    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _detect_best_device() -> torch.device:
    """Detect the best available hardware device.

    Returns:
        torch.device: CUDA if available, then MPS, then CPU.
    """
    if torch.cuda.is_available():
        return torch.device(CUDA_DEVICE)
    if torch.backends.mps.is_available():
        return torch.device(MPS_DEVICE)
    return torch.device(CPU_DEVICE)


def get_device(config: TrainingConfig) -> torch.device:
    """Get the best available device for training.

    Args:
        config: Configuration dictionary containing device settings

    Returns:
        torch.device: Best available device (mps/cuda/cpu)

    """
    dev = config.train.device

    if dev == AUTO_DEVICE:
        return _detect_best_device()

    # Determine availability
    is_cuda = bool(dev == CUDA_DEVICE and torch.cuda.is_available())
    is_mps = bool(dev == MPS_DEVICE and torch.backends.mps.is_available())

    # Map to final device
    res = CPU_DEVICE
    if is_cuda:
        res = CUDA_DEVICE
    elif is_mps:
        res = MPS_DEVICE

    return torch.device(res)


@dataclass(frozen=True)
class TrainingContext:
    """Encapsulates training components and flags."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device
    use_zsharp: bool
    use_half: bool


def _setup_optimizer(
    config: TrainingConfig,
    model: nn.Module,
) -> tuple[torch.optim.Optimizer, str]:
    """Initialize the optimizer based on configuration.

    Args:
        config: Training configuration.
        model: Model whose parameters will be optimized.

    Returns:
        tuple: The optimizer and its resolved type
        (``SGD_OPTIMIZER`` or ``ZSHARP_OPTIMIZER``).
    """
    opt_config = config.optimizer
    opt_type = opt_config.type
    params = list(model.parameters())
    lr = float(opt_config.lr)
    momentum = float(opt_config.momentum)
    wd = float(opt_config.weight_decay)

    if opt_type == SGD_OPTIMIZER:
        optimizer: torch.optim.Optimizer = optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=wd
        )
        return optimizer, SGD_OPTIMIZER

    # ZSharp optimizer
    optimizer = ZSharp(
        params,
        base_optimizer=optim.SGD,
        rho=float(opt_config.rho),
        lr=lr,
        momentum=momentum,
        weight_decay=wd,
        percentile=int(opt_config.percentile),
    )
    return optimizer, ZSHARP_OPTIMIZER


def _run_train_step(
    ctx: TrainingContext,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Perform a single training step.

    Returns:
        tuple: The loss and the model predictions from the first forward
        pass, so callers can compute training accuracy without re-running
        the model.
    """
    if ctx.use_zsharp:
        # ZSharp two-step training
        outputs = ctx.model(x)
        loss = ctx.criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ctx.model.parameters(), MAX_GRADIENT_NORM
        )
        zsharp_opt = cast("ZSharp", ctx.optimizer)
        zsharp_opt.first_step()
        ctx.criterion(ctx.model(x), y).backward()
        zsharp_opt.second_step()
    else:
        # Standard SGD training
        ctx.optimizer.zero_grad()
        outputs = ctx.model(x)
        loss = ctx.criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ctx.model.parameters(), MAX_GRADIENT_NORM
        )
        ctx.optimizer.step()
    return float(loss.item()), outputs.detach()


def _validate(
    ctx: TrainingContext,
    loader: torch.utils.data.DataLoader[torch.Tensor],
) -> tuple[float, float]:
    """Evaluate model on a dataset."""
    ctx.model.eval()
    correct, total, total_loss = 0, 0, 0.0
    pbar = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(ctx.device), y.to(ctx.device)
            if ctx.use_half:
                x = x.half()
            outputs = ctx.model(x)
            loss = ctx.criterion(outputs, y)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            pbar.set_postfix({"Acc": f"{100 * correct / total:.2f}%"})
    acc = 100 * correct / total if total > 0 else 0.0
    return acc, total_loss / len(loader) if len(loader) > 0 else 0.0


def _run_epoch(
    ctx: TrainingContext,
    epoch: int,
    loader: torch.utils.data.DataLoader[torch.Tensor],
) -> tuple[float, float]:
    """Run a single training epoch."""
    ctx.model.train()
    epoch_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")

    for x, y in pbar:
        x, y = x.to(ctx.device), y.to(ctx.device)
        if ctx.use_half:
            x = x.half()
        loss, outputs = _run_train_step(ctx, x, y)
        epoch_loss += loss
        correct += (outputs.argmax(dim=1) == y).sum().item()
        total += y.size(0)
        pbar.set_postfix({"Loss": f"{loss:.4f}"})
    return epoch_loss / len(loader), 100 * correct / total


def _save_results(
    results: ExperimentResults,
    dataset_name: str,
    model_name: str,
    opt_type: str,
) -> None:
    """Save results to a JSON file."""
    path = Path(RESULTS_DIR)
    file_path = path / f"zsharp_{dataset_name}_{model_name}_{opt_type}.json"
    path.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as f:
        json.dump(results.model_dump(), f, indent=2)


def _init_components(
    config: TrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer, str]:
    """Initialize model and optimizer components."""
    ds_name = config.dataset
    if ds_name not in DATASET_METADATA:
        error_msg = f"Unknown dataset: {ds_name}"
        raise ValueError(error_msg)

    classes = cast("int", DATASET_METADATA[ds_name]["num_classes"])
    model_name = config.model
    model = get_model(model_name=model_name, num_classes=classes).to(device)
    optimizer, opt_type = _setup_optimizer(config, model)
    return model, optimizer, opt_type


def _prepare_training(
    config: TrainingConfig,
    device: torch.device,
) -> tuple[
    TrainingContext,
    tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]],
    int,
    str,
]:
    """Prepare training context and loaders.

    Returns:
        tuple: Training context, data loaders, epoch count, and the
        resolved optimizer type.
    """
    cfg = config.train
    m, opt, opt_type = _init_components(config, device)
    use_zsharp = opt_type == ZSHARP_OPTIMIZER
    uh = bool(device.type == "mps" and cfg.use_mixed_precision)
    if uh:
        m = m.half()
    ctx = TrainingContext(
        m, opt, nn.CrossEntropyLoss(), device, use_zsharp, uh
    )
    ldrs = get_dataset(
        dataset_name=config.dataset,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=cfg.pin_memory,
    )
    return ctx, ldrs, int(cfg.epochs), opt_type


@dataclass(frozen=True)
class TrainingHistory:
    """Accumulated per-epoch metrics and final results."""

    train_losses: list[float]
    train_accuracies: list[float]
    test_accuracies: list[float]
    final_test_accuracy: float
    final_test_loss: float
    total_training_time: float


def _create_results(
    config: TrainingConfig,
    ctx: TrainingContext,
    history: TrainingHistory,
    opt_type: str,
) -> ExperimentResults:
    """Consolidate results into dictionary."""
    return ExperimentResults(
        config=config,
        final_test_accuracy=history.final_test_accuracy,
        final_test_loss=history.final_test_loss,
        train_losses=history.train_losses,
        train_accuracies=history.train_accuracies,
        test_accuracies=history.test_accuracies,
        total_training_time=history.total_training_time,
        device=str(ctx.device),
        optimizer_type=opt_type,
    )


def train(config: TrainingConfig) -> Optional[ExperimentResults]:
    """Train a model using the provided configuration."""
    set_seed(DEFAULT_SEED)
    device = get_device(config)
    ctx, (train_ldr, test_ldr), epochs, opt_type = _prepare_training(
        config, device
    )
    start_time = time.time()
    l_list, t_list, v_list = [], [], []

    try:
        for epoch in range(epochs):
            e_loss, a = _run_epoch(ctx, epoch, train_ldr)
            va, _ = _validate(ctx, test_ldr)
            l_list.append(e_loss)
            t_list.append(a)
            v_list.append(va)
            logger.info(
                "Epoch %d: Acc: %.2f%%, Test: %.2f%%", epoch + 1, a, va
            )
    except KeyboardInterrupt:
        return None

    final_acc, final_loss = _validate(ctx, test_ldr)
    res = _create_results(
        config,
        ctx,
        TrainingHistory(
            train_losses=l_list,
            train_accuracies=t_list,
            test_accuracies=v_list,
            final_test_accuracy=final_acc,
            final_test_loss=final_loss,
            total_training_time=time.time() - start_time,
        ),
        opt_type,
    )
    _save_results(res, config.dataset, config.model, opt_type)
    return res
